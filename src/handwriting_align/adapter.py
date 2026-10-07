"""Memory-bounded native Qwen-Image 2.1 loading; algorithms stay upstream."""

from __future__ import annotations

from contextlib import contextmanager
import torch
from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig, QwenImage21Pipeline
from transformers import BitsAndBytesConfig as TextBitsAndBytesConfig
from flow_factory.models.configured_image_output import ConfiguredImageOutputCodec
from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter


class _QwenImage21RGBAOutputCodec(ConfiguredImageOutputCodec):
    """Keep Qwen-Image 2.1's official RGBA VAE boundary explicit."""

    @staticmethod
    def _validate_pixel_values(
        pixel_values: object,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        """Validate the Qwen 2.1 RGBA tensor before VAE encoding.

        The shared configured-image codec is deliberately RGB-only. Qwen-Image
        2.1 instead converts its RGB training target to RGBA immediately before
        its native VAE, so applying the shared RGB check rejects a valid target.
        """
        source = "Qwen-Image 2.1 image_processor.preprocess"
        expected_shape = (batch_size, 4, height, width)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"{source} expected a torch.Tensor, received {type(pixel_values).__name__}")
        if not pixel_values.is_floating_point():
            raise TypeError(f"{source} expected a floating tensor, received {pixel_values.dtype}")
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(f"{source} expected RGBA BCHW shape {expected_shape}, received {tuple(pixel_values.shape)}")
        if not bool(torch.isfinite(pixel_values).all()):
            raise ValueError(f"{source} contains non-finite values")


class SingleGPUQwenAdapter(QwenImage21Adapter):
    """Use identical frozen 8-bit bases for baseline, SFT, and GRPO."""

    def load_pipeline(self) -> QwenImage21Pipeline:
        if self.accelerator.num_processes != 1:
            raise ValueError("This memory-bounded experiment is explicitly single-GPU")
        quantization = PipelineQuantizationConfig(quant_mapping={
            "transformer": BitsAndBytesConfig(load_in_8bit=True),
            "text_encoder": TextBitsAndBytesConfig(load_in_8bit=True),
        })
        pipeline = self._load_diffusers_pipeline(
            QwenImage21Pipeline, self.model_args.model_name_or_path,
            # Flow Factory supplies ``dtype`` from component_load_dtypes.
            # Passing the old diffusers ``torch_dtype`` alias as well breaks
            # current diffusers, which intentionally rejects both arguments.
            low_cpu_mem_usage=True,
            quantization_config=quantization, local_files_only=True,
        )
        saved_tensors_to_cpu = self.model_args.extra_kwargs.get("autograd_saved_tensors_to_cpu", False)
        if not isinstance(saved_tensors_to_cpu, bool):
            raise ValueError(
                "model.autograd_saved_tensors_to_cpu must be a boolean when set, "
                f"received {saved_tensors_to_cpu!r}"
            )
        # This controls only where PyTorch keeps tensors saved for the *policy*
        # backward pass. Parameters, inputs, objective, and precision stay intact.
        self.autograd_saved_tensors_to_cpu = saved_tensors_to_cpu
        if saved_tensors_to_cpu:
            print("QWEN_POLICY_SAVED_TENSORS=cpu", flush=True)
        rollout_transformer_grad = self.model_args.extra_kwargs.get(
            "rollout_transformer_grad_mode", False
        )
        if not isinstance(rollout_transformer_grad, bool):
            raise ValueError(
                "model.rollout_transformer_grad_mode must be a boolean when set, "
                f"received {rollout_transformer_grad!r}"
            )
        # The Qwen INT8 transformer uses a measurably different numerical path
        # when autograd is disabled. For on-policy GRPO, rollout logits must be
        # numerically compatible with the subsequent policy replay. During a
        # rollout this enables gradients only inside each transformer call; the
        # outer no-grad context immediately detaches the velocity, so no graph
        # spans diffusion steps and no rollout gradients are accumulated.
        self.rollout_transformer_grad_mode = rollout_transformer_grad
        if rollout_transformer_grad:
            print("QWEN_ROLLOUT_TRANSFORMER_GRAD_MODE=enabled", flush=True)
        token_chunk_size = self.model_args.extra_kwargs.get("token_linear_chunk_size")
        legacy_mlp_chunk_size = self.model_args.extra_kwargs.get("mlp_forward_chunk_size")
        if token_chunk_size is not None and legacy_mlp_chunk_size is not None and token_chunk_size != legacy_mlp_chunk_size:
            raise ValueError(
                "model.token_linear_chunk_size and deprecated model.mlp_forward_chunk_size "
                "must match when both are set"
            )
        chunk_size = token_chunk_size if token_chunk_size is not None else legacy_mlp_chunk_size
        if chunk_size is not None:
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
                raise ValueError(
                    "model.token_linear_chunk_size must be a positive integer when set, "
                    f"received {chunk_size!r}"
                )
            pipeline.transformer.enable_forward_chunking(chunk_size=chunk_size, dim=1)
            print(f"QWEN_TOKEN_LINEAR_CHUNK_SIZE={chunk_size}", flush=True)
        return pipeline

    def _init_ref_parameters(self) -> None:
        super()._init_ref_parameters()
        if self.training_args.requires_ref_model and self.model_args.finetune_type == "lora":
            # post_init is called after the SFT checkpoint has been loaded and prepared.
            self.add_named_parameters("alignment_sft_reference", device="cpu", overwrite=False)

    def build_output_state_codec(self):
        # Retain the common contract/geometry validation, then replace only
        # the RGB tensor boundary with Qwen-Image 2.1's native RGBA boundary.
        super().build_output_state_codec()
        return _QwenImage21RGBAOutputCodec(self)

    @contextmanager
    def use_ref_parameters(self):
        if self.model_args.finetune_type == "lora" and self.training_args.requires_ref_model:
            # Do not disable LoRA here: that would reference the untuned base instead.
            with self.use_named_parameters("alignment_sft_reference"):
                yield
        else:
            with super().use_ref_parameters():
                yield
