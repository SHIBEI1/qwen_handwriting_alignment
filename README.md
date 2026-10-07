# Qwen-Image 手写文本生成：Base → LoRA SFT → LoRA Flow-GRPO

这是一个面向中文手写文本图像生成的三组对照实验项目。任务输入为目标文本与同一书写者的参考字图，输出为对应的手写风格文本图像。项目比较冻结的 Qwen-Image 2.1 基线、LoRA 监督微调（SFT）和从正式 SFT 权重继续优化的图像 Flow-GRPO。

> 本仓库中的源码、配置、实验元数据与恢复工具可以直接浏览；模型、数据、缓存、日志和全部正式产物通过 GitHub Release 分卷归档完整保存。恢复方法见 [docs/BACKUP_RELEASE.md](docs/BACKUP_RELEASE.md)。

## 实验设计

| 项目 | 固定设置 |
| --- | --- |
| 基础模型 | `Qwen/Qwen-Image-2.1`，固定 revision `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| 三组 | 冻结 INT8 基线；LoRA SFT；从正式 SFT LoRA 初始化的 LoRA Flow-GRPO |
| 可训练参数 | LoRA rank 16、alpha 32；基础 DiT 与文本编码器冻结并以相同 INT8 方式加载 |
| 数据 | 原生 CASIA 行级手写样本：512 条训练、32 条开发、64 条隔离测试；目标页面、文本与参考页按划分隔离 |
| 最终评测 | 三组同为 FP16 推理，256×768、16 步、guidance 1.0、64 个测试条件、两个固定随机种子（共 128 张图/组） |
| SFT | 3 个数据遍历 epoch，batch size 1、梯度累积 2，共 768 个优化器更新 |
| GRPO | 55 个预先固定的 rollout cycle；每 cycle 2 个条件、每条件 2 条候选轨迹、15 个 Flow-SDE 时间步；共 110 个优化器更新与 3,300 个组内相对优势观测 |

SFT 使用监督 flow-matching 损失。Flow-GRPO 从 SFT LoRA 快照开始，使用随机 Flow-SDE 轨迹、同条件候选组内相对优势、裁剪策略比率以及冻结 SFT 策略的 KL 约束进行更新；它不是将奖励简单加权到监督损失上的替代实现。

## 最终匹配 FP16 自动评测

| 组别 | 评估 OCR CER ↓ | 奖励 OCR CER ↓ | 精确匹配率 ↑ | 笔画风格代理 ↑ |
| --- | ---: | ---: | ---: | ---: |
| Base | 1.8739 | 1.9854 | 0.0547 | 0.8388 |
| SFT | 0.3021 | 0.2206 | 0.0156 | 0.8043 |
| Flow-GRPO | **0.2989** | **0.2117** | 0.0234 | 0.8036 |

SFT 相对基础模型的配对评估 CER 改变量为 -1.5718，95% bootstrap 区间为 `[-1.8354, -1.3357]`。Flow-GRPO 相对 SFT 的点估计改变量为 -0.00324（更低），但 95% 区间 `[-0.02314, 0.01803]` 跨越零。因此，本次固定测试集上 GRPO 的自动 CER 点估计最好，但不能将其表述为对 SFT 具有统计显著的普遍优势。

## 自动奖励与评测边界

- 训练奖励：`max(0, 1 - CER) × (0.85 + 0.15 × stroke_style_proxy)`；空白图奖励为零。
- CER 是 OCR 识别结果与目标文本之间的字符错误率；OCR 并不是真实人工标注。
- 笔画风格代理仅比较墨迹/笔画统计，不代表书写者身份，也不等同于人工偏好。
- 评估 OCR 与奖励 OCR 不同，但共享检测器；没有引入人工偏好标注、人工图像核查或多训练种子重复实验。

## 目录

- `configs/`：三组训练和评测的固定配置。
- `src/`、`scripts/`：模型适配、训练、评测、审计与恢复脚本。
- `docs/`：实验状态、依赖版本、独立性审计和完整备份说明。
- `models/`、`data/`、`runs/`、`test_artifacts/`、`logs/`、`cache/`、`vendor/`：完整版本位于对应 GitHub Release 分卷归档中。

## 运行与复现

项目环境不打包在目录内。恢复完整项目后，使用 `requirements.lock.txt` 在目标机器新建 Python/conda 环境；运行项目程序时统一通过：

```bash
python scripts/run.py --log PROGRAM_NAME -- COMMAND ...
```

恢复后可执行以下审计来验证模型、数据、源码和正式产物的记录：

```bash
python scripts/run.py --log audit_project -- python scripts/audit_project.py --verify-hashes
```

完整归档的校验和恢复流程见 [docs/BACKUP_RELEASE.md](docs/BACKUP_RELEASE.md)。
