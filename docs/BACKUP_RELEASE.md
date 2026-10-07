# 完整项目备份与恢复

Git 仓库保留可浏览的源码、配置、实验元数据和恢复脚本；由于完整项目包含模型、数据、缓存、日志和正式产物，体积远超普通 Git 的适用范围，因此全部工作目录另以 GitHub Release 附件的分卷归档保存。

发布的每个分卷最大约 1.8 GiB，并附带 `SHA256SUMS`。分卷归档包含原始 `qwen_handwriting_alignment/` 工作目录中的全部项目文件（不包含 Git 元数据目录 `.git/`）；不使用软链接或硬链接。

## 恢复

1. 下载同一 Release 下的全部 `qwen_handwriting_alignment_full_20261008.tar.part-*` 分卷，以及 `SHA256SUMS`。
2. 将这些文件放入同一空目录。
3. 校验并解包：

   ```bash
   sha256sum -c SHA256SUMS
   cat qwen_handwriting_alignment_full_20261008.tar.part-* | tar -xf -
   ```

   或使用本仓库的 `scripts/restore_full_project.sh`。

4. 在恢复出的项目目录中，根据 `requirements.lock.txt` 重建外部 Python/conda 环境；环境本身不在项目归档内。

恢复后可运行：

```bash
python scripts/run.py --log audit_project -- python scripts/audit_project.py --verify-hashes
```

确认模型、数据、源码和正式产物的记录一致。
