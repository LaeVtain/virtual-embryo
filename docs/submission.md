# 官方提交

将 `SUBMIT_TOKEN` 写入环境变量，或在子命令前添加 `--env-file /path/to/.env`。输入为 H5AD，导出命令见 [README](../README.md#评估导出和测试)。

## 命令

```bash
mkdir receipts
python -m virtual_embryo.challenge inspect --out receipts/inspect.jsonl
```

T2 heart 测试外推：

```bash
python -m virtual_embryo.challenge format-only outputs/heart.h5ad --task T2 --board T2:heart:test_extrap --setting heart --mode extrap --out receipts/heart-format.jsonl

python -m virtual_embryo.challenge submit outputs/heart.h5ad --task T2 --board T2:heart:test_extrap --setting heart --mode extrap --confirm-final --format-receipt receipts/heart-format.jsonl --out receipts/heart-submit.jsonl

python -m virtual_embryo.challenge status SUBMISSION_ID --out receipts/heart-status.jsonl
```

## 参数

| 参数 | 说明 |
|---|---|
| `--task`、`--board` | 上传必填；任务名和榜单标识 |
| `--setting`、`--mode` | T2 必填；`heart/embryo` 与 `interp/extrap` |
| `--out` | 回执文件；上传必填，父目录须存在，文件须为新路径 |
| `--format-receipt` | 提交时使用同一预测文件的成功格式检查回执 |
| `--confirm-final` | 正式测试提交必填，会消耗一次测试机会 |
| `--retry-from` | 网络中断后使用原提交回执重试；文件和提交参数保持一致 |
| `--split` | 可选 `val/test`；默认从榜单派生 |

T1 测试使用 `--task T1 --board T1:test`；T2 embryo 测试插值使用 `--task T2 --board T2:embryo:test_interp --setting embryo --mode interp`。榜单是否开放以 `inspect` 的结果为准。
