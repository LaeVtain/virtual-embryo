# 官方提交接口

客户端固定访问 `https://api.virtualembryo.ai`，使用环境变量 `SUBMIT_TOKEN`。可用 `python -m virtual_embryo.challenge --env-file /path/to/.env ...` 显式读取凭据文件。

先读取当前阶段、榜单、配额和历史：

```bash
mkdir receipts
python -m virtual_embryo.challenge inspect --out receipts/inspect.jsonl
```

以 T2 heart 的测试外推为例：

```bash
python -m virtual_embryo.challenge format-only outputs/heart.h5ad --task T2 --board T2:heart:test_extrap --setting heart --mode extrap --out receipts/heart-format.jsonl

python -m virtual_embryo.challenge submit outputs/heart.h5ad --task T2 --board T2:heart:test_extrap --setting heart --mode extrap --confirm-final --format-receipt receipts/heart-format.jsonl --out receipts/heart-submit.jsonl

python -m virtual_embryo.challenge status SUBMISSION_ID --out receipts/heart-status.jsonl
```

T1 测试使用 `--task T1 --board T1:test`；T2 embryo 测试插值使用 `--task T2 --board T2:embryo:test_interp --setting embryo --mode interp`。其他榜单以 `inspect` 的当前结果为准。

客户端根据榜单派生 `split`，正式测试附带 `confirm_final=true`。提交前记录文件 SHA-256 和幂等键，响应后核对服务端文件标识与榜单。网络中断后用原回执的 `--retry-from` 重用幂等键；重试时须保持文件及提交参数一致。回执路径每次使用新文件。测试只模拟接口调用，运行 `pytest` 不会产生真实提交。
