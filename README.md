# Virtual Embryo models

Virtual Embryo 的 T1 时间预测、T2 时空预测和 T3 已知扰动预测。项目包含当前配方的训练、推理、局部评估、H5AD 导出和官方提交客户端。

## 安装

使用 Python 3.11 或 3.12：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

评估器固定为 `veckit==0.1.1`。原项目使用的 `core_metrics.py`、`shape_metrics.py` 与该版本的文件 SHA-256 一致。

## 模型资产

Git 仓库保存源码、测试和文档。权重、训练数据、预测文件和提交凭据存放在本地 `models/`、`data/`、`outputs/`、`receipts/`，由 `.gitignore` 排除。

从原研究工作区提取当前模型：

```bash
python scripts/import_assets.py --workspace /path/to/original/workspace --destination models
```

导出器记录源文件与结果哈希，移除推理不需要的优化器和随机数状态。T1 另外保留继续训练需要的完整网络参数。已有整理后的模型目录时，可直接复制到 `models/t1`、`models/t2`、`models/t3`，或设置 `VIRTUAL_EMBRYO_ASSETS` 指向 `models` 的位置。

## 训练与推理

各任务的数据契约、训练配方和参数见 [T1](docs/t1.md)、[T2](docs/t2.md)、[T3](docs/t3.md)。默认推理种子为 `101`。

```bash
# T1: 输入列顺序须与模型 genes.json 一致
python -m virtual_embryo.t1.predict --input data/t1_input.npz --output-dir outputs/t1 --source-time 10.5 --target-time 12.5

# T2: NPZ 中包含 x、c、genes
python -m virtual_embryo.t2.predict --setting heart --input data/heart_input.npz --output outputs/heart.npz --source-time 8.75 --target-time 9.5

# T3: 当前已训练的两个条件为 Mab21l2 和 Gata4
python -m virtual_embryo.t3.predict --condition Mab21l2 --input data/wt.npz --output outputs/mab.npz
```

## 本地评估

预测、真实目标和参考群体均可使用含 `x`、`genes` 的 NPZ 或 H5AD；T1 预测可直接传输出目录。T2 还需要坐标 `c`，H5AD 使用 `obsm["spatial_3D"]` 或 `obsm["spatial"]`。表达矩阵应采用对应挑战数据的预处理尺度。T1/T2 的参考群体是源时刻，T3 的参考群体是匹配的 WT。

```bash
python -m virtual_embryo.evaluate --task T2 --prediction outputs/heart.npz --truth data/heart_target.npz --reference data/heart_input.npz --output outputs/heart_metrics.json
```

评估器输出原始指标：T1 四项、T2 八项、T3 五项。`de_score`、`de_direction`、`occupancy_dice` 越高越好；MMD、variogram、D2 越低越好；尺度和严重程度的对数比以 `0` 为最佳。官方 0–100 分由服务端参考锚点归一化。`--max-cells 2000` 可显式限制每侧参与评估的细胞数，报告中同时记录采样参数。没有显著差异基因时，对应未定义指标写为 JSON `null`。

## 导出与提交

```bash
python -m virtual_embryo.export --input outputs/heart.npz --output outputs/heart.h5ad
# T1 示例：--input outputs/t1
```

提交客户端从进程环境或显式 `--env-file` 读取 `SUBMIT_TOKEN`。先执行格式校验；正式测试提交需要 `--confirm-final`，会使用一次测试机会。命令和回执规则见 [官方接口](docs/submission.md)。

## 验证

```bash
python -m pytest -q
```

整理版本的验证范围与精简记录见 [清理记录](docs/cleanup.md)。各任务的原始回放数据保留在本地研究工作区。
