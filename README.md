# Virtual Embryo

## 文件结构

```text
src/virtual_embryo/
├── t1/              时间预测：训练、推理、权重装配、历史指标评估
├── t2/              时空预测：数据准备、训练、分位校准、推理
├── t3/              扰动预测：状态训练、分位校准、推理
├── evaluate.py      T1/T2/T3 统一评估入口
├── export.py        预测结果转 H5AD
├── challenge.py     官方提交接口
├── population.py    群体数据读取与基因对齐
└── paths.py         模型路径配置
scripts/import_assets.py  从原项目导入模型资产
tests/                    测试
docs/                     各任务运行参数与提交命令
pyproject.toml            依赖和包配置
.env.example              环境变量示例
.gitignore                本地产物忽略规则
.gitattributes            文本换行规则
```

## 运行

### 安装

Python 3.11 或 3.12，在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### 准备模型和数据

权重放在 `models/t1`、`models/t2`、`models/t3`，输入数据放在 `data/`。这些目录需在本地准备。已有原项目工作区时，可导入权重：

```bash
python scripts/import_assets.py --workspace /path/to/original/workspace --destination models
```

| 任务 | 推理输入 NPZ |
|---|---|
| T1 | `x`：表达矩阵，列顺序与 `models/t1/genes.json` 一致 |
| T2 | `x`：表达矩阵；`c`：三维坐标；`genes`：基因名 |
| T3 | 同 T2，一次输入完整 WT 群体；支持 `Mab21l2`、`Gata4` |

训练数据准备及参数见 [T1](docs/t1.md)、[T2](docs/t2.md)、[T3](docs/t3.md)。

### 训练

```bash
# T1：fullELBO30 与三教师训练
python -m virtual_embryo.t1.train all --data-dir data/t1 --output-dir outputs/t1_train

# T2：心脏模型续训
python -m virtual_embryo.t2.train --setting heart --mode continue --prepared data/heart_prepared.joblib --output outputs/heart_continued.pt

# T3：Mab21l2 分位校准
python -m virtual_embryo.t3.train --wt data/mab_wt_train.npz --ko data/mab_ko_train.npz --condition Mab21l2 --output-dir outputs/t3_train
```

### 推理

```bash
python -m virtual_embryo.t1.predict --input data/t1_input.npz --output-dir outputs/t1 --source-time 10.5 --target-time 12.5
python -m virtual_embryo.t2.predict --setting heart --input data/heart_input.npz --output outputs/heart.npz --source-time 8.75 --target-time 9.5
python -m virtual_embryo.t3.predict --condition Mab21l2 --input data/wt.npz --output outputs/mab.npz
```

默认读取 `models/` 中的权重；运行新训练模型时，通过 `--model-dir` 指定按任务文档装配的模型目录。T1 的输出目录、T3 的输出文件须使用新路径。

### 评估、导出和测试

评估输入包含 `x`、`genes`，T2 另需 `c`；也支持 H5AD。`--reference` 在 T1/T2 中使用源时刻群体，在 T3 中使用匹配的 WT。

```bash
python -m virtual_embryo.evaluate --task T2 --prediction outputs/heart.npz --truth data/heart_target.npz --reference data/heart_input.npz --output outputs/heart_metrics.json
python -m virtual_embryo.export --input outputs/heart.npz --output outputs/heart.h5ad
python -m pytest -q
```

T1 导出时将 `--input` 改为 `outputs/t1`。官方提交需设置 `SUBMIT_TOKEN`，命令见 [提交说明](docs/submission.md)。
