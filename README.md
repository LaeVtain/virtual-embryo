# Virtual Embryo

当前维护版本为 **2026-11-24**。一次运行可生成 T1 心脏单核表达、T2 整胚插值、T2 心脏外推和 T3 β-catenin 扰动四份预测。[参数与复现步骤](releases/2026-11-24/README.md) · [本地回放与官方已提交成绩](results/2026-11-24.md)

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
scripts/prepare_release_assets.py  从版本参数和本地实测参考装配推理资产
releases/                  分版本的参数、基因面板与 SHA-256
run_models.py             从九份原始输入生成四项冻结版预测
results/                  冻结版的本地得分及与历史版本的对照
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

首次从仓库运行时，按版本说明准备公开参考和赛事 H5AD，再执行：

```bash
python scripts/prepare_release_assets.py --data data/challenge --references data/public-references --output data/assets-2026-11-24
```

已有离线模型包可直接使用解压目录。需要旧神经模型的训练权重时，可从原项目工作区导入：

```bash
python scripts/import_assets.py --workspace /path/to/original/workspace --destination models
```

| 任务 | 推理输入 |
|---|---|
| T1 | 官方 E10.5 源 RNA、E8.5 校准 RNA、E14.5 心脏及野生型谱系参考 H5AD、整胚芯片变化表与面板 |
| T2 | 整胚使用四个已知日龄的空间样本和细胞类型，心脏使用两个阶段及心室、组织切片和三维解剖参考。支持 H5AD 和 NPZ |
| T3 | 同龄野生型群体 NPZ（`x`、`c`、`genes`）。支持 `Mab21l2`、`Gata4`、`bcatenin` |

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

默认运行 2026-11-24 版。将两份 RNA H5AD 和七份空间 H5AD 放在同一输入目录，执行以下命令生成四份预测。文件名见[版本说明](releases/2026-11-24/README.md)。

```bash
python run_models.py --version 2026-11-24 --data data/challenge --assets data/assets-2026-11-24 --output outputs/2026-11-24
```

`--assets` 指向包含 `models/` 和 `panels/` 的目录。在离线模型包内运行时可省略该参数。复现历史版本可指定 `--version 2026-11-23-r2`、`--version 2026-11-23` 或 `--version 2026-11-19`，各用一个输出目录。

单项运行的参数、输入与历史训练入口见各任务文档。[T1 单核表达](docs/t1.md) · [T2 空间预测](docs/t2.md) · [T3 扰动预测](docs/t3.md)

### 评估、导出和测试

评估输入包含 `x`、`genes`，T2 另需 `c`；也支持 H5AD。`--reference` 在 T1/T2 中使用源时刻群体，在 T3 中使用匹配的 WT。

```bash
python -m virtual_embryo.evaluate --task T2 --prediction outputs/heart.npz --truth data/heart_target.npz --reference data/heart_input.npz --output outputs/heart_metrics.json
python -m virtual_embryo.export --input outputs/heart.npz --output outputs/heart.h5ad
python -m pytest -q
```

T1 的 `lineage_forecast` 直接写出 H5AD。旧神经模型的预测目录可通过 `export --input outputs/t1` 导出。官方提交需设置 `SUBMIT_TOKEN`，命令见 [提交说明](docs/submission.md)。

评估默认输出原始指标。已有对应标本的计分上下限时，可增加 `--anchors data/anchors.json`。JSON 以指标名称为键，每项含 `floor` 和 `ceiling`。结果中的 `calibrated` 给出本组标尺的总分和各项得分。[空间计分与实例](results/2026-11-24.md)
