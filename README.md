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

权重放在 `models/t1`、`models/t2`、`models/t3`，输入数据放在 `data/`。这些目录需在本地准备。已有原项目工作区时，可导入权重：

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

复现 11 月 19 日冻结版时，将配套离线模型包中的 `models/` 和 `panels/` 放在仓库根目录。将 E8.5、E10.5 的两份 RNA H5AD，以及 E6.75、E7.25、E7.5、E8.0、E8.75、E9.5、E10.5 的七份空间 H5AD 放在同一输入目录。运行以下命令生成四个测试板的预测文件。

```bash
python run_models.py --data /path/to/challenge_data --output outputs/frozen
```

P3 测试使用以下入口。T1 的芯片、心脏及谱系参考准备命令见 T1 文档。T3 使用 `bcatenin-v2.npz`。具体数据准备和面板对齐命令见各任务文档。

```bash
python -m virtual_embryo.t1.lineage_forecast --source data/E10.5_RNA.h5ad --calibration data/E8.5_RNA.h5ad --reference data/heart_reference/E14.5_heart.h5ad --nuclear-reference data/nuclear_reference/E14.5_wt_nuclei.h5ad --lineage-reference data/lineage_reference.h5ad --chamber-reference data/chamber_reference.npz --cross-gene-dispersion --panel data/T1__test.genes.txt --trend data/geo/stage_effects.csv --interval 5 --source-time 10.5 --target-time 12.5 --cells 4096 --seed 101 --output outputs/T1__test.h5ad
python -m virtual_embryo.t2.embryo_interpolate --stage 6.75 data/E6.75.h5ad --stage 7.25 data/E7.25.h5ad --stage 7.5 data/E7.5.h5ad --stage 8.0 data/E8.0.h5ad --target-time 7.75 --cells 4096 --spatial-model sections --match-capture-depth --section-volume --sample-sections --panel data/T2__embryo__test_interp.genes.txt --output outputs/embryo.npz
python -m virtual_embryo.t2.heart_forecast --earlier data/E9.5.h5ad --later data/E10.5.h5ad --earlier-time 9.5 --later-time 10.5 --target-time 12.5 --cells 8192 --panel data/T2__heart__test_extrap.genes.txt --trend models/t2/heart_ventricular_trend.csv --anatomy models/t2/heart_e145.npz --spatial-rna models/t2/heart_spatial_niches.npz --state-reference models/t2/heart_state_095_145.npz --state-strength animal-horizon --output outputs/heart.npz
python -m virtual_embryo.t3.predict --condition bcatenin-v2 --input data/e875_wt_sample.npz --output outputs/bcatenin.npz
```

默认读取 `models/` 中的权重；运行新训练模型时，通过 `--model-dir` 指定按任务文档装配的模型目录。T1 的输出目录、T3 的输出文件须使用新路径。

### 评估、导出和测试

评估输入包含 `x`、`genes`，T2 另需 `c`；也支持 H5AD。`--reference` 在 T1/T2 中使用源时刻群体，在 T3 中使用匹配的 WT。

```bash
python -m virtual_embryo.evaluate --task T2 --prediction outputs/heart.npz --truth data/heart_target.npz --reference data/heart_input.npz --output outputs/heart_metrics.json
python -m virtual_embryo.export --input outputs/heart.npz --output outputs/heart.h5ad
python -m pytest -q
```

T1 的 `lineage_forecast` 直接写出 H5AD。旧神经模型的预测目录可通过 `export --input outputs/t1` 导出。官方提交需设置 `SUBMIT_TOKEN`，命令见 [提交说明](docs/submission.md)。
