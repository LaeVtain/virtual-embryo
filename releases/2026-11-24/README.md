# 2026-11-24 模型

本版包含 T1 心脏单核表达、T2 整胚插值、T2 心脏外推和 T3 β-catenin 扰动四项预测。仓库保存拟合参数、基因面板和校验清单。原始赛事 H5AD 从队伍有权访问的赛事数据中取得。[版本比较](../../results/2026-11-24.md) · [文件清单与 SHA-256](manifest.json) · [数据使用规则](https://virtualembryo.ai/challenge/rules)

## 从仓库运行

在仓库根目录执行以下命令。复现环境使用 Python 3.12.14。

```bash
python -m pip install -c releases/2026-11-24/constraints.txt -e ".[dev]"
```

**1. 准备 T1 的三份公开参考。** 以下命令从 GEO 和 UCSC 准备参考文件，统一写到 `data/public-references/`。已有相同版本的三份 H5AD 时，可以直接在下一步指定它们所在的目录。

```bash
python scripts/prepare_t1_heart_reference.py --panel releases/2026-11-24/panels/T1__test.genes.txt --output-dir data/public-references
python scripts/prepare_t1_nuclear_reference.py --panel releases/2026-11-24/panels/T1__test.genes.txt --output-dir data/public-references
python scripts/prepare_t1_lineage_reference.py --panel releases/2026-11-24/panels/T1__test.genes.txt --cache data/lineage-cache --output data/public-references/lineage_reference.h5ad
```

**2. 准备赛事输入。** 将下表文件放入 `data/challenge/`，保留原文件名。两步合计使用 12 份不同的 H5AD。

| 用途 | 文件 |
|---|---|
| 四项推理所需的九份输入 | `E8.5_RNA.h5ad`、`E10.5_RNA.h5ad`、`E6.75.h5ad`、`E7.25.h5ad`、`E7.5.h5ad`、`E8.0.h5ad`、`E8.75.h5ad`、`E9.5.h5ad`、`E10.5.h5ad` |
| 首次重建 T3 时另外需要 | `E8.25_late.h5ad`、`E8.5.h5ad`、`E9.5_mab21l2_ko.h5ad` |

**3. 装配参数并生成四项预测。** 装配脚本校验输入，从五份实测空间文件的 `layers/log1p` 重建 T3 的细胞参考库，并核对每项模型文件。输出目录可以是新路径，也可以是此前装配完成的目录。已有配套离线包时，可将解压目录直接传给第二条命令的 `--assets`。

```bash
python scripts/prepare_release_assets.py --data data/challenge --references data/public-references --output data/assets-2026-11-24
python run_models.py --version 2026-11-24 --data data/challenge --assets data/assets-2026-11-24 --output outputs/2026-11-24
```

输出为四份 H5AD。每份文件的细胞数、基因数和参考环境中的 SHA-256 见清单的 `predictions` 字段。T1 参考若仅因 HDF5 写入版本不同而改变文件字节，装配脚本还会核对实际用于模型的表达、细胞、基因及文库标签。[校验清单](manifest.json)

## 参数来源

| 模块 | 参考和拟合内容 |
|---|---|
| T1 | E14.5 整细胞、野生型单核及发育谱系对齐到 32,285 个基因。仓库保存心房和心室变化参数以及芯片时间趋势。[GSE230531](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE230531) · [GSM8394736](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394736) · [UCSC 早期心区](https://cells.ucsc.edu/?ds=mouse-cardiac+wtAll) · [GSE193346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193346) · [GSE28389](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28389) |
| T2 整胚 | 从爱丁堡大学 EMA9、EMA10、EMA21、EMA24 胚胎解剖模型提取组织尺度。[EMA9](https://doi.org/10.7488/ds/2152) · [EMA10](https://doi.org/10.7488/ds/2141) · [EMA21](https://doi.org/10.7488/ds/2144) · [EMA24](https://doi.org/10.7488/ds/2149) |
| T2 心脏 | 拟合心室时间趋势、细胞状态变化、局部组织表达和跨测量方法的成熟趋势。三维几何取自 Neil Horner（MRC Harwell Institute）的 LAMA E14.5 图谱，对心脏标签取样并计算半径。[GSE51483](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51483) · [MOSTA](https://db.cngb.org/stomics/mosta/) · [GSE150817](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150817) · [GSM7999705](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7999705) · [LAMA 原图谱](https://doi.org/10.5281/zenodo.4559800) |
| T3 | 赛事野生型和 Mab21l2 供体数据用于估计跨基因扰动响应。仓库保存 RNA 响应系数、阶段均值及逐基因分位数。包含实测细胞行的两个矩阵由步骤 3 在本地重建。[赛事数据条款](https://virtualembryo.ai/challenge/terms) |

心脏几何参数沿用 LAMA 图谱的 Creative Commons Attribution 4.0 International 许可，处理包括选择标签 95、抽样和计算参考半径。[原图谱与作者](https://doi.org/10.5281/zenodo.4559800) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

赛事数据由加州大学圣迭戈分校 Neil Chi 教授团队和 Qingquan Zhang 博士提供。感谢 Virtual Embryo Challenge 组织者及各公开图谱作者。
