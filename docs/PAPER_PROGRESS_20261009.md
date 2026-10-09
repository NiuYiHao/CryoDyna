# CryoDyna-optpose 论文执行记录：2026-10-09

状态：第一步成像/局部姿态诊断已完成，优化耦合实验进入下一工作包。

执行分支：codex/optpose-paper-20261107。独立工作目录：/home/nyh/.codex/worktrees/optpose-paper-20261107/Cryodyna。

新程序提交：b2f6ea99792eead4af3fb32047894e06b8f19456。入口：scripts/calibrate_1ake_forward.py。

## 已完成的实验

固定128个开发粒子，覆盖全部50个构象，seed=101，CPU4线程。使用共同参考构象50及逐粒子真值构象，分别生成原始密度Fourier投影与残基GMM投影。采用训练使用的v2 CTF、sigma=2 Å/残基电子权重、图像mask及相关损失，比较4 Å与8 Å低通条件。沿x/y/z分别采样−20、−10、−5、0、5、10、20°，保存43,008条原始记录。

| 投影模型 | 8 Å：真值pose的平均cosine | 4 Å：真值pose的平均cosine |
| --- | ---: | ---: |
| 共同参考密度 | 0.235217 | 0.114035 |
| 共同参考残基GMM | 0.230824 | 0.119259 |
| 逐粒子真值密度 | 0.302695 | 0.148053 |
| 逐粒子真值残基GMM | 0.294895 | 0.152358 |

在8 Å条件下，真值密度的384条观测图像轴向局部探针中，46.615%的最低训练损失落在0°；共同参考密度对应24.479%。GMM对应真值结构41.667%、共同参考22.656%。这些比例描述离散局部损失的形状；每个粒子的三个轴探针共享同一观测，数据解释采用粒子层级。

同模型生成的无噪声目标共有3,072条局部探针，全部在0°取得最低损失。三项数值测试通过：非中心密度identity投影、90°旋转约定、逐粒子相关损失与训练定义一致。50张输入密度的XYZ像素均为1 Å。

## 当前判断与下一步

共同参考结构与逐粒子真值结构之间的差异影响图像一致性；当前观测噪声也使部分局部损失最低点偏离真值。密度投影与残基投影的已测一致性接近，当前证据提示优先控制参考结构偏差和噪声驱动的pose更新。该诊断使用oracle结构和pose，结论范围为开发粒子的前向及局部损失，准确pose恢复和RMSD提升继续通过训练实验验收。

下一工作包为：真值pose固定/结构可学习、真值结构固定/pose可学习、参考结构固定/pose可学习、联合学习四组对照；统一输入和预算，检查pose与结构误差的配对变化。随后比较独立pose table、分块更新与信赖域，优先通过0°稳定及5°/10°的联合改善门槛。

## 复现和数据位置

原始结果根：/home/nyh/tools/Cryodyna/output/paper_20261107/step1_20261009/。n128/保存首次运行；n128_committed/保存已提交程序版本的独立重跑。每次输出particle_ids.npy、probe_metrics.csv、summary.json；summary记录输入及源码SHA256、粒子/构象数、参数、运行版本、真实耗时。

```bash
cd /home/nyh/.codex/worktrees/optpose-paper-20261107/Cryodyna
PYTHONPATH=. /home/nyh/anaconda3/envs/cryodyna/bin/python scripts/calibrate_1ake_forward.py \
  --dataset-root /home/nyh/tools/Cryodyna/tutorial_data_1ake \
  --output /path/to/new_output --particles 128 --seed 101 --device cpu --threads 4
```

计算分工继续采用BRCA1–p53项目Slurm集群＋本机；本轮诊断在本机CPU完成。全量训练和多种子任务按集群环境预检与冻结矩阵安排。
