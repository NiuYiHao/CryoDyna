# Synthetic 80S 三方法比较：来源与复现记录

更新：2026-09-30。当前阶段：复用其他任务已下载的数据及正在运行的 DRGN-AI 实验，补齐三方法统一评估。重复下载和附属校验进程已停止，部分文件保留供后续清理。

## 交付约定

方法：CryoDyna-optpose、DRGN-AI、Random SO(3)。主图沿用用户参考图的左学习曲线、右终点柱状图、顶部共享图例。训练使用图像和 CTF，真值 pose 进入独立评估环节。主图采用论文的 out-of-plane/viewing-direction angular error；完整 SO(3) geodesic error 另图报告。两者分别命名、计算和解释。

触发词：**曲线终点双联图**。

示例：`曲线终点双联图；dataset=80S；methods=Optpose, DRGN-AI, Random SO(3)`。

仓库模板入口：`scripts/plot_curve_endpoint.py`。保留源数据、图注、PNG/PDF/SVG和版式核验记录。

## 已核实来源

| 来源 | 已核实事实 |
| --- | --- |
| [正式论文](https://doi.org/10.1038/s41592-025-02720-4) | Nature Methods，2025-06-26 |
| [正式补充表 S1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41592-025-02720-4/MediaObjects/41592_2025_2720_MOESM1_ESM.pdf) | synthetic 80S：100,000 粒子、128 px、3.77 Å/px、homogeneous、30 epochs、4×A100、1 h 25 min |
| 正式论文 Fig. 1c，本地全文 `develop/artifacts/drgnai_reproduction_20260930/paper_final.txt:94` | 视线方向角误差，六次运行的均值与 SD；评估以正式论文为准 |
| [作者项目页](https://cryodrgnai.cs.princeton.edu/) | 页面文字写五次运行，与正式图注存在差异；公开图终点约 2°，此值为读图近似 |
| [预印本 v3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11160740/) | 方法细节的辅助来源，正式全文现已在其他任务中取得 |
| [输入数据](https://zenodo.org/records/14853270) | synthetic-80S.zip，31,462,007,302 bytes；500k 无噪声图像、CTF、真值旋转 |
| [输出数据](https://zenodo.org/records/14847271) | ZIP 目录覆盖其他实验；80S 曲线/结果仍待独立生成 |
| [发表日公开代码](https://github.com/ml-struct-bio/drgnai/tree/083c1f396022f5fb131e0cf08417b0cd6c6d2c47) | 已固定下载该源码；它与作者实验时的源码对应关系仍待核实 |

现有完整 ZIP：`/media/nyh/ce66dee3-f568-4265-9a9c-a9c9b34c3973/dataset/synthetic-80S.zip`；已有准备记录报告整包 MD5 与官方值相符。

实际训练根：`/media/nyh/Elements/Cryodyna/drgnai_80s_reproduction_20260930/`。`inputs/particles.mrcs` 含 100,000 张 128×128 图像，`inputs/ctf.pkl` 与其逐粒子对应。`evaluation_truth/rotations.pkl`、`indices.npy` 仅供评估。其 `preparation.json` 记录前 100k 粒子、Gaussian σ=0.5、加噪 seed=0。

本任务早期来源审计见 [audit.json](optpose_evidence/80s_reproduction/audit.json)，该文件保留早期下载状态。当前状态以本文和实际训练根为准。

DRGN-AI 的现有运行由 `develop/artifacts/drgnai_reproduction_20260930/run80s.sh` 管理。源码 d0872ff、六个种子、5 HPS + 25 SGD、batch 32/8/256，使用保持全局 batch 的 coordinate-chunk checkpointing。其他任务已保存前向/梯度校验。seed0 已完成 10k 随机 pose 预训练，正在 HPS epoch 0；训练输出由该任务继续管理。

## 影响严格复现的具体差异

1. **图像生成条件。** 预印本方法给出随机 SO(3)、零平移、lognormal 离焦量、加性白噪声 σ=0.5。官方输入包的 `ctf.pkl` 有 500,000 行、仅一种取值：defocus U/V 均为 15,000 Å。该差异需要沿生成代码核实。100k 粒子索引与噪声种子还需建立可追溯记录。
2. **生成器参数。** [CryoAI 官方生成器](https://github.com/compSPI/cryoAI/tree/5663246dd9e67add5c50156ba52d7ed3439748e7)公开 80S 128³、3.77 Å 的体积与 100k 生成入口。其示例配置采用 lognormal 离焦量（中心 2 μm、log-space σ=0.2）、平移标准差 3 Å；噪声实现为 `power_signal/snr`。DRGN-AI 的零平移和 σ=0.5 要单独配置、验证，并记录体积来源与幅值标定。
3. **训练轮数。** 固定源码 `src/reconstruct.py:391` 计算 `max(2, n_imgs_pose_search // N + 1)`；默认 500k 对 100k 粒子产生 6 个 HPS pass。`epochs_sgd` 另行相加。作者公开图描述 5 HPS + 后续 SGD、横轴到 30。因此照搬 `epochs_sgd=30` 会改变图示训练预算。实际配置须逐阶段核对。
4. **预训练。** `src/reconstruct.py:494` 将新实验的 `start_epoch` 设为 -1；负 epoch 运行固定随机 pose 的 10k 图像预训练。该阶段独立记录。
5. **硬件与 batch。** 论文表列 4×A100；本地为 RTX 4060 Ti 8 GB。官方 `multigpu` 将配置 batch 乘以 GPU 数。单卡缩小 batch 的结果须携带硬件适配说明；论文条件等价性另行验收。

## 后续验收依据

- 数据：记录图像、CTF、粒子索引、加噪与训练种子的校验值，保留全部定义好的评估粒子。
- DRGN-AI：固定源码、完成配置、每轮预测和日志；六次运行汇总主指标，检查约 2° 文献目标对应的指标和预算。实际误差决定准确复现的结论。
- Optpose：使用相同图像和评估粒子，记录结构先验、HPS 与训练预算。
- Random SO(3)：固定种子的 Haar 随机旋转，与其余方法共用评估口径。
- 出图：三方法都有真实观测值后输出 PNG/PDF/SVG、CSV、图注与来源记录；保留失败运行的记录。

当前源码、元数据、补充材料及模板的准备工作均已记录。80S 三方法准确度仍待实际训练与统一评估。

## 2026-09-30 实际运行补充

已从其他任务的临时目录找到现成参考密度 `/tmp/cryoAI_audit/mrcfiles/80S_128.mrc`，并保存到 `/media/nyh/Elements/Cryodyna/optpose_80s_reproduction_20260930/reference/`。来源为 CryoAI commit `5663246dd9e67add5c50156ba52d7ed3439748e7` 的同名体积，128³、3.77 Å；来源与 SHA256 记录在 `reference/provenance.json`。

新增 **Optpose fixed-volume control（固定密度参考变体）**：复用现有 Hopf 网格细分与 PoseTable，以固定外部密度的 Fourier slice 为前向模型，执行姿态搜索和 Adam pose 优化。该对照具有结构先验；原子形变/VAE 模型属于另一路径。图例明确使用 `Optpose (density prior)`，同时记录与 DRGN-AI 的先验差别。

64 粒子链路检查使用前 64 个粒子，HPS 初始化后平均视线误差 0.4068°，30 次 pose 优化后 0.1303°，完整 SO(3) 误差 0.1386°。训练输入仅含图像、CTF、参考密度；真值 pose 在单独评估程序中读取。该检查属于组件验证。

完整 100k 对照已在 CPU 上完成30轮，复算后视线方向误差为0.144339°、完整SO(3)误差为0.153260°。完整逐轮数据见 `optpose_evidence/80s_full_control/metrics.csv`。训练，独立目录 `volume_control_full/`，入口 `scripts/run_80s_volume_pose.py`。GPU 上的既有 DRGN-AI 实验继续运行。参考前向模型的中心 Fourier 切片、YX 平移相位、旋转梯度有限差分三项测试通过。

`scripts/watch_80s_figure.py` 已运行，跟随真实检查点自动更新比较。输出根为 `results/80s_three_method_comparison/volume_control/`。当三个方法都有实测数据时生成阶段图，准确度初筛结果同时写入状态文件；图注明确标记 interim。六次30轮完成后生成完整观测汇总，论文协议差异的审核状态单独保留。设置 `CRYODYNA_FIGURE_QA` 后，每次生成图自动检查 PDF 文字大小与碰撞，人工视觉审核待出图后完成。

当前运行日志及进程记录位于 `optpose_80s_reproduction_20260930/volume_control_full.log`、`volume_control_process.json`、`figure_watch.log`、`figure_watch_process.json`。单次确定性密度参考对照单独呈现；DRGN-AI 和随机对照按六个种子汇总。

## 已接好的三方法合图入口

Optpose 的 pose-free STAR 已由 `scripts/prepare_80s_optpose_inputs.py` 生成，路径为 `/media/nyh/Elements/Cryodyna/optpose_80s_reproduction_20260930/inputs/particles.star`。它直接引用现有图像，完整保留 100k 粒子顺序和 CTF；训练输入仅包含图像与 CTF。参考结构和 80S 训练适配仍待完成。

六次 Haar 随机对照已完成真实评估，见 [原始记录](optpose_evidence/80s_reproduction/random_baseline.json)。其中保存真值及评估器 SHA256、逐种子全局对齐和角度统计。

`scripts/figure_80s_comparison.py` 直接读取现有 DRGN-AI checkpoints；Optpose 接口为 `seedN/evaluation/epochNNN.npz`，包含 `rotations` 和与共同输入完全一致的 `particle_ids`。epoch 0 表示初始化，图中训练 pass 从 1 起算。

```bash
python scripts/figure_80s_comparison.py \
  --drgnai-root "$DRGNAI_80S_ROOT" \
  --optpose-root "$OPTPOSE_80S_ROOT/volume_control_full" \
  --volume-control --output results/80s_three_method_comparison
```

评估器 `cryodyna/optpose/volume_metrics.py` 与绘图器 `scripts/plot_curve_endpoint.py`
随仓库提供。默认要求六个种子各30个完整训练pass；数据待齐时输出 `status.json`，
返回码2。`--progress` 根据已完成的检查点生成带阶段说明的图。
所有实测结果均保留，准确度 `<5°` 初筛结果单独报告。
论文协议差异逐项核验，科学结论依据实际曲线和训练条件。

主图使用视线角度，另附完整 SO(3) 图；原始逐种子数值、mean±sample SD 和 checkpoint SHA256 随图保存。随机对照使用六个固定种子的 Haar SO(3) 样本。

## 固定密度实验的完整复现命令

下列变量指向已有数据。`DRGNAI_80S_ROOT` 下含 `inputs/particles.mrcs`、
`inputs/ctf.pkl` 和 `evaluation_truth/rotations.pkl`；`VOLUME_80S` 指向
CryoAI的 `80S_128.mrc`（SHA256 `44543347b6eb70b03bf664a2f6b11e752736816d79dac2649d89e063ef42da26`）。
`OPTPOSE_RUN` 指向新的输出目录。使用 `--count 64` 复现本次全过程pilot；
`--count 100000` 运行全量条件。训练入口仅接收图像、CTF、密度。

```bash
python scripts/run_80s_volume_pose.py \
  --particles "$DRGNAI_80S_ROOT/inputs/particles.mrcs" \
  --ctf "$DRGNAI_80S_ROOT/inputs/ctf.pkl" --volume "$VOLUME_80S" \
  --output "$OPTPOSE_RUN" --count 64 --epochs 30 --seed 0 \
  --threads 2 --batch 32 --device cpu --lr .003 --record-search-stages
python scripts/plot_80s_pose_stages.py --run "$OPTPOSE_RUN" \
  --truth "$DRGNAI_80S_ROOT/evaluation_truth/rotations.pkl" \
  --output results/80s_stages
python scripts/evaluate_volume_pose.py --run "$OPTPOSE_RUN" \
  --truth "$DRGNAI_80S_ROOT/evaluation_truth/rotations.pkl" \
  --output results/80s_metrics
```

0.130269°对应64粒子pilot，0.144339°对应100,000粒子全量实验；两者均为
固定密度条件下的平均视线角。输入绝对路径与SHA256保留在各自的provenance文件。
固定外部密度提供强结构先验，DRGN-AI通过图像学习密度，两种条件分别解释。
正式Fig.1c和补充表S1对应30轮、六次运行；Fig.1e的速度测试统计5轮HPS和
100轮SGD。两个子图的预算分别记录。
