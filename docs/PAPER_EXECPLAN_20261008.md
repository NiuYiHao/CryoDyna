# CryoDyna-optpose：30 天论文闭环与 main 发布任务书

版本：v1.1 · 建立于2026-10-08，资源更新于2026-10-09 · 截止：2026-11-07 23:59（Asia/Shanghai）

## 1. 交付目标与执行结论

目标是在 30 天内完成一次 main 分支提交，交付可复现程序、完整实验、原始数据索引、论文级图表与自洽的论文论证。科学验收同时要求：模拟及实验数据的有效覆盖达到对标工作的规模；所有预注册可比较任务的主指标严格优于适用的强对照。当前状态为“任务书就绪，科学门槛待攻关”。

第一行动：用 72 小时解决“如何把 pose 改善传递到结构改善”的机制判断，并冻结全量实验协议。与此同时建立数据清单、运行官方对照和实验数据评估链路。10 月 15 日启动冻结版本的全量矩阵，11 月 1 日进入证据复算与论文整合，11 月 7 日完成 main 发布验收。

性能提升属于待验证的科学目标。日期约束写入排程，实测结果写入验收；当前证据支持开展攻关，严格优势的可行性需要首周验证。发生差距时保留差距、原因和补救任务，paper_ready 的取值由结果决定。

首版矩阵目标为 8 类模拟系统、9 组实验单颗粒数据，并配置 1 组 cryo-ET/STA 扩展验收。数据数、粒子数、状态覆盖、噪声条件、种子数和验证独立性分别记账。10 月 10 日前逐项核对直接对标论文；论文覆盖表中的新增必需条目随即进入冻结矩阵。任务书里的目标数量属于计划，完成数量按经过评估的运行统计。

用户已指定按一个月目标推进，并于10月9日明确使用BRCA1–p53项目的集群资源和本机。资源协调作为执行工作流内部任务，运行时间由短测量估算；本计划的科学目标保持完整。

最终交付包：main commit SHA、源码和环境锁定文件、一键运行与一键重画入口、逐粒子/逐状态指标、全部种子及失败记录、6 张主图、补充图表、Typst/PDF 论文结果稿、数据与模型访问记录、逐项验收 JSON。

## 2. 当前起点与证据账本

2026-10-08 核对：本地及远端发布分支 codex/optpose-initial-release 均为 15ada55d23475b1bd7b6f0047e89ade27ecffdaa。开发版已提供扰动、HPS、学习曲线、历史输出导入、对比与预测回放。原始验收文档为 docs/OPTPOSE_ACCEPTANCE.md，26 项软件测试的历史记录随提交保存。

| 证据 | 已测结果 | 对论文的含义与下一门槛 |
| --- | --- | --- |
| 1AKE 扰动：50k 训练、40k 评估、20 轮、seed 1 | 15°：RMSD 1.572150→1.356881 Å；SO(3) 15→10.270291°。20°：2.445782→1.714370 Å；20→14.389820° | 两档支持联合改善；需覆盖所有正扰动、多种子及外部模拟系统 |
| 1AKE 小扰动 | 0°：Optpose 2.511655°；5° RMSD 1.145531→1.151190 Å；10° RMSD 1.333641→1.701989 Å | 姿态与结构解耦、零扰动稳定性是首要攻关 |
| 1AKE 无输入 pose 开发试验 | 103 评估粒子：Optpose 80.257181°、DRGN-AI 107.486008°、Random 132.128213° | 相同评估子集；先验与训练预算不同。准确恢复门槛为均值≤5°、P90≤10° |
| 80S 固定密度：100k 全粒子 | 搜索→Adam30：视线 0.492777→0.144339°；完整 SO(3) 0.566096→0.153260° | 固定结构先验下的 pose-only 链路证据 |
| 80S DRGN-AI：100k、seed 0、HPS 两轮 | 第二轮视线 0.481853°；完整 SO(3) 0.559051°；视线 P90 0.679659° | 10 月 1 日本地实测；完成六次全程训练及协议核验是下一门槛 |

80S DRGN-AI 最新表位于 develop/artifacts/drgnai_reproduction_20260930/80s_evaluation/summary.json，进展图为同目录上一级的 80s_measured_progress.png。该目录属于本地忽略目录，进入正式结果包时需要连同输入、源码版本和完整评估记录归档。其六次完整运行验收当前为 completed_runs=0、required_runs=6。

现有 1AKE 文件位于 tutorial_data_1ake/；生成设置位于 uniform_snr0-0001_ctf/settings.yaml。现有 80S 输入位于 /media/nyh/Elements/Cryodyna/drgnai_80s_reproduction_20260930/inputs/。首先复用已下载数据，其他数据逐项核对现有项目资产与共享目录。

历史 80S 0.130269°来自 64 粒子固定密度 pilot 的视线角；100k 对应值为 0.144339°。报告分别标注粒子数、先验和指标。现有原始图统一位于 docs/optpose_evidence/，保持历史证据原样。

## 3. 论文论证闭环与方法主张

拟定科学问题：在存在构象变化、噪声和成像误差时，如何联合估计粒子姿态与分子结构，使姿态纠正带来可验证的结构恢复，并在真实图像上保持一致？

论证链：输入 pose 误差影响结构恢复 → 分析姿态/形变的混淆机制 → 提出分阶段、具有几何约束的联合优化 → 用真值模拟验证机制和准确性 → 用独立实验图像验证重建与运动 → 用消融、复现和失败边界确定结论范围。

| 主张 | 核心实验 | 判定证据 |
| --- | --- | --- |
| C1：姿态优化能改善结构 | 1AKE 五档扰动、外部异质性模拟、相同结构先验对照 | SO(3) 与 Cα RMSD 两项同时改善；图像一致性同步报告 |
| C2：输入仅图像/CTF时可恢复 pose | 无输入 pose 的 80S 与异质性模拟；联合学习密度的公平对照 | 完整曲线、均值/P90/失败率、密度 FSC、独立评估 |
| C3：优势可迁移到实验数据 | 9 组 SPA 的独立半图与留出图像验证 | 数据集逐项的重建质量、预测质量及状态稳定性 |
| C4：改进来自所提机制 | 逐部件消融、前向模型校准、结构先验敏感性 | 同预算下移除部件后的配对变化及适用边界 |
| C5：达到同类实验覆盖且可复现 | 文献覆盖表、完整运行清单、独立重放 | 每条论文要求映射到具体运行、表格和图件 |

方法实现采取两个明确输入模式。P 模式使用共同参考结构，开展 pose/原子结构联合优化；V 模式输入图像和 CTF，联合学习密度与 pose。V 模式需要实现并验证训练适配；固定外部密度继续作为组件对照。P/V 共享经过验证的旋转、平移、CTF 与评估工具。

主分析层固定为fixed_density、shared_atomic_prior、image_ctf_only三类。每层独立列出基线、主指标、区间和结论；跨层结果用于展示先验敏感性。C2的证据全部来自image_ctf_only层。

候选技术路线：从粗到细的 HPS；姿态与构象分块更新；姿态步长信赖域或近端约束；低频初始化逐渐放开高频；依据图像证据的更新置信度。先以最少组件完成有效方案，再用消融证明必要性。每个候选保持固定预算与开发集，候选选择依据预注册验证指标。

论文创新验收要求对“已有 HPS+Adam”给出可测的新增机制与收益。模块复用记录来源、许可证、所改公式及代码位置。

## 4. 同类工作与公平比较契约

直接比较分为三条轨道。结构先验轨道：原版 CryoDyna、CryoSTAR、Optpose-P；补充 DynaMight、CryoSPHERE 的适用重建任务。图像/CTF轨道：DRGN-AI、Optpose-V、CryoAI及可复现的近期 pose 方法。密度/异质性轨道：DRGN-AI、cryoDRGN、CryoSPIRE，并在匹配数据上加入 RECOVAR、DynaMight。传统重建对照采用可运行的 RELION；现有 cryoSPARC 可用时附上版本与完整项目导出。Random SO(3)作为基本检验对照。

| 文献证据 | 已核实覆盖 | 本计划如何对齐 |
| --- | --- | --- |
| DRGN-AI，Nature Methods 2025 [R1] | 2 类模拟、6 组实验 SPA、1 组 STA；80S 角度曲线含6次重复 | 相同任务的数据、粒子数、训练阶段和指标逐项建表；完整论文覆盖附 STA |
| CryoSTAR，Nature Methods 2024 [R2] | 1AKE 模拟与4组实验：10180、10073、10059、10827 | 四组实验纳入；1AKE 使用已核实的50k图像与50状态 |
| CryoBench，NeurIPS 2024 [R3] | IgG-1D、IgG-RL、Spike-MD、Ribosembly、Tomotwin-100；额外噪声档 | 五类完整纳入；采用官方密度及异质性指标 |
| CryoSPIRE，2025 [R4] | 近期异质性/密度竞争方法 | 纳入匹配条件的强对照，10月10日前冻结论文版本、源码和配置 |
| CryoFastAR，ICCV 2025 [R5] | 预训练模型进行快速 pose 推断 | 单列预训练数据与测试重叠；适用条件下进行对比 |
| CryoDyna公开预印本 [R6]、DynaMight [R7] | 原模型和物理/形变建模对照 | 逐项补齐论文设置与数据来源；新增匹配数据进入覆盖表 |

同预算比较与作者推荐收敛设置各报告一张表。前者控制图像次数、训练预算和调参次数；后者保留每种方法的推荐训练长度。所有方法共用原始粒子、预处理、CTF、像素大小、评估频带与分组规则。冷启动、预训练和结构先验分别计入成本与条件。

“工作量达到同类”按向量验收：独立系统数、完整粒子数、状态与噪声覆盖、重复数、适用基线数、独立验证项。每一维均达到预注册对标值。粒子子集按原实验规模记账；同一系统的噪声变体记为条件数；额外训练轮数记为预算。

10 月 10 日形成 literature_coverage.csv：paper、task、dataset、N、conditions、replicates、protocol、source、our_run_ids、status。每个未知值先标 pending，再依据论文补充材料和实际文件补齐；coverage_pass 要求所有必需项有证据。

同日锁定baseline_eligibility.json：逐个dataset×task写入method、官方源码SHA、输入/先验、训练与调参预算、split/hash、适用性及理由、失败处理、主指标。适用基线的训练失败保留为失败和资源结果，准确率比较保持pending直至完成协议核查；适用性由任务定义与实现接口判定。冻结后的修改记录原因、时间及受影响的全部重跑任务。

CryoSTAR扩展图图注同时写“50,000总粒子、50状态、每状态5,000”，其数值关系存在冲突。本计划采用本地生成设置与实际图像数量核实50k总量；逐状态计数由STAR统计保存，文字冲突记录于来源表。

## 5. 全量数据与工作量矩阵

下表是首版必跑范围。final N、输入哈希、文件可用性及完整训练轮数于10月10日冻结。每个待核对值通过元数据补齐。目标是完整作者实验栈；经预注册的额外留出验证作为另一条评估轨道，论文协议复现表保留原有训练粒子定义。

| 模拟ID | 系统与规模 | 任务与评价 |
| --- | --- | --- |
| S1 | 1AKE，50k、50构象、现有 SNR=0.0001 | 0/5/10/15/20°；自定义数组；P模式RMSD+SO(3)；无pose恢复 |
| S2 | 80S，100k | V模式从头恢复；视线/SO(3)/平移/密度；P或固定密度单列 |
| S3 | DRGN-AI 1D motion，50k、50状态 | 强构象变化、pose/构象联合恢复 |
| S4 | CryoBench IgG-1D，100k/官方条件 | 官方3个噪声档；pose与异质性恢复 |
| S5 | CryoBench IgG-RL，完整官方栈 | 复杂非刚性变化；逐粒子密度FSC与结构指标 |
| S6 | CryoBench Spike-MD，完整官方栈 | 大规模构象覆盖；分布/邻域与密度指标 |
| S7 | CryoBench Ribosembly，完整官方栈 | 组成异质性；V模式/显式占据机制，状态与密度指标 |
| S8 | CryoBench Tomotwin-100，完整官方栈 | 多组分压力测试；V模式或混合模型能力门槛 |

| 实验ID | 系统 / 来源 | 作者使用规模或冻结动作 |
| --- | --- | --- |
| E1 | spliceosome，EMPIAR-10180 | DRGN-AI全栈327,490；CryoSTAR协议另行匹配 |
| E2 | tri-snRNP，EMPIAR-10073 | CryoSTAR使用栈及筛选索引逐项核对 |
| E3 | TRPV1，EMPIAR-10059 | 同上；膜蛋白及先验敏感性 |
| E4 | α-LCT，EMPIAR-10827 | 同上；小体系与连续运动 |
| E5 | assembling 50S，EMPIAR-10076 | 131,899；组成异质性 |
| E6 | SARS-CoV-2 spike，DRGN-AI来源 | 369,429；官方输入记录固定accession及哈希 |
| E7 | DSL1–SNARE，EMPIAR-11846 | 全栈214,511；75,854筛选子集单列 |
| E8 | V-ATPase，EMPIAR-10874 | 全栈267,216；177,481筛选子集单列 |
| E9 | ankyrin-1，EMPIAR-11043 | 710,437；20k验证子集单列 |
| E10 | M. pneumoniae70S，EMPIAR-10499，STA扩展 | 18,466颗粒；原协议41倾角、使用11倾角；以颗粒/断层为独立单位 |

9组SPA覆盖面超过上述直接对标工作的单篇SPA数量；完整论文覆盖的验收进一步逐任务对应，并包含STA扩展。DynaMight/其他论文若被纳入“全论文覆盖”主张，其额外系统同步进入必需表；匹配数据对比主张按对应交集报告。当前固定拓扑原子模型对组成变化的能力需要适配，S7/S8/E5提前列为高风险任务。

S2建立两个独立run family：S2-paper使用已核对的论文生成协议，S2-deposited使用现有固定CTF沉积栈；两者共享系统ID，各自报告结果与协议差异。E6的accession/官方归档版本/输入哈希为冻结必填字段。附件benchmark_matrix.csv目前为数据集级清单，方法SHA、预算、split、输入hash、run_ids由冻结manifest展开成作业级矩阵后进入覆盖验收。

全量最低重复数为每个适用的随机训练方法6个种子（0–5）；论文采用更多重复的条目沿用更高数目。确定性基线完整执行一次并说明确定性。每组实验至少包含Optpose、任务对应官方方法、一个独立强基线。实验独立半图每个种子训练两半；任务清单预计至少324个半图训练单元（9×3×6×2），另计模拟、扩展、消融及官方全栈协议运行。

1AKE扰动最小矩阵为2方法×5角度×6种子=60个全量运行；S1/S2/S3的无pose三方法比较至少54个运行，重复条目通过run_id去重。对工作量的最终结论依据冻结后的覆盖表和完成清单。

## 6. 指标、独立性与严格优势

模拟主要指标：完整SO(3)角度均值、P90、达到5°的比例；存在逐粒子原子真值时报告proper Kabsch后的Cα RMSD；所有密度方法报告官方定义的per-image FSC/AUC-FSC。视线角、平移误差、损失、每粒子耗时和显存作为分列指标。平移评价校正全局原点偏差，单位统一为Å和pixel。对称体采用预注册对称群下的等价姿态误差，并同时提供原始数值。

跨方法只比较共同定义的量：仅输出密度的方法参加密度评价；原子方法在共同原子集合上比较RMSD。真值pose/结构仅进入评估和受控扰动生成；P模式的共同参考结构拥有独立来源记录。训练日志记录真实优化器与参数组。

实验主要指标：独立半图FSC对应的分辨率与留出图像预测质量作为共同主指标。固定half-set、频带、mask生成规则、地图对齐及滤波；mask与共享结构先验的影响通过无mask结果和高频噪声替换校正核查。学习密度/pose在两半独立训练，初始化采用相同协议规定的低分辨率先验。采用原子参考时额外报告模型偏倚诊断。

留出评估按micrograph分组；STA按tomogram/颗粒分组。训练、调参和最终评估索引分离。留出粒子的pose/latent只使用预定低频或独立频率子集拟合，预测损失在隔离的高频集合计算；密度参数保持冻结。统一白化与CTF前向模型，报告标准化负对数似然或预注册的预测误差。异常值、可估计频带与分母规则在测试前冻结。

实验数据的已发表pose与原子模型作为参考一致性指标，真实pose和逐粒子结构精度由具有真值的模拟证据支撑。实验状态图的分类规则在开发集确定，跨半图匹配采用固定算法；给出全部预注册状态、population及重复稳定性。

G4强制包含：独立half-set、留出micrograph/tomogram、先验置换或截短、共享高频先验消融、噪声替换校验与训练/评价隔离审计。实验论文结论使用“重建质量、预测一致性、运动重复性”；参考一致性和真值精度分别列名。只有完整通过独立性检查的实验结果参加主指标优势判断。

严格优势的判定单元为“数据集×任务×适用基线×共同主指标”。误差指标定义差值baseline−ours，质量指标定义ours−baseline；每个必需单元要求均值差为正，且预注册的95%置信区间下界大于0。使用种子级配对差异；micrograph/粒子只用于嵌套不确定性估计。多比较采用同时置信区间或Holm校正，公开原始种子结果及敏感性分析。

6个种子是最小起点；10月10日前用独立pilot估计精度与所需重复数，冻结正式样本量及最大预算。正式测试期间按冻结设计收集全部结果。所有必需单元同时通过，strict_superiority_pass取true；汇总平均、最佳seed和单一成功case各自作为描述数据。

配对分析以同dataset、同split、同初始化随机条件的method差值为输入；各数据集分别推断。种子区间描述固定数据上的算法波动，micrograph/tomogram嵌套重采样描述样本波动；跨数据集汇总只作辅助分析。6为正式重复数下限，pilot决定满足目标区间宽度的最终数量。精度不足的比较保持pending，并显示区间宽度及完整样本数。

工程上的幅度目标单列：正扰动RMSD及SO(3)相对最强适用基线分别改善≥10%；无pose均值≤5°、P90≤10°；0°稳定性≤0.05°且结构退化处于预注册容差内。幅度目标是研究目标；对严格优势、准确恢复与零误差稳定性分别验收。

0°是稳定性对照；原始误差接近数值下限时采用预注册非劣容差，同时报告绝对变化。论文的“严格更优”主张指向正扰动和未知pose任务；0°稳定性单独列明。

## 7. 技术关键路径与首个72小时

第一天：冻结现有证据；建立dataset_manifest和文献覆盖表；统一粒子ID、旋转约定、手性、像素单位、CTF符号、中心/平移相位及频带。核查1AKE生成器与当前GMM/密度前向差异。用相同输入复算已存终点。各官方基线独立环境启动协议检查。

第二天：执行诊断矩阵：真值pose+可学习结构、真值结构+可学习pose、共同参考结构+可学习pose、两者联合学习；每种同时比较匹配生成器的无噪声、逐级加噪及现有1AKE图像。oracle仅作为可辨识性诊断。输出姿态误差—结构误差—图像残差的逐粒子关系，定位forward mismatch、结构先验偏差或优化耦合。

第三天：固定开发集上的最小候选比较：原MLP、独立pose table、分块更新、分块更新+信赖域。候选共享参考初始化、图像顺序与调参预算。通过0°稳定、5°/10°同时改善的候选进入完整训练；10°/20°用于检查较大偏差恢复。所有候选保留结果。

10月11日至14日：校准前向模型与联合优化，实现V模式；完成独立半图和留出评估适配；给所有数据建立可执行配置。达到同等信息条件的官方baseline是必需质量门槛，复现质量对齐后进入正式比较。

1AKE的已测oracle诊断误差约32.887°来自既有小样本条件。首周将检查原始图像生成与参考前向的一致性；若oracle在正确模型与全部评估协议下仍处于高误差，记录该SNR下的信息限制，并把该条件作为显式困难case。原准确恢复目标继续保留为待解决项，最终科学验收据实标记。

10月14日方法冻结。后续发现实现错误时记录修复SHA及受影响run_id并重跑；同一正式测试集上的自适应调参计入探索，重新划定隔离评估证据。新增结构分支、组成占据机制与STA输入适配各设独立测试，以控制关键路径。

10月14日三个能力门：V模式在图像/CTF-only输入完成训练、预测和真值隔离审计；组成模式在两个已知不同组分的模拟状态上输出可评价密度/占据及粒子对应；STA模式保持tilt-group和颗粒ID、完成CTF/倾角输入、半图评估与恢复。每个能力门保存最小运行、独立重放及pass/pending/fail状态。能力门完成后相应全量任务进入正式队列；P模式与固定密度证据维持各自归属。

## 8. 四周排程、分工与验收点

| 日期 | 任务 | 可核查交付与门槛 |
| --- | --- | --- |
| 10/08–10/10 | 证据盘点、诊断、文献/数据/指标冻结 | 版本清单、覆盖CSV、split/hash、oracle矩阵、候选结果、固定种子数 |
| 10/11–10/14 | 方法与评估实现、基线协议复现 | P/V模式、0°稳定、小扰动联合改善、half-map链路；方法冻结SHA |
| 10/15–10/21 | 全量模拟与首批实验并行 | S1–S8全矩阵进入运行；E1–E4半图与留出评估；逐日差距表 |
| 10/22–10/28 | 全量实验、组成/STA扩展、消融 | E5–E10；所有必需基线和全部seed完成；10/28训练收口 |
| 10/29–10/31 | 故障补跑、统计与强对照核验 | 100%必需run有状态；全部主指标表、置信区间、失败归因 |
| 11/01–11/04 | 6张主图、论文故事、独立重放 | 图源CSV/检查点逐项对应；Typst/PDF结果稿；clean-env复现 |
| 11/05–11/06 | 候选发布、main整合、外部复核 | candidate SHA、完整差异审查、最终数据锁、验收JSON |
| 11/07 | 缓冲与main提交 | 所有科学/覆盖/复现门槛通过后发布paper-ready main commit |

职责按可并行的工作包划分：方法与诊断；数据与官方基线；统一评估与统计；实验重建与生物解释；图表/论文/发布。每个工作包记录负责任务、文件所有权和依赖，探查代理用于独立核验，主任务负责方法决策与最终修改。

正式作业入队条件为manifest/split/hash锁定、方法adapter测试通过、官方baseline配置检查通过、统一评估链路通过、预算短测完成。待齐项继续以pilot状态推进。上述8类模拟、9组SPA与STA要求持续保留，完成率使用全部必需任务为分母。

计算资源已明确为BRCA1–p53项目使用的Slurm集群＋本机。10月9日只读核验：集群gpu1/gpu2/gpu3分区登记7个节点，每节点登记8张GPU；这些数目描述资源登记规模，可分配并发由账户配额、节点状态与队列决定。GPU具体型号/显存及PyTorch兼容性在计算节点环境预检时记录。本机实测为RTX 4060 Ti，8188 MiB显存。

| 执行位置 | 分配任务 | 调度与复现要求 |
| --- | --- | --- |
| BRCA1–p53同一Slurm集群 | 全量训练、官方baseline、多seed、实验独立半图、长时消融 | dataset×method×seed×half-set作业数组；按配额限制并发，保存环境/SHA/作业ID |
| 本机CPU与RTX 4060 Ti 8 GB | 代码开发、小样本诊断、数据预检、指标汇总、图表与Typst/PDF | 用小批次校验数值；批量评估按显存需求转入集群 |
| 共享输入与结果归档 | 复用已下载数据、传输所需输入与checkpoint | 输入hash一致；项目独立目录、日志、环境和作业前缀；本地保留精简证据 |

BRCA1–p53现有作业保持其运行安排；CryoDyna使用独立输出目录与作业前缀，通过Slurm申请资源。集群登录节点用于提交与轻量检查，训练与大批量计算进入计算节点。GPU资源登记类型含geforce/rtx，实测吞吐和显存决定分组调度；跨设备对照记录硬件，并保持相同有效batch、图像次数与收敛协议。

算力计划在10月10日前依据集群和本机的短测量填写：总GPU小时=各作业耗时×GPU数之和；需求并发数=剩余GPU小时/(剩余运行天数×24×可用率)。剩余天数按10月28日训练收口计算，故障预留25%。按关键路径排队，官方baseline、模拟主矩阵和实验半图并行调度，运行配置保持一致。

每天产生一次status.json：必需/完成/失败/排队run数、剩余预算、门槛状态、最新图件、风险与下一动作。此文档定义执行记录要求；调度器配置在实施阶段落地。

10月14日决策门：小扰动联合改善与V模式pose恢复决定技术风险。10月21日决策门：正式数据上的优势置信区间与实验链路决定补跑优先级。10月28日决策门：任何待完成任务进入逐项补救清单，截止日期和完整科学目标持续同时跟踪。

## 9. 程序接口、具体操作与恢复

现有入口scripts/benchmark_optpose.py转发到cryodyna/optpose/benchmark.py。复用geometry.py的旋转/RMSD，pose.py的PoseTable和细分搜索，volume_prior.py的Fourier前向，audit.py的预测回放，plotting.py的统一出图。现有接口继续支持旧版/opt输出目录、多简称、默认和自定义角度。

拟新增接口（实施任务）：scripts/paper_benchmark.py及configs/paper/，统一manifest、方法adapter、job账本、半图评估、统计和release gate。现有训练核心逐项接入；外部方法保存独立环境和源码SHA。此入口当前状态为planned。

| 接口 | 输入 | 预期产物与行为 |
| --- | --- | --- |
| validate-data | dataset manifest、split | 粒子/结构/CTF单位及哈希报告；缺项明确定位 |
| plan / run / resume | matrix、method、seed、预算 | 确定的run_id、dry-run清单、checkpoint与中断恢复 |
| evaluate | outdir、隔离truth或half-set | 逐粒子/逐状态及聚合指标、实际样本数 |
| compare | label=outdir列表 | 统一条件检查、学习曲线与终点图、统计差值 |
| audit-release | 锁定矩阵、结果索引 | coverage、strict_superiority、reproducibility、paper_ready |

当前可执行的接口核验与已有结果回放如下。执行环境为已安装项目依赖的cryodyna环境。第二条命令对历史矩阵复算，产出单独的本次审计文件。

现有acceptance命令的范围为1AKE的original/opt矩阵；--endpoint-only核验已存终点。论文全矩阵验收由新入口与schema实现后执行，两个审计范围分别记录。

```bash
python scripts/benchmark_optpose.py --help
python scripts/benchmark_optpose.py acceptance --endpoint-only \
  --outdir /media/nyh/Elements/Cryodyna/optpose_release_20260928/full \
  --output results/paper_20261107/baseline_replay.json
python scripts/benchmark_optpose.py compare \
  --runs Original=/media/nyh/Elements/Cryodyna/optpose_release_20260928/full/original \
         Optpose=/media/nyh/Elements/Cryodyna/optpose_release_20260928/full/opt \
  --output results/paper_20261107/baseline_figures
```

未来冻结入口的调用契约如下；实现完成时CLI帮助、配置schema与这些命令一同测试。

```bash
python scripts/paper_benchmark.py validate-data --manifest configs/paper/datasets.json
python scripts/paper_benchmark.py plan --matrix configs/paper/matrix.json --dry-run
python scripts/paper_benchmark.py run --matrix configs/paper/matrix.json --resume
python scripts/paper_benchmark.py evaluate --runs results/paper_20261107
python scripts/paper_benchmark.py audit-release --matrix configs/paper/matrix.json \
  --runs results/paper_20261107 --output docs/paper_results/acceptance.json
```

恢复规则：run_id由输入哈希、源码SHA、配置和seed确定；完整checkpoint含优化器、调度器、随机状态与粒子顺序。已完成作业进行checksum核验后复用。部分输出使用临时文件和原子替换，评估/重画可重复执行。环境或协议变化产生新run_id，既有证据保留。所有待比较运行来自明确版本集合。

## 10. 论文图表、主分支验收与发布

| 图 | 论证任务 | 必需内容 |
| --- | --- | --- |
| Fig.1 | 问题与方法 | pose/形变混淆、HPS→分块优化、P/V输入边界、信息隔离 |
| Fig.2 | 模拟准确性 | 1AKE五档loss/RMSD/SO(3)曲线，多seed终点，0°稳定性 |
| Fig.3 | 全流程pose恢复 | 80S网格→Adam完整曲线、DRGN-AI/Random对照、强先验控制单列 |
| Fig.4 | 跨系统与异质性 | S1–S8逐数据集结果、最强基线、误差区间、困难case |
| Fig.5 | 实验数据闭环 | 全部SPA主指标汇总、典型状态图、独立半图/留出验证；STA扩展 |
| Fig.6 | 机制与边界 | 部件消融、先验偏差、噪声/组成敏感性、效率与失败率 |

每图交付PNG/PDF/SVG、CSV、caption、source_record；曲线终点双联图沿用既定顶部共享图例、左曲线、右终点柱。图和表使用相同run_id及统计脚本。选取展示状态的规则事先固定，补充材料保留完整数据集表。

main验收为五个门槛的逻辑与：G1接口/输入隔离/重启可复现；G2论文覆盖表全部必需项完成；G3所有预注册共同主指标严格优势及准确恢复门槛；G4实验独立性与先验审计；G5全部结论可追溯到冻结代码、检查点、数据和图。门槛各自记录pass、pending或fail及证据路径。

工作分支建议codex/optpose-paper-20261107，从已审计发布版本创建隔离checkout。main当前已有独立历史；发布前检查共同祖先、差异和README入口，采用可审阅整合提交。保留其他任务的工作目录和修改，记录最终merge/commit与远端一致性。

大体积原始数据、模型与逐粒子数组存放持久归档；仓库保存访问地址、许可证、SHA256、恢复脚本和精简汇总。独立复核者在干净环境重算全部终点、随机抽取中间检查点并重画全图；核心训练至少在1AKE和80S各完整复跑一次。全部运行保留执行记录与复现容差。

11月7日以真实验收状态决定发布标签。全部门槛通过时提交paper-ready main版本；存在待达标项时完整记录deadline_gap与下一动作，任务目标继续标为待完成。提交动作和科学达标分别审计。

## 11. 进度、决策与风险记录

Progress / 2026-10-08：已完成当前提交与原始结果核验；已建立首版论文论证、实验矩阵、统计与发布契约；已生成任务书。下一阶段为输入清单、文献补充材料逐项核验、72小时诊断与官方对照运行。

Surprises & Discoveries：1AKE正扰动的角度改善与结构改善存在分离；固定密度80S已达到低误差；DRGN-AI较完整的两轮HPS结果保存在提交外目录；CryoSTAR图注计数有文字冲突；组成变化、half-map评估及STA需要新增适配。

Decision Log：2026-10-08，采用两个输入模式分别作公平比较，理由为结构先验影响可辨识性；采用逐数据集共同主指标联合门槛，理由为用户要求严格优势；采用30天截止并提前10天训练收口，理由为需要统计、重放与发布缓冲；资源协调内置于执行，依据用户要求推进。

Decision Log / 2026-10-09：用户明确资源为BRCA1–p53项目的集群＋本机。已读取该项目的连接记录，并只读核验Slurm资源登记与本机GPU；采用集群生产训练、本机开发与分析的分工。集群型号、可用配额和吞吐由环境预检补齐，截止日期保持11月7日。

| 风险 | 首次核验日期 | 应对与保留证据 |
| --- | --- | --- |
| 1AKE低SNR或前向差异限制准确恢复 | 10/10 | oracle诊断、生成器核查、噪声分层；保存失败与可辨识性结论 |
| 参考结构偏差制造表观优势 | 10/12 | 共同先验、错误/截短先验、V模式、独立图像验证 |
| 组成异质性/STA超出现有接口 | 10/10 | 明确adapter与表示需求，单列开发任务及完整覆盖门槛 |
| 官方baseline复现欠佳 | 10/14 | 校对作者配置及数据；作者推荐收敛表与等预算表并列 |
| 基线范围遗漏近期强方法 | 10/10 | 冻结检索截止日期和来源；CryoSPIRE/CryoFastAR等适用项进入矩阵 |
| 全量作业延误或存储压力 | 每日 | pilot吞吐估算、并发排队、checkpoint、25%故障预留 |
| 严格优势置信区间跨零 | 10/21 | 按冻结样本量完成、报告效应与误差；科学门槛保持待验证 |

Outcomes & Retrospective：当前产出为可执行的任务文档与验收设计。当前科研状态延续开发版证据，完整paper-ready条件仍需上述实验验证。每个里程碑完成时更新Progress、Decision Log、Surprises、Outcomes及机器可读状态。

## 12. 来源、附件与核验范围

[R1] CryoDRGN-AI，Nature Methods 2025，DOI 10.1038/s41592-025-02720-4。https://www.nature.com/articles/s41592-025-02720-4 。本地正式全文：develop/artifacts/drgnai_reproduction_20260930/paper_final.txt；数据设置见Methods及补充表。当前80S沉积栈固定CTF，论文写lognormal离焦；子集/噪声种子差异在docs/80S_REPRODUCTION.md记录。论文协议轨道与沉积栈复跑轨道分别保留。

[R2] CryoSTAR，Nature Methods 2024，DOI 10.1038/s41592-024-02486-1。https://www.nature.com/articles/s41592-024-02486-1 。数据来源和扩展图1用于核对模拟/实验范围。代码：https://github.com/bytedance/cryostar 。

[R3] CryoBench，NeurIPS 2024。https://cryobench.cs.princeton.edu/ ；https://github.com/ml-struct-bio/CryoBench 。官方仓库中的当前数据版本入口优先核验，输入版本与哈希随manifest保存。

[R4] CryoSPIRE，Reconstructing Heterogeneous Biomolecules via Hierarchical Gaussian Mixtures and Part Discovery。https://arxiv.org/abs/2506.09063 。作为近期竞争方法列入协议冻结任务。

[R5] CryoFastAR，ICCV 2025。https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_CryoFastAR_Fast_Cryo-EM_Ab_initio_Reconstruction_Made_Easy_ICCV2025_paper.html ；https://github.com/Cellverse/CryoFastAR 。预训练数据重叠与真实数据测试划分单列。

[R6] CryoDyna公开预印本。https://arxiv.org/abs/2510.16510 ；本项目原始代码来源：https://github.com/Qmi3/CryoDyna 。完整数据集与训练设置逐项核查列为首个里程碑。

[R7] DynaMight，Nature Methods 2024，DOI 10.1038/s41592-024-02377-5。https://www.nature.com/articles/s41592-024-02377-5 ；https://github.com/3dem/DynaMight 。

本任务书的实验数量、验收阈值与排程是预注册草案；现有实测数字仅来自第2节列出的仓库和本地结果。文献粒子数分别对应作者处理栈，实际落地的数量与筛选索引由manifest证明。source SHA、文献版本、半图划分与测试协议在10月10日前形成锁定文件。

附件：PAPER_EXECPLAN_20261008.typ、PAPER_EXECPLAN_20261008.pdf、paper_plan_20261008/acceptance_contract.json、paper_plan_20261008/benchmark_matrix.csv、paper_plan_20261008/qa.json。仓库相对路径以/home/nyh/tools/Cryodyna为根；新的计划与历史结果分别保存。
