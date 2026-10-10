# CryoDyna-optpose 论文执行记录：2026-10-10

本次更新纳入DRGN-AI任务完成的1AKE固定真实密度旋转搜索诊断。相关训练与准确恢复门槛继续按论文任务书执行。

## 探索与独立粒子验证

搜索使用逐粒子真实class密度作为oracle先验，初始SO(3)网格4,608姿态，beam=8，四次细化各评估64个候选；平移固定为0。真实旋转用于生成clean对照与独立评估。角度为raw完整SO(3)，包含面内旋转。

| 粒子组 | growing：半径12/17/22/27/32 | cap17：半径12/17/17/17/17 |
| --- | ---: | ---: |
| 32粒子探索，原始图像 | 43.049° | 26.618° |
| 排除探索组后另抽128粒子验证 | 29.088842° | 29.959421° |

32粒子的clean growing对照终点均值为0.592°。128粒子验证组由seed=20261010抽取；两配置在验证前固定。manifest核对显示探索与验证粒子交集为0。

验证组配对差值cap17−growing为0.870578°，95%粒子percentile bootstrap区间为[−4.63,6.35]°，10,000次、seed=20261011。该区间包含0，当前证据支持继续保留growing配置。粒子重采样区间描述本次粒子抽样变化；训练种子重复与跨数据集优势继续作为后续验收项目。

## 评分与姿态的分离

验证组growing配置第一轮细化至第四轮的平均角度为25.701→29.089°，增加3.388°。统一在半径32重算相关性后，125/128粒子的分数提高，其中60个粒子的角度同时增大，平均相关性增加0.01018。

这一观察与10月9日局部损失诊断共同支持对“评分改善是否伴随姿态和结构改善”进行分项验收。下一工作包继续受控区分评分、噪声、前向匹配和结构先验的贡献，并实施固定pose/固定真值结构/固定参考结构/联合学习四组对照。原MLP、独立pose table、分块更新与信赖域按统一预算比较。

## 来源与报告衔接

原始结果：/home/nyh/tools/Cryodyna/output/pose_refinement_20261010/；独立验证为validation128/。manifest包含两组所有particle_ids、STAR哈希、配置及复现命令。核验时scripts/diagnose_1ake_refinement.py的SHA256与manifest一致：c0efdd0466c77130b7a109a71a8a05c618372b1367aa83015e30f17d1869270c；代码提交记录为937cbc4357ef81b5f575776d12bd73a239d3fce1。

最终图：output/pose_refinement_20261010/refinement_diagnosis.svg。对应Typst小节：/home/nyh/working/files/ilm/sections/drgnai_80s_20261001.typ。主入口仍为/home/nyh/working/files/ilm/20260903.typ；新编译报告为/home/nyh/tools/Cryodyna/output/pose_refinement_20261010/report/20260903.pdf，29页，新图11位于第15页，随后衔接Optpose第16页。

当前结果属于固定真实密度oracle诊断。DRGN-AI联合密度学习、Optpose结构/pose联合学习，以及真实实验数据独立验证各自保留输入条件、指标与验收记录。原始1AKE的SNR公式与操作顺序继续核对；本地加噪诊断采用明确记录的独立定义。
