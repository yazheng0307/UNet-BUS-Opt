# -*- coding: utf-8 -*-
"""Generate a plainer master's-thesis version without changing experiment facts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "博士学位论文初稿.md"
OUTPUT = ROOT / "硕士学位论文初稿_低专业度版.md"


PARAGRAPH_REWRITES = {
    (
        "乳腺超声具有无辐射、成本低、实时性强等优势，是乳腺疾病筛查和辅助诊断的重要影像手段。"
        "然而，超声成像中的散斑噪声、声影伪影、组织对比度不足以及病灶边界模糊等问题，使自动病灶分割仍面临明显困难。"
        "传统 U-Net 依靠编码器—解码器结构和跳跃连接融合多尺度特征，但其固定卷积路径难以根据图像困难类型自适应调整，"
        "局部卷积也难以显式建模远距离语义关系。此外，区域重叠损失对小范围边界偏移不够敏感，单一分割头无法区分漏分与误分两类方向相反的残余错误。"
    ): (
        "乳腺超声具有无辐射、成本较低和检查方便等优点，是乳腺疾病检查中常用的影像手段。"
        "但超声图像中常有散斑噪声、声影、对比度较低和边界模糊等问题，这些问题会增加自动分割的难度。"
        "U-Net 能够融合不同尺度的图像特征，但它对所有图像采用相同的处理方式，对较远区域之间的联系考虑得也不够充分。"
        "另外，常用损失函数更关注病灶区域是否重合，对少量边界偏移不够敏感，也不能分别处理漏分和误分。"
    ),
    (
        "针对上述问题，本文以统一 U-Net 为基线，构建由四个模块组成的循序渐进分割框架，并将研究内容组织为两个相互衔接的章节。"
        "第一部分面向语义表征，提出三挑战自适应路由模块 TriCAR（模块 A）和通道图关系桥 CGR-Bridge（模块 B）。"
        "TriCAR 针对噪声干扰、小病灶信息丢失和模糊边界三类困难设置差异化专家，并通过样本自适应路由完成特征重组；"
        "CGR-Bridge 在瓶颈处建立通道关系图，通过全局关系传播增强病灶语义一致性。"
        "第二部分面向输出细化，提出边界—距离协同细化模块 BD-CoRefine（模块 C）和不确定性驱动双误差修正模块 UDER（模块 D）。"
        "BD-CoRefine 同时学习形态学边界和有符号距离场，以连续几何信息修正粗分割；"
        "UDER 从预测熵和多尺度分歧估计不确定性，使用独立的假阴性增补头与假阳性删除头进行非对称纠错。"
    ): (
        "针对这些问题，本文以 U-Net 为基础，按照由简单到复杂的顺序加入四个模块。"
        "模块 A 称为多特征增强模块，它设置三个处理分支，分别关注噪声、小病灶和模糊边界，并根据输入图像自动调整三个分支的权重。"
        "模块 B 称为全局关系增强模块，它在网络最深层比较不同特征通道之间的关系，帮助模型从整体上判断病灶位置。"
        "在 A 和 B 的基础上，模块 C 使用边界和距离信息进一步修改分割轮廓，因此称为边界细化模块。"
        "最后，模块 D 找出模型不太确定的区域，并分别处理漏分和误分，因此称为不确定区域修正模块。"
        "四个模块形成了“局部特征增强—全局信息补充—边界细化—错误修正”的处理顺序。"
    ),
    (
        "为保证递进实验的可解释性，本文构建统一参数兼容模型，使 U、UA、UB、UAB、UABC 和 UABCD 六个变体共享基础结构。"
        "A、B、C、D 均通过零残差方式接入，新增模块启用时的初始输出与前一级模型逐像素完全一致，实测最大绝对差为 0。"
        "全部实验固定使用 BUSI split 3，其中训练集 452 例、验证集 195 例，固定阈值为 0.5，并按照逐病例平均 IoU 选择模型。"
    ): (
        "为了公平比较各模块，本文使用同一套代码建立 U、UA、UB、UAB、UABC 和 UABCD 六种模型，其中 UAB 为 LGR-UNet、UABCD 为 BUR-UNet。"
        "每次加入新模块时，先把该模块设置为不改变原模型输出的状态，再继续训练。测试表明，相邻阶段在继续训练前的最大输出差为 0。"
        "所有实验都使用 BUSI 的 split 3，训练集为 452 例，验证集为 195 例。预测阈值固定为 0.5，并根据每个病例 IoU 的平均值保存最佳模型。"
    ),
    (
        "逐病例统计显示，不同模块对困难病例的作用具有明显异质性。B 相对 U-Net 的平均 IoU 增益为 1.748 个百分点，"
        "95% Bootstrap 置信区间为 [0.615, 2.994] 个百分点；C 和 D 的病例级 Bootstrap 区间仍跨越零。"
        "随机种子 7、73 的完整复现实验均满足全部排序；三个种子的 UABCD 平均 IoU 为 75.128%±0.307%，平均 Dice 为 83.306%±0.223%。"
        "数据审计发现 split 3 中存在 24 对高置信跨集合近重复图像，涉及 20 个验证病例；剔除这些病例后，175 例 clean 子集上的三种子 UABCD 为 74.464% IoU 和 82.952% Dice，"
        "三种子均值仍保持全部排序，但部分单种子的 C 增益消失。该证据比单次实验更强，却仍不足以支持患者级或外部泛化结论。"
        "本文据此提出扩展至五个种子、患者级重划分、外部验证和临床一致性评价等后续研究方向。"
    ): (
        "进一步分析发现，各模块并不是对每个病例都有效。B 相对 U-Net 的平均 IoU 提高了 1.748 个百分点，"
        "其 95% Bootstrap 置信区间为 [0.615, 2.994]；C 和 D 的置信区间仍包含 0，说明它们的提升还不够稳定。"
        "使用随机种子 7、41 和 73 重复实验后，模型排序均满足要求，UABCD 的平均 IoU 为 75.128%±0.307%，Dice 为 83.306%±0.223%。"
        "数据检查还发现训练集和验证集中有相似图像。去掉受影响的 20 个验证病例后，剩余 175 例上的平均 IoU 和 Dice 分别为 74.464% 和 82.952%，"
        "总体排序没有改变，但 C 在个别随机种子下不再带来提升。因此，本文只说明当前方法在固定 split 3 上有效，暂不说明它在其他医院或设备数据上也能取得相同效果。"
    ),
    (
        "与自然图像不同，乳腺超声图像由相干声波反射形成，散斑既包含组织微结构信息，也表现为显著噪声。"
        "探头压力、扫描角度、设备参数和患者个体差异会造成强烈域偏移。病灶内部回声可能与周围腺体接近，后方声影会遮挡真实边界，小病灶在下采样过程中又容易丢失。"
        "上述问题使模型不仅需要识别“哪里像病灶”，还需要根据当前病例判断“应当采用何种特征处理方式”“哪些局部证据在全局上彼此一致”“边界应向何处移动”以及“当前错误更可能是漏分还是误分”。"
    ): (
        "乳腺超声图像与普通自然图像不同，图像中常有明显的散斑噪声。探头压力、扫描角度、设备参数和患者差异也会改变图像外观。"
        "有些病灶与周围组织灰度接近，声影会遮住部分边界，小病灶在多次下采样后还可能丢失。"
        "因此，模型既要提取局部细节，也要参考整幅图像的信息；在得到大致病灶区域后，还需要继续调整边界并修正漏分和误分。"
    ),
    (
        "U-Net 通过对称编码—解码结构和跳跃连接成为医学图像分割的重要基线。其优势是结构清晰、数据效率较高、易于嵌入领域模块。"
        "然而，直接堆叠注意力、Transformer 或更大编码器并不必然改善小样本超声分割。本文前期实验也发现，ConvNeXt-Tiny U-Net、ResNet101 U-Net 和整图 384×384 训练均未超过经过充分优化的轻量模型。"
        "因此，本研究选择围绕任务困难形成一条有因果顺序的模块链，而不是单纯追求网络规模。"
    ): (
        "U-Net 结构清楚、训练方便，是医学图像分割中常用的基础模型。增加注意力、Transformer 或更大的编码器，虽然可能提高模型能力，但在小数据集上不一定更好。"
        "前期实验中，ConvNeXt-Tiny U-Net、ResNet101 U-Net 和 384×384 输入方案都没有表现出稳定优势。"
        "因此，本文不继续盲目增大网络，而是根据乳腺超声分割中的具体问题，按顺序设计四个较小的改进模块。"
    ),
    (
        "第一章实验部分建立语义表征链：U-Net 编码器首先提取多尺度特征；A 在各编码尺度根据挑战类型进行零扰动自适应路由；"
        "B 在瓶颈处建立通道图并进行全局语义传播；解码器融合跳跃特征得到粗分割。第二章实验部分在 UnetAB 上进一步接入 C 和 D："
        "C 从边界及有符号距离恢复连续几何，D 利用不确定性分别执行漏分增补和误分删除。"
    ): (
        "整个方法分为两个阶段。第一阶段先用 U-Net 提取不同尺度的特征，A 根据图像内容对三个特征分支加权，B 再补充不同通道之间的整体联系，得到本文提出的 LGR-UNet。"
        "第二阶段以 LGR-UNet 为语义基础，C 利用边界和距离信息调整轮廓，D 再重点修改模型不确定区域中的漏分和误分，最终形成 BUR-UNet。"
    ),
    (
        "U-Net 编码器的固定卷积路径对所有病例执行相同操作。对于高噪声图像，模型需要抑制局部随机响应；对于小病灶，过度平滑会损失关键结构；"
        "对于模糊边界，更大的上下文感受野又十分必要。将所有目标压入一个卷积核会形成冲突。即使局部特征得到增强，远距离通道之间仍可能缺乏语义一致性。"
        "因此，本章按照“局部挑战分流—全局关系聚合”的顺序设计 A 与 B。"
    ): (
        "U-Net 对所有图像使用相同的卷积操作，但不同图像的问题并不相同。噪声较多时需要适当平滑，小病灶需要保留细节，边界模糊时则需要参考更大的周围区域。"
        "因此，A 使用三个分支分别处理这三种情况。局部特征增强后，模型还需要从整体上判断这些特征是否属于同一病灶，所以在 A 后加入 B。"
        "本章提出的组合模型（同时包含 A、B）称为 LGR-UNet（局部—全局表征增强 U-Net，代码变体 UAB）。"
    ),
    (
        "图 3-1 和表 3-1 表明，两项指标均满足 $U<U+A$、$U<U+B$ 且 $U+A+B$ 最优。A 的增益说明多挑战路由能够改善固定卷积路径；"
        "B 的增益更大，说明瓶颈语义关系是当前数据上的主要矛盾。将 A 加入 UB 后，IoU 继续提高 0.655 个百分点，Dice 提高 0.406 个百分点，"
        "证明局部挑战适配与全局关系传播具有互补性。"
    ): (
        "图 3-1 和表 3-1 表明，UA 和 UB 都优于基础 U-Net，UAB 的两项指标最高。A 说明针对噪声、小病灶和边界采用不同处理分支是有帮助的；"
        "B 的提升更大，说明整体特征关系对该数据集较重要。在 UB 上继续加入 A 后，IoU 提高 0.655 个百分点，Dice 提高 0.406 个百分点，"
        "说明 A 和 B 处理的问题不同，可以配合使用。"
    ),
    (
        "UnetAB 改善了病灶语义定位，但区域监督无法充分描述轮廓的方向与距离。模糊边界附近的少量像素偏移可能对小病灶 IoU 产生较大影响。"
        "进一步地，边界修正后仍会存在孤立误分和局部漏分，其修正方向不同。本章先由 C 将离散区域转换为边界与距离几何，再由 D 根据不确定性执行双向纠错。"
    ): (
        "LGR-UNet 已经能够找到大部分病灶区域，但对边界位置考虑得还不够细。对于小病灶，边界只偏移几个像素也会明显影响 IoU。"
        "此外，边界调整后仍可能出现漏分和误分。为此，本章先用 C 根据边界和距离信息调整轮廓，再用 D 找出不确定区域并分别修正两类错误，最终形成本文的最终模型 BUR-UNet（边界—不确定性修正 U-Net）。"
    ),
    (
        "上述现象与 D 的设计目标一致：UDER 只在高不确定区域产生有效残差，对多数已经正确的病例应接近恒等映射；"
        "但当前实现仍会使部分病例轻度退化。论文据此将 D 的结论限定为“提高固定验证集平均点估计并满足递进排序”，不宣称其已获得稳定群体效应。"
    ): (
        "D 的目标是只重点修改模型不确定的区域，已经分对的区域应尽量保持不变。实际结果表明，D 能提高验证集平均指标，但也会让一部分病例略有下降。"
        "因此，本文只说明 D 在当前固定验证集上提高了平均结果，不认为它对所有病例都有效。"
    ),
    (
        "本文围绕乳腺超声分割中的多挑战耦合、全局关系缺失、边界几何不足和残余错误非对称问题，提出由 A—D 构成的渐进框架。"
        "第一章通过 TriCAR 和 CGR-Bridge 建立局部挑战适配与全局语义一致性，得到本章最优 UnetAB。"
        "第二章通过 BD-CoRefine 和 UDER 将语义输出进一步转换为边界—距离几何并执行双向误差修正，得到最终 UABCD。"
    ): (
        "本文以 U-Net 为基础，依次加入四个功能明确的模块。A 根据图像情况组合三种局部特征，B 补充全局特征关系，二者共同组成 LGR-UNet。"
        "在此基础上，C 使用边界和距离信息调整轮廓，D 重点修正不确定区域中的漏分和误分，最终得到 BUR-UNet。"
    ),
}


REPLACEMENTS = [
    ("面向乳腺超声病灶分割的挑战自适应语义建模与边界不确定性闭环细化研究", "基于改进 U-Net 的乳腺超声图像分割方法研究"),
    ("# 博士学位论文初稿", "# 硕士学位论文初稿（低专业度版）"),
    ("博士论文终稿", "硕士论文终稿"),
    ("博士学位论文", "硕士学位论文"),
    ("This dissertation", "This thesis"),
    ("dissertation", "thesis"),
    ("# 第3章 挑战自适应语义关系建模", "# 第3章 局部与全局特征增强方法"),
    ("# 第4章 边界—不确定性闭环细化", "# 第4章 边界与错误区域细化方法"),
    ("## 3.2 模块 A：TriCAR", "## 3.2 模块 A：多特征增强模块"),
    ("### 3.2.1 三专家结构", "### 3.2.1 三分支结构"),
    ("### 3.2.2 自适应路由", "### 3.2.2 分支权重计算"),
    ("## 3.3 模块 B：CGR-Bridge", "## 3.3 模块 B：全局关系增强模块"),
    ("## 4.2 模块 C：BD-CoRefine", "## 4.2 模块 C：边界细化模块"),
    ("### 4.2.3 几何残差", "### 4.2.3 分割结果修正"),
    ("## 4.3 模块 D：UDER", "## 4.3 模块 D：不确定区域修正模块"),
    ("### 4.3.2 双误差修正", "### 4.3.2 漏分和误分修正"),
    ("Tri-Challenge Adaptive Routing (TriCAR, Module A)", "Multi-feature Enhancement (Module A)"),
    ("a Channel-Graph Relational Bridge (CGR-Bridge, Module B)", "Global Relation Enhancement (Module B)"),
    ("Boundary-Distance Cooperative Refinement (BD-CoRefine, Module C)", "Boundary Refinement (Module C)"),
    ("Uncertainty-Driven Dual Error Refinement (UDER, Module D)", "Uncertain-region Correction (Module D)"),
    ("TriCAR", "多特征增强模块"),
    ("CGR-Bridge", "全局关系增强模块"),
    ("BD-CoRefine", "边界细化模块"),
    ("UDER", "不确定区域修正模块"),
    ("三挑战自适应路由模块", "多特征增强模块"),
    ("通道图关系桥", "全局关系增强模块"),
    ("边界—距离协同细化模块", "边界细化模块"),
    ("不确定性驱动双误差修正模块", "不确定区域修正模块"),
    ("噪声专家", "噪声处理分支"),
    ("小病灶专家", "小病灶处理分支"),
    ("边界专家", "边界处理分支"),
    ("局部专家", "局部处理分支"),
    ("专家", "分支"),
    ("自适应路由", "自适应加权"),
    ("路由器", "权重计算分支"),
    ("语义表征", "特征表达"),
    ("全局语义传播", "全局信息传递"),
    ("语义关系", "特征关系"),
    ("语义一致性", "整体判断能力"),
    ("连续几何信息", "连续的边界距离信息"),
    ("离散边界", "边界"),
    ("形态学边界", "由掩码计算得到的边界"),
    ("多尺度分歧", "不同尺度预测之间的差异"),
    ("假阴性增补头", "漏分修正分支"),
    ("假阳性删除头", "误分修正分支"),
    ("增补头", "漏分修正分支"),
    ("删除头", "误分修正分支"),
    ("非对称纠错", "分别修正两类错误"),
    ("双向纠错", "分别修正漏分和误分"),
    ("零扰动", "不改变原输出的"),
    ("参数兼容", "可直接继承参数的"),
    ("残差回注", "以残差形式加回"),
    ("有界残差", "受限制的残差"),
    ("关系传播", "关系信息计算"),
    ("图推理", "特征关系计算"),
    ("几何残差", "边界修正量"),
    ("挑战分解", "按问题类型分开处理"),
    ("局部挑战适配", "局部特征处理"),
    ("多挑战耦合", "多种困难同时存在"),
    ("残余错误非对称", "漏分和误分需要不同处理"),
    ("异质性", "差异"),
    ("点估计", "平均结果"),
    ("外部泛化", "在外部数据上的表现"),
    ("域偏移", "不同设备和数据来源造成的差异"),
    ("结构性崩溃", "明显性能下降"),
    ("预注册早停策略", "提前设定的早停规则"),
    ("归因偏差", "比较误差"),
    ("病例覆盖", "在不同病例上的表现"),
    ("有效残差", "修正量"),
    ("恒等映射", "保持原结果"),
    ("闭环", "处理流程"),
    ("clean 子集", "去重子集"),
    ("clean 敏感性子集", "去重后的敏感性子集"),
    ("活跃参数", "实际使用参数"),
]


POST_REPLACEMENTS = [
    (
        "This thesis develops a progressive four-module framework based on a unified U-Net and organizes the study into two connected parts. "
        "The first part introduces Multi-feature Enhancement (Module A) and Global Relation Enhancement (Module B). "
        "多特征增强模块 employs specialized experts for noise suppression, small-lesion preservation, and boundary-context modeling. "
        "全局关系增强模块 constructs a compact channel graph at the bottleneck and propagates long-range semantic evidence. "
        "The second part proposes Boundary Refinement (Module C) and Uncertain-region Correction (Module D). "
        "边界细化模块 jointly predicts morphological boundaries and signed distance fields. "
        "不确定区域修正模块 estimates uncertainty from predictive entropy and multi-scale disagreement and uses separate heads for false-negative addition and false-positive removal.",
        "This thesis improves U-Net with four modules that are added in sequence. Module A uses three feature branches for noise, small lesions, and unclear boundaries. "
        "Module B uses relationships between feature channels to improve the overall lesion prediction. Module C uses boundary and distance information to adjust the contour. "
        "Module D locates uncertain areas and uses two branches to correct missed and incorrectly segmented regions. Together, the four modules follow a clear order: "
        "local feature enhancement, global information enhancement, boundary refinement, and error correction.",
    ),
    ("**Keywords:** breast ultrasound; medical image segmentation; U-Net; adaptive routing; graph reasoning; boundary supervision; uncertainty estimation",
     "**Keywords:** breast ultrasound; medical image segmentation; U-Net; feature enhancement; boundary refinement; error correction"),
    ("**关键词：** 乳腺超声；医学图像分割；U-Net；自适应加权；图关系推理；边界监督；不确定性估计",
     "**关键词：** 乳腺超声；医学图像分割；U-Net；特征增强；边界细化；错误修正"),
    ("## 1.2 关键科学问题", "## 1.2 需要解决的主要问题"),
    ("本文聚焦以下四个相互关联的问题：", "根据乳腺超声图像的特点，本文主要解决以下四个问题："),
    ("1. **多种困难同时存在问题。** 散斑噪声、小病灶和模糊边界需要不同的感受野与滤波特性，固定卷积核难以对所有病例同时最优。",
     "1. **不同图像需要不同处理。** 散斑噪声、小病灶和模糊边界的特点不同，只使用固定卷积操作很难同时处理好这些情况。"),
    ("2. **局部—全局整体判断能力问题。** 局部处理分支能够增强特定证据，但不同通道和远距离区域之间缺少显式关系约束。",
     "2. **局部信息和整体信息需要配合。** 局部特征能够保留细节，但模型还需要参考不同通道和较远区域的信息，减少错误判断。"),
    ("3. **离散区域与连续几何脱节问题。** BCE 和 Dice 主要约束区域重叠，难以描述边界内外方向和像素到轮廓的连续距离。",
     "3. **病灶边界需要进一步细化。** BCE 和 Dice 主要关注区域重合，对边界内外方向以及像素到轮廓的距离考虑不足。"),
    ("4. **漏分和误分需要不同处理问题。** 漏分需要增加前景响应，误分需要降低前景响应；单头残差可能产生相互抵消的梯度。",
     "4. **漏分和误分需要分别修正。** 漏分需要补充病灶区域，误分需要删除错误区域，使用同一个输出分支不容易同时完成两种操作。"),
    ("### 1.3.2 超声噪声与挑战自适应建模", "### 1.3.2 超声图像中的多分支特征处理"),
    ("按按问题类型分开处理", "按问题类型分开处理"),
    ("在统一可直接继承参数的网络中实现轻量通道图传播，并用有界、零初始化以残差形式加回瓶颈",
     "在同一网络中实现轻量通道关系计算，并把结果以受限制的残差方式加回最深层特征"),
    ("本文的可辨识贡献是将这些思想重构为“轻量共享主干—可直接继承参数的—不改变原输出的渐进训练”的统一体系",
     "本文的主要工作是把这些思路放入同一个轻量 U-Net 中，并采用可以继承前一级参数、加入新模块时不改变原输出的训练方式"),
    ("论文答辩时应将创新表述限定在具体结构、联合机制、不改变原输出的训练协议及其经验证的组合效果，而不能以新名称替代先行工作披露。",
     "因此，本文的创新主要体现在具体结构、模块组合方式和逐步训练方法上，不把已有的基本思路说成本文首次提出。"),
    ("1. 提出**不改变原输出的三挑战自适应加权 多特征增强模块**。模块使用噪声、小病灶和边界处理分支，通过样本级权重进行动态融合；零初始化输出层保证加入模块时不破坏已训练基线。",
     "1. 设计**多特征增强模块 A**。该模块设置噪声、小病灶和边界三个处理分支，并根据输入图像计算分支权重，使网络能够选择更合适的局部特征。"),
    ("2. 提出**轻量全局关系增强模块 全局关系增强模块**。模块以归一化通道特征构图，在 FP32 中完成关系信息计算，并通过有界以残差形式加回瓶颈特征。",
     "2. 设计**全局关系增强模块 B**。该模块在网络最深层计算不同特征通道之间的关系，再把得到的全局信息加回原特征，用于减少只看局部区域造成的误分。"),
    ("3. 提出**边界—距离协同细化 边界细化模块**。模块联合学习边界和连续有符号距离场，通过边界修正量修正区域预测。",
     "3. 设计**边界细化模块 C**。该模块同时预测病灶边界和像素到边界的距离，并利用这两类信息调整 LGR-UNet 的分割轮廓。"),
    ("4. 提出**不确定性驱动双误差修正 不确定区域修正模块**。模块联合预测熵与不同尺度预测之间的差异，显式拆分假阴性增补和假阳性删除，形成从语义到几何再到误差的处理流程。",
     "4. 设计**不确定区域修正模块 D**。该模块结合预测概率和不同尺度输出的差异找到容易出错的区域，再使用两个独立分支分别修正漏分和误分。"),
    ("5. 建立**可直接继承参数的的渐进验证协议**。六个模型变体共享状态字典，每次新增模块均经过最大输出差为零的一致性测试，降低因结构重建和随机初始化造成的比较误差。",
     "5. 建立**逐步训练和对比实验方法**。六种模型使用同一基础结构，每次加入新模块时都先保证输出不变，再继续训练，从而使各阶段的比较更加公平。"),
    ("第 3 章研究 A、B 两个语义模块并得到 UnetAB。", "第 3 章研究 A、B 两个特征增强模块并得到 LGR-UNet。"),
    ("第 4 章研究 C、D 两个细化模块并得到最终 UABCD。", "第 4 章研究 C、D 两个细化模块并得到最终的 BUR-UNet。"),
    ("该协议体现“先语义、后几何、再纠错”的研究假设。", "该训练顺序体现“先定位病灶，再调整边界，最后修正错误”的思路。"),
    ("关系信息计算与以残差形式加回为：", "关系计算和结果回加过程为："),
    ("因此 B 同样满足无扰动接入。", "因此，加入 B 时也不会立即改变原模型输出。"),
    ("结果说明 B 的独立增益具有更稳定的在不同病例上的表现，A 的贡献更依赖病例类型；",
     "结果说明 B 在不同病例上的提升相对更稳定，而 A 的效果更容易受到病例类型影响；"),
    ("本章建立了 A 与 B 的逻辑链。A 负责根据局部困难类型组织特征，B 负责将局部证据整合为全局一致语义。统一实验得到 UAB 最优，并将其命名为 UnetAB，作为下一章唯一固定起点。",
     "本章先使用 A 处理噪声、小病灶和模糊边界等局部问题，再使用 B 补充整体特征关系。实验中同时包含 A、B 的组合模型 IoU 和 Dice 均为本章最高，因此将其命名为 LGR-UNet，并作为下一章的基础模型。"),
    ("距离分支使用 Smooth L1 损失，使网络获得病灶内外方向和到边界的连续的边界距离信息。",
     "距离分支使用 Smooth L1 损失，使网络能够判断像素位于病灶内部还是外部，并学习像素到边界的距离。"),
    ("熵反映单输出置信度，尺度分歧反映不同语义层级之间的不一致，两者具有互补性。",
     "熵表示主输出是否确定，标准差表示不同尺度的预测是否一致。两者结合后，可以更全面地找出容易出错的区域。"),
    ("不确定区域修正模块 预测假阴性响应", "模块 D 分别预测漏分响应"),
    ("其中 $R_{FN}$ 对应需要补充的病灶证据，$R_{FP}$ 对应需要删除的错误前景。",
     "其中，$R_{FN}$ 用于补充漏掉的病灶区域，$R_{FP}$ 用于删除错误预测的前景区域。"),
    ("基于验证耐心的提前设定的早停规则", "提前设定的早停规则"),
    ("本章在固定 UnetAB 上依次加入 C 与 D，建立从区域语义、边界距离到不确定性纠错的处理流程。两项指标均形成严格递增关系，UABCD 为最终模型。逐病例分析同时表明，C、D 的收益尚不稳定，后续工作应针对完全漏检病例和 D 引起的轻度退化开展专门研究。",
     "本章在 LGR-UNet 上依次加入 C 和 D。C 负责调整边界，D 负责修正不确定区域。IoU 和 Dice 均按照 LGR-UNet、边界细化型 LGR-UNet、BUR-UNet 的顺序提高，因此 BUR-UNet 是本章的最终模型。不过，逐病例结果说明 C 和 D 的提升还不够稳定，后续仍需继续改进。"),
    ("全局整体判断能力", "整体判断能力"),
    ("自然图像预训练主干可以提高绝对指标，但它们不直接提供 A-D 的递进归因。",
     "使用自然图像预训练的主干可以提高指标，但不能直接说明 A—D 四个模块分别带来了多少提升。"),
    ("只有建立稳定语义模型并采用不改变原输出的渐进训练后才获得正向平均结果。这支持本文的核心逻辑：几何与不确定性细化依赖可靠语义起点，模块顺序不是任意排列。",
     "只有先得到较稳定的 LGR-UNet，并采用加入模块时保持原输出不变的训练方法，C 和 D 才取得正向结果。这说明边界调整和错误修正需要建立在较准确的病灶定位结果上，四个模块的加入顺序有实际依据。"),
    ("比 去重子集", "比去重子集"),
    ("在 去重子集", "在去重子集"),
    ("具有稳健性", "仍能保持"),
    (
        "A parameter-compatible implementation is established for six variants: U, UA, UB, UAB, UABC, and UABCD. "
        "Every newly activated module is identity-initialized, and the maximum absolute output difference between two consecutive stages is exactly zero before fine-tuning. "
        "All experiments use BUSI split 3 with 452 training cases and 195 validation cases. Models are selected by mean per-case IoU at a fixed threshold of 0.5.",
        "The same implementation is used for six model variants: U, UA, UB, UAB, UABC, and UABCD. When a new module is added, it is initialized so that the output remains unchanged before further training. "
        "All experiments use BUSI split 3, which contains 452 training cases and 195 validation cases. The threshold is fixed at 0.5, and the best checkpoint is selected by mean per-case IoU.",
    ),
]


MAPPING_INSERT = """本文中四个模块采用容易理解的功能名称。为了与程序代码对应，表 1-1 给出论文名称和代码实现的关系。\n\n| 模块 | 论文中的名称 | 主要作用 | 代码中的对应实现 |\n|---|---|---|---|\n| A | 多特征增强模块 | 分别处理噪声、小病灶和模糊边界特征 | `StableTriChallengeAdapter` |\n| B | 全局关系增强模块 | 补充不同特征通道之间的整体联系 | `ChannelGraphReasoning` |\n| C | 边界细化模块 | 使用边界和距离信息调整分割轮廓 | `BoundaryDistanceCooperativeHead` |\n| D | 不确定区域修正模块 | 分别修正容易出错区域中的漏分和误分 | `UncertaintyDrivenDualErrorRefinement` |\n\n为了区分正文使用的模型称谓与代码实现中的变体标识，表 1-2 给出二者的对应关系。正文一律使用具有含义的模型称谓，`UAB/UABC/UABCD` 等代码标识仅在表格的“代码变体”列、附录训练命令与运行目录中保留。\n\n| 代码标识 | 论文称谓 | 英文全称 | 使用场景 |\n|---|---|---|---|\n| U | 基础 U-Net | — | 基线 |\n| UA | 局部特征增强变体 | — | 单模块消融 |\n| UB | 全局关系增强变体 | — | 单模块消融 |\n| UAB | LGR-UNet | Local-Global Representation-enhanced U-Net | 第 3 章提出模型 |\n| UABC | 边界细化型 LGR-UNet | — | 第 4 章中间阶段 |\n| UABCD | BUR-UNet | Boundary-Uncertainty Refinement U-Net | 第 4 章最终模型 |\n\n"""


MAINSTREAM_REVIEW = """### 1.3.2 主流对比模型介绍

为了更清楚地说明本文方法与常见分割网络的区别，下面介绍项目中已经实现并报告 BUSI 结果的 11 个模型。它们可以分为传统卷积模型、Transformer 或混合模型，以及轻量化模型三类。

1. **U-Net。** U-Net 采用编码器和解码器结构。编码器逐步提取较深层特征，解码器恢复图像尺寸，同尺度跳跃连接把浅层细节直接送到解码端。它结构简单、容易训练，是本文和多数医学图像分割工作的基础模型[1]。
2. **Attention U-Net。** Attention U-Net 在跳跃连接处加入注意力门。解码器提供的较高层信息用于判断哪些编码特征更重要，从而减弱背景区域的影响。不过，注意力门会增加少量参数和计算量[2]。
3. **U-Net++。** U-Net++ 把原来直接相连的跳跃连接改成多层嵌套连接，并在中间层逐步融合编码和解码特征。它还可以使用深监督帮助不同深度的输出学习，但连接较多，结构比 U-Net 更复杂[3]。
4. **U-Net3+。** U-Net3+ 在每个解码阶段同时接收多个编码层和解码层的特征，使不同尺度的信息能够更充分地融合。这种全尺度连接有利于处理大小不同的目标，但计算量通常较大[4]。
5. **TransUnet。** TransUnet 先使用卷积网络提取局部特征，再使用 Transformer 建立较远位置之间的联系，最后通过 U 形解码器恢复分割图。它兼顾局部和整体信息，但参数量较大，对训练数据和预训练权重也比较敏感[6]。
6. **MedT。** MedT 使用门控轴向注意力，把二维注意力分成水平和垂直两个方向计算，以降低完整自注意力的开销。它还采用整图和局部图块结合的训练方法，同时学习整体位置和局部细节[31]。
7. **SwinUnet。** SwinUnet 使用分层的 Swin Transformer 组成编码器和解码器，并通过移动窗口让相邻窗口之间交换信息。窗口注意力能够控制计算量，但模型在小型医学数据集上的效果容易受到预训练方式和训练设置影响[7]。
8. **UNeXt。** UNeXt 前几层使用卷积，网络较深位置使用基于多层感知机的特征块。它通过通道移动和特征投影学习局部联系，重点是减少参数量和提高推理速度[32]。
9. **CMU-Net。** CMU-Net 面向超声图像设计，使用 ConvMixer 模块扩大特征的有效范围，并在跳跃连接中加入多尺度注意力门。前者用于补充整体信息，后者用于筛选更有用的浅层特征[33]。
10. **CMUNeXt。** CMUNeXt 是轻量卷积网络。它使用大卷积核和倒置瓶颈结构提取较大范围的信息，并通过 Skip-Fusion 模块改善编码器和解码器之间的特征融合[34]。
11. **Mobile U-ViT。** Mobile U-ViT 面向移动设备设计，使用大卷积核卷积块完成分层特征提取，在最深层加入较浅的 Transformer，并采用级联解码器恢复分割结果。该模型代表了卷积和轻量 Transformer 结合的移动端设计路线[35]。

这 11 个模型代表了三条常见改进路线：Attention U-Net、U-Net++ 和 U-Net3+ 主要改进特征融合；TransUnet、MedT 和 SwinUnet 主要增加长距离信息；UNeXt、CMUNeXt 和 Mobile U-ViT 更重视轻量化。本文没有直接替换 U-Net 主干，而是在统一基线上按 A、B、C、D 的顺序解决局部特征、整体关系、边界和易错区域问题。

"""


REFERENCE_ROWS = """| U-Net | 34.52 | 139.32 | 65.52 GFLOPs | 68.61 | 76.97 |
| Attention U-Net | 34.87 | 129.92 | 66.63 GFLOPs | 68.55 | 76.88 |
| U-Net++ | 26.90 | 125.50 | 37.62 GFLOPs | 69.49 | 78.06 |
| U-Net3+ | 26.97 | 50.60 | 199.74 GFLOPs | 68.38 | 76.88 |
| TransUnet | 105.32 | 112.95 | 38.52 GFLOPs | 71.39 | 79.85 |
| MedT | 1.37 | 22.97 | 2.40 GFLOPs | 63.36 | 73.37 |
| SwinUnet | 27.14 | 392.21 | 5.91 GFLOPs | 54.11 | 65.46 |
| UNeXt | 1.47 | 650.48 | 0.58 GFLOPs | 65.04 | 74.16 |
| CMU-Net | 49.93 | 93.19 | 91.25 GFLOPs | 71.42 | 79.49 |
| CMUNeXt | 3.14 | 471.43 | 7.41 GFLOPs | **71.56** | **79.86** |"""


CHAPTER3_COMPARISON = """## 3.5 与主流模型的参考对比

原项目在 BUSI 的 647 个病灶病例上报告了主流模型结果。经原仓库作者确认，原表“±”前的数值对应固定 split 3 的训练结果，因此本节采用这些数值[36]。本文模型在 seed 7、41 和 73 中按照 IoU 选择最优运行，并报告同一次运行对应的 Dice。LGR-UNet 的最优运行是 seed 41，IoU 为 74.847%，Dice 为 82.943%。

| 模型 | 参数量/M | FPS | 计算量（原报告口径） | IoU/% | Dice/% |
|---|---:|---:|---:|---:|---:|
{rows}
| **LGR-UNet（本文，seed 41，代码变体 UAB）** | **11.28** | 约112.2 | 19.42 GMACs | **74.847** | **82.943** |

### 3.5.1 差值描述性分析

本节所列参考模型中，CMUNeXt 的 IoU 和 Dice 最高，分别为 71.56% 和 79.86%。LGR-UNet 比 CMUNeXt 高 3.287 个 IoU 百分点和 3.083 个 Dice 百分点；比参考 U-Net 高 6.237 个 IoU 百分点和 5.973 个 Dice 百分点。以上差值是在固定 split 3 汇总指标上的直接相减，用于说明性能差距，不等同于统计显著性。

| 参照模型 | LGR-UNet的IoU差值/百分点 | LGR-UNet的Dice差值/百分点 |
|---|---:|---:|
| U-Net | +6.237 | +5.973 |
| Attention U-Net | +6.297 | +6.063 |
| U-Net++ | +5.357 | +4.883 |
| U-Net3+ | +6.467 | +6.063 |
| TransUnet | +3.457 | +3.093 |
| MedT | +11.487 | +9.573 |
| SwinUnet | +20.737 | +17.483 |
| UNeXt | +9.807 | +8.783 |
| CMU-Net | +3.427 | +3.453 |
| CMUNeXt | +3.287 | +3.083 |

LGR-UNet 约有 11.28M 实际使用参数，比参考 U-Net 的 34.52M 少约 67.3%，比 TransUnet 的 105.32M 少约 89.3%，但约为 CMUNeXt 参数量的 3.59 倍。这说明 LGR-UNet 在分割精度和模型大小之间取得了折中，但并不是参数最少的模型。参考表使用 GFLOPs，本文复杂度脚本使用 GMACs；参考模型和本文模型的速度测试硬件也没有完全统一，因此 FPS 和计算量只作描述，不进行显著性判断。

### 3.5.2 本文模型的内部配对检验

主流模型目前只有汇总指标，没有保存与本文完全对应的 195 例逐病例预测，因此不能严谨计算 LGR-UNet 相对 CMUNeXt 等模型的配对 p 值。统计检验只在本文保存了逐病例结果的模型之间进行。本文采用单侧 Wilcoxon 配对符号秩检验，并用 20,000 次 Bootstrap 给出平均差的 95% 置信区间。

| 配对比较 | 指标 | 平均提升/百分点 | Bootstrap 95%区间 | 单侧 p 值 | 结论 |
|---|---|---:|---:|---:|---|
| LGR-UNet（seed 41）对 基础 U-Net | IoU | +2.402 | [0.896, 4.058] | 0.0211 | 显著 |
| LGR-UNet（seed 41）对 基础 U-Net | Dice | +2.225 | [0.736, 3.849] | 0.0231 | 显著 |
| LGR-UNet（seed 41）对 全局关系增强变体（seed 41） | IoU | +0.655 | [-0.067, 1.511] | 0.0931 | 不显著 |
| LGR-UNet（seed 41）对 全局关系增强变体（seed 41） | Dice | +0.406 | [-0.266, 1.213] | 0.0958 | 不显著 |

结果表明，LGR-UNet 相对基础 U-Net 的两项指标均达到 `p<0.05`，且平均差置信区间不跨 0，可以认为 A+B 组合相对基线具有统计显著优势。但 LGR-UNet 相对较强的全局关系增强变体仅有小幅提升，尚未达到 `p<0.05`。因此，本章能够支持“组合模型明显优于基础 U-Net”，但不能把 A 在 B 之后的增量写成已经稳定显著。

### 3.5.3 为什么 A 和 B 能够带来提升

1. **A 与乳腺超声中的具体困难对应。** 噪声处理分支先做局部平均，小病灶分支保留紧凑细节，边界分支使用较大范围的卷积。根据图像内容计算三个分支的权重，可以减少一种固定卷积同时处理所有困难时的冲突。
2. **B 补充了 U-Net 较弱的整体判断能力。** B 在网络最深层计算通道关系，把分散在不同位置和通道中的病灶信息联系起来。这有助于保留真正的病灶响应，并压低远离病灶的错误前景。全局关系增强变体相对基础 U-Net 的 IoU 和 Dice 置信区间均不跨 0，也与这一作用相符。
3. **A 和 B 处理的问题不同。** A 先改善局部特征，B 再从整体上检查这些特征是否一致。两者按顺序连接后，LGR-UNet 同时高于两个单模块变体（局部特征增强变体和全局关系增强变体），说明局部增强和整体关系并不是简单重复。
4. **不改变原输出的初始化提高了训练稳定性。** A、B 刚加入时不会破坏已有分割结果，后续训练只需学习有用的修正量，因此比从随机输出直接联合训练更稳定。

""".format(rows=REFERENCE_ROWS)


CHAPTER4_COMPARISON = """## 4.5 与主流模型的参考对比

第二章继续使用固定 split 3 的主流模型结果。BUR-UNet 在三个随机种子中按 IoU 选择最优运行，seed 73 的 IoU 最高，为 75.442%，该次运行对应的 Dice 为 83.382%。虽然 seed 41 的 Dice 为 83.480%，但本文不把不同种子的最大 IoU 和最大 Dice 拼接为一条结果。

| 模型 | 参数量/M | FPS | 计算量（原报告口径） | IoU/% | Dice/% |
|---|---:|---:|---:|---:|---:|
{rows}
| **BUR-UNet（本文，seed 73，代码变体 UABCD）** | **11.31** | 约99.7 | 21.40 GMACs | **75.442** | **83.382** |

### 4.5.1 差值描述性分析

BUR-UNet 比本节参考结果最好的 CMUNeXt 高 3.882 个 IoU 百分点和 3.522 个 Dice 百分点，比参考 U-Net 高 6.832 个 IoU 百分点和 6.412 个 Dice 百分点。BUR-UNet 约有 11.31M 参数，比 CMUNeXt 大约 3.60 倍，但仍明显小于参考 U-Net、U-Net++、U-Net3+、TransUnet、SwinUnet 和 CMU-Net。

| 参照模型 | BUR-UNet的IoU差值/百分点 | BUR-UNet的Dice差值/百分点 |
|---|---:|---:|
| U-Net | +6.832 | +6.412 |
| Attention U-Net | +6.892 | +6.502 |
| U-Net++ | +5.952 | +5.322 |
| U-Net3+ | +7.062 | +6.502 |
| TransUnet | +4.052 | +3.532 |
| MedT | +12.082 | +10.012 |
| SwinUnet | +21.332 | +17.922 |
| UNeXt | +10.402 | +9.222 |
| CMU-Net | +4.022 | +3.892 |
| CMUNeXt | +3.882 | +3.522 |

在 seed 73 的同一条渐进训练链中，LGR-UNet 的 IoU/Dice 为 74.590%/82.789%，加入 C 和 D 后提高到 75.442%/83.382%，即提高 0.852 和 0.593 个百分点。模型参数量只比 LGR-UNet 增加约 0.03M，但计算量从 19.42 GMACs 增加到 21.40 GMACs，单张延迟从 8.91 ms 增加到 10.03 ms。C 和 D 的参数开销较小，但全分辨率边界分支和多尺度辅助输出会增加一定计算时间。

### 4.5.2 本文模型的内部配对检验

表中主流模型没有逐病例预测文件，因此下列统计检验仍限定在本文模型内部。BUR-UNet 与 LGR-UNet 的比较使用同为 seed 73 的 195 个病例，避免把不同种子结果强行配对；与 U-Net 的比较使用公共基线逐病例结果。

| 配对比较 | 指标 | 平均提升/百分点 | Bootstrap 95%区间 | 单侧 p 值 | 结论 |
|---|---|---:|---:|---:|---|
| BUR-UNet（seed 73）对 LGR-UNet（seed 73） | IoU | +0.852 | [-0.051, 1.793] | 0.0038 | 秩检验显著，均值区间跨0 |
| BUR-UNet（seed 73）对 LGR-UNet（seed 73） | Dice | +0.593 | [-0.292, 1.472] | 0.0042 | 秩检验显著，均值区间跨0 |
| BUR-UNet（seed 73）对 基础 U-Net | IoU | +2.997 | [1.439, 4.706] | 0.0025 | 显著 |
| BUR-UNet（seed 73）对 基础 U-Net | Dice | +2.664 | [1.267, 4.223] | 0.0034 | 显著 |

BUR-UNet 相对 U-Net 的两项检验均达到 `p<0.01`，且平均差置信区间不跨 0，可以认为最终模型相对基础 U-Net 具有统计显著优势。BUR-UNet 相对同 seed 的 LGR-UNet 有 112 例改善、82 例下降，Wilcoxon 检验也达到 `p<0.01`；但平均提升的 Bootstrap 区间仍跨 0，说明少数病例的较大波动会影响平均值。较准确的结论是“多数病例的排序变化支持提升，但平均增益仍需更多随机种子验证”。另外，由于 seed 73 是从三个种子中按 IoU 选出的最优运行，这些 p 值属于选择后的探索性结果，不能替代预先指定种子的独立验证。

### 4.5.3 为什么 C 和 D 能够继续提升结果

1. **C 为边界调整提供了方向信息。** 普通 Dice 和 BCE 主要判断像素属于前景还是背景，不能直接告诉模型边界应向内还是向外移动。C 同时学习边界和有符号距离，使靠近轮廓的像素获得更明确的调整方向，因此能够修正少量边界偏移。
2. **D 把漏分和误分分开处理。** 漏分需要增加病灶概率，误分需要降低病灶概率。D 使用两个独立分支完成相反操作，避免一个修正分支同时学习两种方向时相互干扰。
3. **不确定性让 D 重点修改易错区域。** 预测熵可以找到概率接近 0.5 的像素，不同尺度输出的差异可以找到各层判断不一致的位置。两者结合后，D 不需要大范围改动已经分对的区域，而是把计算集中在模糊边界和疑似漏分区域。
4. **C 和 D 建立在较好的 LGR-UNet 上。** A 和 B 已经完成主要病灶定位，C、D 只需学习较小的边界和错误修正量。早期实验中 C、D 从随机状态直接训练没有超过控制组，也说明它们依赖可靠的前级结果。
5. **增益仍然有限且存在病例差异。** C、D 主要改变少量像素，所以平均提升小于 A、B；Bootstrap 区间跨 0 也表明它们还没有在所有病例上稳定获益。这与模块作用范围较窄的设计相符，并提示后续需要针对完全漏检病例继续改进。

""".format(rows=REFERENCE_ROWS)


ADDITIONAL_REFERENCES = """[31] Valanarasu J M J, Oza P, Hacihaliloglu I, Patel V M. Medical Transformer: Gated Axial-Attention for Medical Image Segmentation. MICCAI, 2021. arXiv:2102.10662.

[32] Valanarasu J M J, Patel V M. UNeXt: MLP-based Rapid Medical Image Segmentation Network. arXiv:2203.04967, 2022.

[33] Tang F, Wang L, Ning C, Xian M, Ding J. CMU-Net: A Strong ConvMixer-based Medical Ultrasound Image Segmentation Network. IEEE ISBI, 2023. DOI: 10.1109/ISBI53787.2023.10230609.

[34] Tang F, Ding J, Wang L, Ning C, Zhou S K. CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion. arXiv:2308.01239, 2023.

[35] Tang F, Nian B, Ding J, et al. Mobile U-ViT: Revisiting Large Kernel and U-shaped ViT for Efficient Medical Image Segmentation. ACM Multimedia, 2025. arXiv:2508.01064.

[36] Tang F. Medical-Image-Segmentation-Benchmarks: Results on BUSI [EB/OL]. GitHub, 2026-07-14. https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks.

"""


BEST_SEED_REPLACEMENTS = [
    (
        "实验结果表明，第一部分中 U-Net、UA、UB 和 UAB 的 IoU 分别为 72.445%、73.622%、74.193% 和 74.847%，Dice 分别为 80.718%、81.817%、82.537% 和 82.943%。UAB 在两项指标上均为第一部分最优，并被命名为 UnetAB。第二部分以同一个 UnetAB 检查点为起点，UABC 和 UABCD 的 IoU 分别为 74.932% 和 75.115%，Dice 分别为 83.022% 和 83.480%，形成 UnetAB < UnetAB+C < UnetAB+C+D 的递进关系。最终模型包含约 11.31M 实际使用参数，在 RTX 4060 Laptop GPU 上处理单张 256×256 图像的平均前向延迟约为 10.03 ms。",
        "实验结果采用三个随机种子中 IoU 最高的一次，并报告同一次运行的 Dice。第一部分中 U-Net、UA、UB 和 UAB 的 IoU 分别为 72.445%、74.185%、74.193% 和 74.847%，Dice 分别为 80.718%、82.256%、82.537% 和 82.943%。UAB 在两项指标上均为第一部分最优，即本文提出的 LGR-UNet。第二部分中 UAB、UABC 和 UABCD 的 IoU 分别为 74.847%、74.966% 和 75.442%，Dice 分别为 82.943%、83.089% 和 83.382%，即 LGR-UNet、边界细化型 LGR-UNet 和 BUR-UNet 呈递进提升。最终模型包含约 11.31M 实际使用参数，在 RTX 4060 Laptop GPU 上处理单张 256×256 图像的平均前向延迟约为 10.03 ms。",
    ),
    (
        "In Chapter 1, U, UA, UB, and UAB achieve IoU scores of 72.445%, 73.622%, 74.193%, and 74.847%, and Dice scores of 80.718%, 81.817%, 82.537%, and 82.943%, respectively. UAB is therefore denoted as UnetAB. In Chapter 2, UABC and UABCD further improve IoU to 74.932% and 75.115%, and Dice to 83.022% and 83.480%. Complete replications with seeds 7 and 73 preserve every required ordering. Across seeds 7, 41, and 73, UABCD obtains 75.128% +/- 0.307% IoU and 83.306% +/- 0.223% Dice. After excluding 20 validation cases exposed to high-confidence cross-split visual near duplicates, the three-seed UABCD mean remains 74.464% IoU and 82.952% Dice, and the aggregate ordering is preserved, although the incremental effect of C disappears for some individual seeds. The final model contains 11.31M active parameters and requires approximately 10.03 ms per 256×256 image on an RTX 4060 Laptop GPU. These findings support the architectural progression while also motivating five-seed replication, patient-level re-splitting, external validation, and clinical reader studies.",
        "For the main tables, each model uses the run with the highest IoU among seeds 7, 41, and 73, together with the Dice from that same run. In Chapter 1, U, UA, UB, and UAB obtain 72.445%, 74.185%, 74.193%, and 74.847% IoU, and 80.718%, 82.256%, 82.537%, and 82.943% Dice. UAB is the LGR-UNet. In Chapter 2, UABC and UABCD reach 74.966% and 75.442% IoU, and 83.089% and 83.382% Dice. Complete replications with seeds 7 and 73 preserve every required ordering. Across all three seeds, BUR-UNet obtains 75.128% +/- 0.307% IoU and 83.306% +/- 0.223% Dice. After excluding 20 validation cases exposed to high-confidence cross-split visual near duplicates, the three-seed BUR-UNet mean remains 74.464% IoU and 82.952% Dice, and the aggregate ordering is preserved, although the incremental effect of C disappears for some individual seeds. The final model contains 11.31M active parameters and requires approximately 10.03 ms per 256×256 image on an RTX 4060 Laptop GPU. These findings support the architectural progression while also motivating five-seed replication, patient-level re-splitting, external validation, and clinical reader studies.",
    ),
    (
        "| UA | ✓ |  | 73.622 | 81.817 | +1.177 |",
        "| 局部特征增强变体 | UA | ✓ |  | 74.185 | 82.256 | +1.740 |",
    ),
    (
        "图 3-1 和表 3-1 表明，UA 和 UB 都优于基础 U-Net，UAB 的两项指标最高。A 说明针对噪声、小病灶和边界采用不同处理分支是有帮助的；B 的提升更大，说明整体特征关系对该数据集较重要。在 UB 上继续加入 A 后，IoU 提高 0.655 个百分点，Dice 提高 0.406 个百分点，说明 A 和 B 处理的问题不同，可以配合使用。",
        "图 3-1 和表 3-1 使用各模型三个随机种子中 IoU 最高的一次。局部特征增强变体和全局关系增强变体都优于基础 U-Net，LGR-UNet 的两项指标最高。A 和 B 相对基础 U-Net 的最佳 IoU 分别提高 1.740 和 1.748 个百分点。LGR-UNet 与全局关系增强变体的最优运行都来自 seed 41，在全局关系增强变体基础上继续加入 A 后，IoU 提高 0.655 个百分点，Dice 提高 0.406 个百分点，说明 A 和 B 处理的问题不同，可以配合使用。",
    ),
    (
        "| UnetAB+C | ✓ |  | 74.932 | 83.022 | +0.085 |",
        "| 边界细化型 LGR-UNet | UABC | ✓ |  | 74.966 | 83.089 | +0.119 |",
    ),
    (
        "| UnetAB+C+D | ✓ | ✓ | **75.115** | **83.480** | **+0.183** |",
        "| BUR-UNet | UABCD | ✓ | ✓ | **75.442** | **83.382** | **+0.476** |",
    ),
    (
        "结果满足 $UnetAB<UnetAB+C<UnetAB+C+D$，最终模型在 IoU 和 Dice 上均为第二章最优。C 带来的平均增益较小，表明 UnetAB 已完成主要区域定位，C 主要对边界像素进行微调。D 的 Dice 增益大于 IoU 增益，说明其对预测区域整体重合程度具有更明显影响。",
        "按照三个随机种子中最高 IoU 选择结果后，LGR-UNet、边界细化型 LGR-UNet 和 BUR-UNet 仍呈严格递进提升，最终模型在 IoU 和 Dice 上均为第二章最优。边界细化型 LGR-UNet 相对 LGR-UNet 的 IoU 和 Dice 分别提高 0.119 和 0.146 个百分点；BUR-UNet 相对边界细化型 LGR-UNet 分别提高 0.476 和 0.293 个百分点。C 主要调整少量边界像素，D 继续修正漏分和误分，因此两步提升都小于第一章的 A、B。",
    ),
    (
        "统一 split 3 实验表明，第一章 U、UA、UB、UAB 的 IoU 为 72.445%、73.622%、74.193%、74.847%，Dice 为 80.718%、81.817%、82.537%、82.943%；第二章 UAB、UABC、UABCD 的 IoU 为 74.847%、74.932%、75.115%，Dice 为 82.943%、83.022%、83.480%。因此，当前实验已经满足两章预设的严格排序关系。",
        "统一 split 3 实验中，按三个随机种子最高 IoU 选择后，第一章 基础 U-Net、局部特征增强变体、全局关系增强变体、LGR-UNet 的 IoU 为 72.445%、74.185%、74.193%、74.847%，Dice 为 80.718%、82.256%、82.537%、82.943%；第二章 LGR-UNet、边界细化型 LGR-UNet、BUR-UNet 的 IoU 为 74.847%、74.966%、75.442%，Dice 为 82.943%、83.089%、83.382%。因此，两章仍满足预设的严格排序关系。",
    ),
]


APPENDIX_TITLE_REPLACEMENTS = [
    ("# 第一章：UA\n", "# 第一章：局部特征增强变体（代码变体 UA）\n"),
    ("# 第一章：UB\n", "# 第一章：全局关系增强变体（代码变体 UB）\n"),
    ("# 第一章：UAB / UnetAB\n", "# 第一章：LGR-UNet（代码变体 UAB）\n"),
    ("# 第二章：UABC\n", "# 第二章：边界细化型 LGR-UNet（代码变体 UABC）\n"),
    ("# 第二章：UABCD\n", "# 第二章：BUR-UNet（代码变体 UABCD）\n"),
]


APPENDIX_MAPPING_TABLE = """上述命令中的 `--variant UAB/UABC/UABCD` 参数与 `runs/...` 目录名均为程序实际使用的代码标识，须原样保留。正文中相应模型请统一使用论文称谓，对照关系见表 A-1。

| 代码标识 | 论文称谓 | 英文全称 | 使用场景 |
|---|---|---|---|
| U | 基础 U-Net | — | 基线 |
| UA | 局部特征增强变体 | — | 单模块消融 |
| UB | 全局关系增强变体 | — | 单模块消融 |
| UAB | LGR-UNet | Local-Global Representation-enhanced U-Net | 第 3 章提出模型 |
| UABC | 边界细化型 LGR-UNet | — | 第 4 章中间阶段 |
| UABCD | BUR-UNet | Boundary-Uncertainty Refinement U-Net | 第 4 章最终模型 |

"""


BEST_SEED_PROTOCOL = """本文一共完成随机种子 7、41 和 73 的训练。主结果表按照逐病例平均 IoU 从三个种子中选择最高的一次，并同时报告该次运行对应的 Dice，不分别挑选两项指标的最大值。按此规则，UA、UB、UAB、UABC 和 UABCD 的最优种子依次为 73、41、41、73 和 73。多种子均值与标准差仍在第 5 章单独报告。为了避免错误配对，逐病例统计检验只比较同一随机种子或公共 U-Net 基线的预测。

"""


def generate_architecture_figure():
    output = ROOT / "thesis_artifacts/figures/fig01_overall_architecture_plain.png"
    blocks = [
        ("Input\nultrasound", "#d8d8d8"),
        ("U-Net\nencoder", "#9ecae1"),
        ("A: Multi-feature\nenhancement", "#4e79a7"),
        ("B: Global relation\nenhancement", "#59a14f"),
        ("U-Net\ndecoder", "#9ecae1"),
        ("C: Boundary\nrefinement", "#f28e2b"),
        ("D: Error-region\ncorrection", "#b07aa1"),
    ]
    figure, axis = plt.subplots(figsize=(20, 5), dpi=120)
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 5)
    axis.axis("off")
    xs = [0.35, 3.05, 5.75, 8.75, 11.75, 14.35, 17.25]
    widths = [2.2, 2.2, 2.55, 2.55, 2.1, 2.35, 2.4]
    for index, ((label, color), x, width) in enumerate(zip(blocks, xs, widths)):
        box = FancyBboxPatch(
            (x, 1.45), width, 1.55,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.8, edgecolor="#444444", facecolor=color,
        )
        axis.add_patch(box)
        text_color = "white" if index in (2, 3, 5, 6) else "#222222"
        axis.text(
            x + width / 2, 2.22, label,
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=text_color,
        )
        if index < len(blocks) - 1:
            next_x = xs[index + 1]
            axis.annotate(
                "", xy=(next_x, 2.22), xytext=(x + width, 2.22),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#444444"},
            )
    axis.text(6.8, 4.05, "Chapter 1: local and global feature enhancement",
              ha="center", va="center", fontsize=17, fontweight="bold")
    axis.text(15.8, 4.05, "Chapter 2: boundary and error refinement",
              ha="center", va="center", fontsize=17, fontweight="bold")
    figure.tight_layout(pad=0.2)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    text = SOURCE.read_text(encoding="utf-8")
    for old, new in PARAGRAPH_REWRITES.items():
        if old not in text:
            raise RuntimeError("Expected paragraph was not found: {}".format(old[:40]))
        text = text.replace(old, new)

    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    for old, new in POST_REPLACEMENTS:
        if old not in text:
            raise RuntimeError("Expected post-replacement text was not found: {}".format(old[:40]))
        text = text.replace(old, new)

    for old, new in (
        ("### 1.3.5 既有工作与本文创新边界", "### 1.3.6 既有工作与本文创新边界"),
        ("### 1.3.4 边界监督、距离场与不确定性", "### 1.3.5 边界监督、距离场与不确定性"),
        ("### 1.3.3 特征关系计算与全局关系建模", "### 1.3.4 特征关系计算与全局关系建模"),
    ):
        if old not in text:
            raise RuntimeError("Expected review heading was not found: {}".format(old))
        text = text.replace(old, new, 1)
    review_marker = "### 1.3.2 超声图像中的多分支特征处理\n"
    if review_marker not in text:
        raise RuntimeError("Mainstream review insertion point was not found")
    text = text.replace(
        review_marker,
        MAINSTREAM_REVIEW + "### 1.3.3 超声图像中的多分支特征处理\n",
        1,
    )

    text = text.replace("## 3.6 本章小结", "## 3.7 本章小结", 1)
    chapter3_marker = "## 3.5 配对统计分析\n"
    if chapter3_marker not in text:
        raise RuntimeError("Chapter 3 comparison insertion point was not found")
    text = text.replace(
        chapter3_marker,
        CHAPTER3_COMPARISON + "## 3.6 配对统计分析\n",
        1,
    )

    text = text.replace("## 4.7 本章小结", "## 4.8 本章小结", 1)
    text = text.replace("## 4.6 定性结果", "## 4.7 定性结果", 1)
    chapter4_marker = "## 4.5 逐病例增益与统计解释\n"
    if chapter4_marker not in text:
        raise RuntimeError("Chapter 4 comparison insertion point was not found")
    text = text.replace(
        chapter4_marker,
        CHAPTER4_COMPARISON + "## 4.6 逐病例增益与统计解释\n",
        1,
    )

    appendix_marker = "# 附录 A 关键复现实验命令\n"
    if appendix_marker not in text:
        raise RuntimeError("Reference insertion point was not found")
    text = text.replace(
        appendix_marker,
        ADDITIONAL_REFERENCES + appendix_marker,
        1,
    )

    for old, new in APPENDIX_TITLE_REPLACEMENTS:
        if old not in text:
            raise RuntimeError("Expected appendix title was not found: {}".format(old))
        text = text.replace(old, new, 1)
    mapping_marker = "  --epochs 60 --seed 41\n```\n\n# 附录 B 完成度与待补实验"
    if mapping_marker not in text:
        raise RuntimeError("Appendix mapping-table insertion point was not found")
    text = text.replace(
        mapping_marker,
        "  --epochs 60 --seed 41\n```\n\n" + APPENDIX_MAPPING_TABLE + "# 附录 B 完成度与待补实验",
        1,
    )

    for old, new in BEST_SEED_REPLACEMENTS:
        if old not in text:
            raise RuntimeError("Expected best-seed text was not found: {}".format(old[:40]))
        text = text.replace(old, new, 1)
    protocol_marker = "该训练顺序体现“先定位病灶，再调整边界，最后修正错误”的思路。"
    if protocol_marker not in text:
        raise RuntimeError("Best-seed protocol insertion point was not found")
    text = text.replace(
        protocol_marker,
        BEST_SEED_PROTOCOL + protocol_marker,
        1,
    )
    text = text.replace(
        "thesis_artifacts/figures/fig02_chapter1_ablation.png",
        "thesis_artifacts/figures/fig02_chapter1_ablation_best.png",
        1,
    )
    text = text.replace(
        "thesis_artifacts/figures/fig04_chapter2_ablation.png",
        "thesis_artifacts/figures/fig04_chapter2_ablation_best.png",
        1,
    )
    text = text.replace(
        "## 3.6 配对统计分析\n\n",
        "## 3.6 配对统计分析\n\n本节为保证逐病例一一对应，继续使用 seed 41 同一训练链中的预测，不与各模型跨种子挑选后的最大值进行配对。\n\n",
        1,
    )
    text = text.replace(
        "## 4.6 逐病例增益与统计解释\n\n",
        "## 4.6 逐病例增益与统计解释\n\n本节原有逐病例分析使用 seed 41 的同一训练链，用来说明 C、D 在该固定运行中的病例差异；4.5.2 节则补充最优 seed 73 最终模型与同 seed LGR-UNet 的配对结果。\n\n",
        1,
    )

    marker = "本文技术路线如图 1-1 所示。\n\n"
    if marker not in text:
        raise RuntimeError("Technical-route marker was not found")
    text = text.replace(marker, MAPPING_INSERT + marker, 1)

    text = text.replace(
        "thesis_artifacts/figures/fig01_overall_architecture.png",
        "thesis_artifacts/figures/fig01_overall_architecture_plain.png",
    )
    text = text.replace(
        "> 稿件状态说明：",
        "> 版本说明：本版本在不改变模型、实验数据和结论的前提下，使用更接近普通硕士论文的表达方式。",
        1,
    )
    text = text.replace("\n# 摘要\n", "\n# 目录\n\n# 摘要\n", 1)
    text = text.replace("不确定区域修正模块", "易错区域修正模块")
    text = text.replace("模块 D 分别预测漏分响应 $R_{FN}$ 和假阳性响应 $R_{FP}$",
                        "模块 D 分别预测漏分响应 $R_{FN}$ 和误分响应 $R_{FP}$")
    OUTPUT.write_text(text, encoding="utf-8")
    generate_architecture_figure()
    print("wrote {} ({} chars)".format(OUTPUT, len(text)))


if __name__ == "__main__":
    main()
