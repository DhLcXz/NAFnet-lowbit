# 目标
优化NAFNet 在 low-level 任务性能（降低计算量和内存，比如：将卷积替换为加、减法），保持精度

# 20241114
进展：
1. 讨论项目的定量评测。最终目的是降低功耗、同时保持精度，但是我们无法分析功耗降低多少，这部分由芯片专家给出；我们可以分析计算量和内存降低了多少。low-level 任务可以由PSNR和SSIM算出数值，但更重要的是，人眼看图片是否由劣化（即图片质量变差）。建议每组实验列出10～20+的对比图片。
2. 加 WHT 与 NAFNet baseline 的指标（PSNR、SSIM）接近，但图片质量劣化，查看 tensorboard 分析原因，loss 曲线看不出来。
3. 空间可分离卷积的优先级暂时往下调，集中 WHT 这条技术路线。

遗留：
1. 实验结果展示更多对比图片
2. 分析整个模型的计算开销热点
3. 在github仓库维护会议纪要

# 20241107
进展：用FWHT 替换1x1卷积，在NAFNet 进行实验

遗留：
1. 指标（PSNR、SSIM）接近，图片质量掉了，查看 tensorboard 分析原因 
2. 训练矩阵计算，测试用蝶式计算

# 20241031
遗留：
1，构建 github，维护代码
2，维护文档，更新试验结果（包括定量和定性）

# 20240926
视频去噪任务。卷积乘法是计算量的开销热点，需要压缩。思路：FWHT，空间可分离卷积（3x3卷积替换成两个1x3卷积），加法神经网络。
约束：卷积用定点（整数）计算，W8A16（芯片上用8位乘法器实现）


