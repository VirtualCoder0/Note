## DeepSeek-V3 技术报告阅读



#### 摘要

​	DeepSeek-V3 是一个强大的混合专家（MOE）语言模型，其总参数为671b，但推理训练过程中每个令牌只激活了 37B 参数。主要贡献为：

- 多头注意力（Multi-head Latent Attention (MLA)）
- 混合专家模型（DeepSeekMoE architectures）
- 一种新的负载均衡策略（auxiliary-loss-free strategy for load balancing）
- 提出 multi-token 预测的训练目标

​	DeepSeek-V3 的训练语料为 14.8T 的多种类高质量数据，并使用了有监督微调和强化学习手段激发模型能力。全部训练流程总计需要 2.788M H800 GPU hours，模型仓库：https://github.com/deepseek-ai/DeepSeek-V3.

![image-20250226112256290](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/desktop/image-20250226112256290.png)

#### 背景介绍 & Introduction

​	近年来大语言模型飞速迭代发展，逐渐缩小距离实现 AGI 的技术差距，开源模型取得了巨大进展。DeepSeek 采用 Multi-head Latent Attention (MLA) (DeepSeek-AI, 2024c) 和 DeepSeekMoE (Dai et al., 2024) 两种架构实现低成本的高效训练，同时他们还采取了 auxiliary-loss-free strategy 和 multi-token prediction training objective 这两种训练策略。