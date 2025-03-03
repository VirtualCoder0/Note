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

​	近年来大语言模型飞速迭代发展，逐渐缩小距离实现 AGI 的技术差距，开源模型取得了巨大进展。DeepSeek 采用 Multi-head Latent Attention (MLA) (DeepSeek-AI, 2024c) 和 DeepSeekMoE (Dai et al., 2024) 两种架构（DeepSeekV2 (DeepSeek-AI, 2024c)的工作中提出）实现低成本的高效训练，同时他们还采取了 auxiliary-loss-free strategy 和 multi-token prediction training objective 这两种训练策略，并且支持了FP8混合精度训练；设计 DualPipe algorithm 以进行有效的管道并行性，减少管线 bubble；开发了有效的跨节点通信内核。

​	预训练使用 14.8T 高质量多种类语料，训练过程非常平稳。接下来采用两阶段的上下文扩展训练

- 第一阶段，最大上下文扩展至 32K
- 第二阶段，进一步扩展为 128K

​	进一步进行 post-training，包括有监督微调、强化学习，来对齐人类偏好、解锁模型潜力。在 post-training 阶段，将 DeepSeek-R1 模型的推理能力蒸馏到 V3 模型中，并且在模型准确率和生成长度之间维持平衡。 

![image-20250227104611380](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/image-20250227104611380.png)

​	预训练阶段，在 1T TOKENS 上训练 DeepSeek-V3 仅需要180K H800 GPU小时，整个训练阶段在不到两个月的时间内完成，花费了2664K GPU小时，上下文长度延长阶段需要 119k GPU小时，post-training 则需要 5K GPU小时，完整训练流程总计 2.788M GPU Hours。（该成本计算不包括前期研究、消融实验——架构、算法、数据的花费）

​	该技术报告主要贡献包括：

- 模型架构
  - an auxiliary-loss-free strategy for load balancing （最大程度优化专家负载问题带来的性能）
  - a Multi-Token Prediction (MTP) objective 
- Pre-Training
  - 混合精度训练框架（验证了 FP8 在大模型上训练的可行性和高效性）
  - 算法、框架和硬件的联合优化，克服了跨节点 MOE 训练中的通信瓶颈，提高了训练效率，降低了训练成本。
- Post-Training
  - 从 longChain-of-Thought (CoT) 模型中将推理能力蒸馏到训练的基座模型（verification and reflection patterns），保持 DeepSeek-V3 的输出格式和长度。
- Summary of Core Evaluation Results
  - Knowledge（与 GPT-4o and Claude-Sonnet-3.5 性能相当）
    - achieving 88.5 on MMLU, 75.9 on MMLU-Pro, and 59.1 on GPQA
    - SimpleQA 落后于 GPT-4o and Claude-Sonnet-3.5，Chinese SimpleQA DeepSeek-V3能力更强
  - Code, Math, and Reasoning
    - MATH-500 超过 Open-O1
    - Code 任务 DeepSeek-v3 的性能略低于Claude-sonnet-3.5

#### 结构细节

​	DeekSeek-V3 采用多头潜在注意力（MLA）（DeepSeek-Ai，2024c）和 DeepSeekMOE(Dai et al., 2024) 架构进行训练，提出 Multi-Token Prediction (MTP) 训练目标。

##### 基本体系结构（Transformer）

![image-20250227112858992](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/image-20250227112858992.png)

- Multi-Head Latent Attention
- 