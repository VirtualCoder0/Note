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

- MLA（Multi-head Latent Attention）	

  ​	DeepSeek进一步优化，推出了多头潜在注意力机制（MLA）。MLA旨在进一步缩小KV缓存的大小，同时在性能上超越之前提到的注意力机制（包括MHA)。它通过将KV缓存压缩到低维潜在空间，成功将缓存大小减小了93.3% ！下面我们详细看看它是如何做到的。

  - **低秩键值联合压缩**：MLA不会像传统方式那样计算和存储每个令牌的键和值，而是使用下投影矩阵$W(DKV)$把它们压缩成潜在向量$C(KV)$。在推理时，再通过每个头的上投影矩阵$W(UK)$（用于键）和$W(UV)$（用于值）从这个潜在向量中重建KV对。为了降低计算成本，MLA还进行了巧妙的优化：把矩阵$W(UK)$合并到$W(Q)$中，这样就不用显式计算键$K(i)$了；把矩阵$W(UV)$合并到$W(O)$中，也就无需显式计算值$V(i)$了。
  
    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224048661.png)
  
    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224104539.png)
  
    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224119202.png)
  
    ![img](https://segmentfault.com/img/remote/1460000046119625)
  
    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224149320.png)
  
    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224457572.png)
  
  - **查询的低秩压缩**：MLA对查询也进行了类似的压缩。
  
    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224522636.png)
  
    使用下投影矩阵$W(DQ)$将查询压缩成潜在表示$C(Q)$，需要时再用上投影矩阵$W(UQ)$进行重建。**虽然这样做不会减少KV缓存的大小，但能降低训练期间的激活内存使用**。（激活内存是训练过程中前向传播时用于存储中间激活的内存，反向传播计算梯度时会用到这些激活。)在使用MHA训练时，每一层都会在内存中显式计算和存储查询，且数量会随着层数线性增加。而**在MLA中，只存储查询的压缩表示，减少了反向传播时存储的总激活量**。不过要注意，在推理时，每个令牌计算一次查询后就会丢弃，不会存储用于反向传播的激活。所以，**查询压缩主要是提高了训练效率，对推理性能没有影响。**
  
    ![](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305224622198.png)

​	研究人员尝试在MLA中使用旋转位置嵌入（RoPE）来加入令牌位置信息，可这遇到了一些问题。在深入探讨之前，我们先来了解一下位置编码在大语言模型中的工作原理。

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164202691.png)

​	Transformer架构并行处理令牌，这虽然让它比RNN在计算上更有优势，但也导致它对令牌顺序不敏感。比如，“The cat sits on the mat.”和“The mat sites on the cat.”这两句话，对Transformer来说没什么区别。但在语言处理中，顺序很重要，所以需要添加位置信息。位置嵌入主要有两种类型：绝对位置嵌入，给每个令牌根据其位置分配唯一编码；相对位置嵌入，编码的是令牌之间的相对距离，而不是绝对位置。这两种嵌入又可以分为固定的（用数学函数计算）和可学习的（模型训练时通过反向传播更新参数）。在原始的Transformer论文中，作者使用的是固定的绝对位置嵌入，通过交替的正弦和余弦函数在偶数和奇数维度上计算位置嵌入$PE$，公式为：$PE(pos,2i)=sin(pos/10000^{2i/d(model)})$，$PE(pos,2i + 1)=cos(pos/10000^{2i/d(model)})$，其中$pos$是令牌索引，$i$是令牌嵌入维度的索引，$d(model)$是总令牌嵌入维度。这些位置嵌入和令牌嵌入维度相同，可以直接相加后再输入Transformer进行处理。

​	后来，2023年的一项研究提出了旋转位置嵌入（RoPE），这是一种在注意力机制中直接编码绝对和相对位置的新方法。RoPE不会像之前那样添加位置嵌入，而是根据令牌的位置旋转令牌嵌入。具体来说，对于位置$m$处维度为$d$的令牌嵌入$x(m)$，分别使用权重矩阵$W(q)$和$W(k)$将其转换为查询向量$q(m)$和键向量$k(n)$ 。在进行自注意力计算前，使用与位置相关的旋转矩阵$R(m)$对这些向量进行旋转。$R(m)$会独立作用于$q$和$k$中的每对维度。以二维向量为例，旋转矩阵$R(m)$定义为：$\begin{bmatrix}cos(m\theta)& -sin(m\theta)\\sin(m\theta)&cos(m\theta)\end{bmatrix}$，这个矩阵会将向量逆时针旋转，旋转角度与位置$m$成正比，为$m\theta$（$d = 2$时，$\theta = 1$ ）。对于更高维的向量（假设维度为偶数），会将相邻的维度两两配对，分别进行二维旋转。通过这种方式，RoPE可以让注意力分数编码令牌的相对位置，而且还能体现出相距较远的令牌之间联系的相对重要性低于较近的令牌。

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164202944.png)

###### 为什么RoPE与MLA不兼容？

​	回到MLA，我们知道它通过创建键和值的潜在压缩表示$C(KV)$来减少内存使用和提高推理效率。而RoPE需要在计算注意力分数前，根据位置信息用旋转矩阵$R(m)$旋转查询和键。但由于**MLA存储的是压缩的键值缓存**，不是完整的键，所以如果应用RoPE，每次生成新令牌时都得重新计算所有之前的键 ，这就破坏了使用压缩KV表示带来的效率提升。另外，之前为了优化，MLA把键向上投影矩阵$W(UK)$合并到了$W(Q)$中，而RoPE的旋转操作会导致矩阵乘法不满足交换律，使得$W(UK)$无法像原来那样与$W(Q)$解耦和合并。

![image-20250305111345626](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250305111345849.png)

######  为什么要缓存 $c_t^{KV}$ 

​	考虑另一个问题， $c_t^{KV}$有什么用，为什么缓存 $c_t^{KV}$？首先 $c_t^{KV}$ 会在后面用 $W^{UK},W^{UV}$ 升维，之后的计算则和MHA的一致。而为什么缓存 $c_t^{KV}$ 是有效的，我们要考虑KV Cache是干什么的。前面已经提到，KV Cache是为了减少重复计算的。显然缓存 $k_{t,i}$ 是不行的，这样就变成和MHA一样，没有减少KV Cache缓存。而似乎我们的 $c_t^{KV}$ 在每次推理的时候都必须通过 $W^{UK},W^{UV}$  重新计算出 $k_{t,i},v_{t,i}$ ，这样虽然减少了缓存，但是没有减少任何的计算量。这也是理解的关键，实际上我们并不需要显式的计算出 $k_{t,i},v_{t,i}$  ，它可以被 $W^{UQ},W^{O}$ 通过预计算的方式吸收。让我们展开不带位置编码时 $q_t^Tk_t$ 的情况
$$
q_t^Tk_t=(W^{UQ}c_t^Q)^TW^{UK}c_t^{KV}=c_t^Q(W^{UQ})^TW^{UK}c_t{KV}=c_t^QWc_t^{KV}
$$
​	可以看到，我们不需要计算出 $W^{UK}c_t^{KV}$ ，可以把 $W^{UK}$ 吸收进 $W^{UQ}$ ，这样就避免了重复计算。我们每次缓存的 $c_t^{KV}$ 都可以直接参与计算，而不需要显示的计算出$K$。这样我们MLA就达到了克服以往方法中KV Cache过大的问题并且保留的KV Cache该有的减少重复计算的功能。

​	让我们考虑添加旋转位置编码，在RoPE中介绍过，旋转位置编码在主流大模型中基本都在用，作者希望延续使用旋转位置编码，但是这并不容易。RoPE它首先不能作用在式(22)上，因为矩阵通常不满足乘法交换律。
$$
k_t^C=RoPE(W^{UK}c_t^{KV}) \\
(W^Qq_t)^Tk_t^C=(W^Qq_t)^TRoPE_t(W^{UK}c_t^{KV})≠q_t^T(W^Q)^TW^{UK}RoPE_t(c_t^{KV})
$$
​	这样我们不得不去计算 $W^{UK}c_t{KV}$ ，这是我们不愿意看到的，因为在上文中已经说明，我们需要 $W^{UK}$ 被吸收，我们不希望看到重复计算。因此作者考虑解耦的QK，即上述公式所示。专门生成多头的 $q_t^R$ 和共享的 $k_t^R$ 。共享的 $k_t^R$ 指的是每个头的$K$都用这同一个 $k_t^R$ 。然后再通过 $concat$ 操作得到完整的Q和K。这样所有的queries中 $d_h$ 维都是不带位置编码信息的，而另外 $d_h^R$ 维（实践中取 $1/2 d_h$ ）则是带旋转位置编码信息的。此时我们仍然能把 $W^{UK}$ 吸收进 $W^{UQ}$ 。
$$
q_{t,i}^Tk_{j,i}= 
\left[
\matrix{
	c_t^Q(W^{UQ})^T & (q_t^R)^T 
}
\right]
\left[
\matrix{
W^{UK}c_t^{KV}\\
k_t^R
}
\right] \\
\left[
\matrix{
c_t^Q(W^{UQ})T,(q_t^R)T
}
\right]
\left[
\matrix{
W^{UK}c_t^{KV}\\
k_t^R
}
\right]
=c_t^Q(W^{UQ})^TW^{UK}c_t^{KV}+(q_t^R)^Tk_t^R
$$
这样MLA就圆满了，即完成了首要任务减少KV Cache，也保住了RoPE的应用，还保住了没有重复计算。

###### 那么RoPE如何在MLA中使用呢？

​	在MLA中，研究人员引入了一种新方法——解耦旋转位置嵌入（Decoupled Rotary Position Embedding）。首先，计算两种类型的键：一种是之前讨论过的压缩键$K(C)$；另一种是位置敏感或解耦键$K(R)$，它是未压缩的键，用于存储应用RoPE所需的位置信息。查询也会进行类似计算，得到潜在查询$Q(C)$和用于RoPE的位置敏感或解耦查询$Q(R)$。这些计算是在推理时进行的，不会存储。这种方法既保留了低秩KV压缩的优势，又能通过单独存储$K(R)$来应用位置敏感变换，还不会影响RoPE的注意力计算。在这种方式下，每个令牌需要缓存$K(R)$和$C(KV)$，总共缓存$[d(c) + d(h)(R)]×L$个元素（$d(c)$是潜在密钥维度，$d(h)(R)$是解耦密钥的每个头维度，$L$是MLA中的层数） ，相比传统Transformer模型，效率大大提高。最后，使用压缩和位置敏感的查询和键来计算注意力分数，得到最终输出。

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164203583.png)

- DeepSeekMoE with Auxiliary-Loss-Free Load Balancing

  ![image-20250306161119163](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250306161119581.png)

  - 不平衡的专家负载会导致路由崩溃并且降低专家并行中的计算效率，传统方法使用辅助损失来避免负载不均衡，但是过大的辅助损失会影响模型性能。DeekSeek-V3 在 FFNs 中应用 DeepSeekMoE 架构，DeepSeekMoE 使用更细粒度的专家，分为共享专家和路由专家，并采用了一种无需辅助损失函数的负载均衡策略，即为每个专家设置偏置 $b_i$ 。

    ![image-20250306161238783](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250306161238879.png)

    ​	 $u_t$ 代表第 $t$ 个输入 $token$， $N_s$ 和 $N_r$  表示共享专家和路由专家的数量， $g_{i,t}$ 代表对于第 $t$ 个输入$token$，第 $i$ 个专家的门控得分， $FFN_i^{(r)}$,$FFN_i^{(s)}$ 代表第 $i$ 个共享和路由专家， $s_{i,t}$ 代表输入和专家的亲合度，具体的计算方式是将输入向量和每个专家的一个内置向量做内积并经过 $sigmoid$ 处理， $K_r$  代表选择的topk个路由节点中的 $k$ 。对于输入 $token$ 向量，首先计算它和每一个专家的亲合度  $s_{i,t}$ ，随后选出 $topk$，作为门控得分 $g_{i,t}$ ，并对门控得分进行归一化，得到选出的 $topk$ 的专家各自的权重 gi,t′ ，最后按照该权重将各个专家的输出进行加权，和共享专家的计算结果、输入向量加和（**残差结构**）得到最终输出。

  - 传统MoE的负载均衡

    ​	MoE模型经常面临负载不均衡的问题，例如多个token指派到同一个专家处理。而这种问题在训练当中则会逐渐恶化。因为随着输入训练数据的增加，经常被选择的专家将会得到更多参数更新的机会，从而生成更好的回答，继而更倾向于被选择。事实上，在具体的训练和推理过程中，MoE的参数是稀疏的，因为只有实际被激活的专家会进行参数的更新或者推理。

    ​	针对负载不均的问题，传统的思路一般是在门控中添加噪声，从而避免反复路由到同一个专家。公式如下（摘自huggingface）：

    ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250306162823450.jpeg)

  - DeepSeekMoE 的负载均衡策略

    ​	deepseek v3 主要采用了两种方式进行负载均衡：Auxiliary-Loss-Free Load Balancing（辅助无损负载均衡）和 Complementary Sequence-Wise Auxiliary Loss（互补序列辅助损失）。Auxiliary-Loss-Free Load Balancing 仅在选择 topk 个亲合度分数的时候，为每一个专家添加一个偏置量 $b_i$ （这个偏置量仅仅用于topk筛选，不加入后续的权重计算）。在每个训练步骤结束的时候，如果某个专家过载，则按照某一特定比例减少其偏置量；如果某个专家负载不足，则相应的按照同一比例增加其偏置量。

    ![image-20250306164215114](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250306164215217.png)

    ​	Complementary Sequence-Wise Auxiliary Loss 主要是为了解决单个输入序列内部的极端负载不均衡。 $T$ 是输入序列的总长度， $s_{i,t}^′$ 代表归一化的输入序列和各个专家的亲和力，$P_i$ 代表第 $i$ 个专家和序列内的每一个token的亲合度均值，代表了该专家和序列的整体亲合度， $f_i$ 代表第 $i$ 个专家在该序列预测过程中的选中频率，$α$ 为较小的常数超参数。可以看到， $f_iP_i$ 代表了第 $i$ 个专家的负载强度，当部分专家反复在topk中被选中的时候， $L_{Bal}$ 会增大，即体现了对负载不均衡的惩罚。

    ![image-20250306164706889](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250306165103112.png)

    ​	最后，在通信方面，DeepSeek-V3 使用限制路由机制来限制训练期间的通信成本，即每个token最多被发送到 $M$ 个算力节点，这些节点是根据分布在每个节点上的专家的最高个 $K_r/M$ 亲和力分数之和来选择的。在此约束下，deepseek v3的 MoE 训练框架几乎可以实现完全的计算-通信重叠。DeepSeek-V3 的负载均衡策略保证了在训练期间不会丢弃任何token。
- Multi-Token Prediction

  ![image-20250306214133003](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250306214133338.png)

  ​	通过解码阶段的优化，将1-token的生成，转变成multi-token的生成，从而提升训练和推理的性能。具体来说，在训练阶段，一次生成多个后续token，可以一次学习多个位置的label，进而有效提升样本的利用效率，提升训练速度；在推理阶段通过一次生成多个token，实现成倍的推理加速来提升推理性能。
  $$
  h^{′k}_i = M_k[RMSNorm(h^{k−1}_i);RMSNorm(Emb(t_{i+k}))]
  $$
  ​	第 K 个 MTP模块由共享嵌入层EMB(·)，共享输出头Outhead(·)，变压器块$TRM_K$(·)和一个投影矩阵 $M_K \in R^{d×2d}$组成。对于第 $i$ 个输入 token $t_i$，在d]第 k 个预测深度上，我们首先将第 i 个token 在第 k-1 层的表示$h^{k-1}_i \in R^d$ 和第 i + k 个token 的嵌入向量 $Emb(t_{i + k}) \in R_D$ 使用线性投影层进行合并。

  ​	每个MTP模块的嵌入层与主模型共享。组合后的$h^{'k}_i$用作第 k 层 Transformer 块的输入，输出下一层的输出 $h_i^k$：
  $$
  h^k_{1:T−k} = TRM_k(h^{′k}_{1:T −k})
  $$
  ​	其中 T 表示输入序列长度，而 $_{i：j}$ 表示切片操作（左右边界）。最后，以 $h_i^k$ 为输入，共享输出头将计算第k 个额外预测 token $p^k_{i+1+k} \in R^V$ 的概率分布，其中 V 是词汇表大小:
  $$
  P^k_{i+k+1} = OutHead(h^k_i)
  $$

#### 硬件设施

##### 计算集群

​	DeekSeek-V3 在 2048 张 H800 GPUs 集群上进行训练，该集群中每个节点包括 8 张 H800，通过 NVLink 和 NVSwitch 互联，使用 InfiniBand(IB) 进行跨节点通信。

##### 训练框架

​	DeekSeek-V3 使用 HAI-LLM 框架进行训练，总体而言，DeekSeek-V3 采用 16 路管线并行，64 路专家并行。为了保证训练的高效性，研发b 团队应用了许多工程优化方法。首先，团队设计了双管算法（DualPipe algorithm），以进行有效的管道并行性。与现有的PP(Pipeline Parallelism)方法相比，DualPipe的管道气泡较少。更重要的是，它重叠了向前和向后过程的计算和通信阶段，从而解决了跨节点专家并行性引入的重大通信开销的挑战。其次，我们开发有效的跨节点全能通信内核来充分利用IB和NVLINK带宽，并保护专用于通信的流媒体多处理器（SMS）。最后，我们精心优化了训练过程中的内存足迹，从而使我们能够训练DeepSeek-v3，而无需使用昂贵的张量并行性（TP）。

- DualPipe and Computation-Communication Overlap

  ​	跨节点专家并行引入的通信开销导致效率低下的计算与通信率约为1：1，研究团队设计了一种创新的管道并行性算法，称为DualPipe，它不仅通过有效地重叠向前和向后计算通信阶段来加速模型培训，还可以减少管道气泡

  ​	在模型训练中，主要计算量来源于 **ATTENTION-O(L²)** 和 **MLP-O(H²)**。由于分布式训练，前向和后向计算均需要通信，通信包括 **dispatch**（将输入分到各个 weight、expert）和 **combine**（将各个 weight、expert 的输出结果聚合）两部分。在同一个 batch 中，通信和计算是交替进行的，这会导致效率低下。为此，DeepSeek V3 提出了双管路的方法，使得通信与计算能够并行。

  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310172250218.png)

  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310172316610.png)

  ​												   前向传播

  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310172342505.png)

  ​                                                                                                     反向传播

  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310172555618.png)

- Efficient Implementation of Cross-Node All-to-All Communication

- Extremely Memory Saving with Minimal Overhead

  - Extremely Memory Saving with Minimal Overhead

  - Exponential Moving Average in CPU

    EMA参数存储在CPU内存中，并在每个训练步骤之后异步更新

  - Shared Embedding and Output Head for Multi-Token Prediction

##### FP8 精度训练

​	受到低精度训练的最新进展的启发（Dettmers等，2022; Noune等，2022; Peng等，2023b），我们提出了一种利用FP8数据形式的细粒度混合精度训练框架，该框架利用FP8数据形式来训练DeepSeek-V3。尽管近期研究在推理量化方面取得了重大进展（Frantar等，2022; Xiao等，2023），但很少有研究证明在大规模语言模型预训练过程中应用低精度技术训练的有效性。

![image-20250310180715455](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310180715710.png)



#### 消融实验

##### MTP消融实验

![](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310175857918.png)

##### Auxiliary-Loss-Free 负载均衡策略消融实验

![image-20250310175929275](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250310175929406.png)
