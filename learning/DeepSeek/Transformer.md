# Transformer 架构深入学习



## Attention is All You Need

#### Sinusoidal位置编码（正弦位置编码）

##### 1、为什么需要位置编码？

- 单词位置和顺序是任何语言的重要组成部分，它们定义了句子的语法和实际语义。早期工作中，递归神经网络（RNN）按顺序逐字解析句子，实质上也考虑了单词的顺序。
- Transformer 架构为了避免 RNN 的递归方法消耗大量训练时间，放弃递归机制，采用多头注意力机制（从句子中捕获更长的依赖项）。由于句子中的所有单词同时流经 Transformer 的编码器/解码器。因此模型本身对每个单词没有感知位置信息，研究者开始思考如何将单词的顺序信息合并到模型中。
  - 思路一：[0,1]范围内的每个时间步长分配一个单词，0 表示第一个单词，1 表示最后一个单词。缺点：无法表示特定范围内存在多少单词
  - 思路二：为每个时间步长线性分配一个数字，即第一个单词“1”，第二个单词“2”。问题：位置编码可能值变得很大，并且实际推理时可能出现比训练中的句子更长的句子，不利于模型泛化。
- 理想情况下，位置编码应满足以下条件：
  - 为每个时间步（单词在句子中的位置）输出唯一的编码
  - 任意两个时间步长之间的距离在不同长度的句子中应该是一致的
  - 位置编码模型应该可以推广的更长的句子，即编码值应是有界的
  - 确定性

##### 2、Sinusoidal位置编码

![image-20250227173433738](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/image-20250227173433738.png)

 $p_{k,2i},p_{k,2i+1}$分别是位置 k 的编码向量的第 2i, 2i+1 个分量，d 是向量维度

- 泰勒展开

  ​	假设我们的模型为 $f(...,x_m,...,x_n...)$，其中标记出来的 $x_m,x_n$ 分别表示第 $m,n$ 个输入，不失一般性，设 $f$ 是标量函数。像 Transformer 这样的纯 Attention 模型，它是全对称的，即对于任意的 $m,n$，都有：

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228103825180.webp)

  - 正是由于这种全对称性，模型天然满足 $f(x,y) = f(y,x)$，模型无法对输入的位置信息进行感知。很直觉的想法是在输入的每个位置上加上不同的位置编码向量，进而打破对称性。

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228103834244.webp)

  - 为了简化问题并且进一步分析，我们首先只考虑 $m,n$ 这两个位置上的编码，将其视为扰动项，泰勒展开到二阶

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/640)

  - 展开式中第一项与位置无关，第 2~5 项只依赖于单一位置，可视为绝对位置信息的编码。第 6 项是同时包含 $p_m,p_n$ 的交互项，将其记为 $p_m^THp_n$，表示相对位置信息。

- 相对位置编码

  - 进一步简化分析，假设 $H = I$ 是单位阵，此时 $p_m^THp_n = p_m^Tp_n = <p_m,p_n>$ 是两个位置编码的内积，我们希望在这个简单例子中该项表达相对位置信息，即存在某个函数 $g$ 使得：

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228103849605.webp)

  - 这里 $p_m,p_n$ 是 d 维向量，进一步简化为 $d = 2$ 进行分析。对于 2 维向量，我们借助复数推导，即将向量 [x, y] 视为复数 $x+yi$，根据复数乘法的运算法则有：

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228103907348.webp)

  - 其中 $p_n^*$ 是 $p_n$ 的共轭复数，$Re[]$ 代表复数的实部。为了满足式（5），进一步假设存在复数 $q_{m-n}$ 使得：

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228103915413.webp)

  - 即两边同时取实部就得到了式（5）。为了求解该方程，使用复数的指数形式，即设 $p_m = r_me^{iϕ_m},p_n^* = r_ne^{-iϕ_n},q_{m-n} = R_{m-n}e^{iΦ_{m-n}} $有

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228103927397.webp)

  - 对于第一个方程，带入 n=m 得 $r_m^2 = R_0$，即 $r_m$ 是一个常数，简单起见设为 1；对于第二个方程，带入 n=0 得 $ϕ_m -ϕ_0 = Φ_m$，简单起见设 $ϕ_0 = 0$，那么$ϕ_m  = Φ_m$，即 $ϕ_m -ϕ_n = ϕ_{m-m}$，带入 $n = m-1$ 得 $ϕ_m -ϕ_{m-1} = ϕ_{1}$，那么 $\{ϕ_m\}$ 是一个等差数列。通解为 $m\theta$，因此我们就得到二维情形下位置编码的解为：

  ![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228110437286.webp)

  - 由于内积满足线性叠加性，所有更高维的偶数位置编码，我们可以表示为多个二维位置编码的组合（式5 的特解）：

  ![](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228110737824.webp)

##### 远程衰减

​	基于前面的假设，我们推导出了位置编码的形式（10），它跟标准的 Sinusoidal 位置编码（1）形式基本一样了，只是 $sin,cos$ 的位置有点不同。一般情况下，神经网络的神经元都是无序的，所以哪怕打乱各个维度，也是一种合理的位置编码，因此除了各个 $\theta_i$ 没确定下来外，式（10）和式（1）并无本质区别

​	式（1）的选择是 $\theta_i = 10000^{-2i/d}$，这个选择有什么意义呢？事实上，这个形式有一个良好的性质：**它使得随着 |m-n| 的增大，** **$<p_m,p_n>$有着趋于零的趋势**。按照我们的直观想象，相对距离越大的输入，其相关性应该越弱，因此这个性质是符合我们的直觉的。只是，明明是周期性的三角函数，怎么会呈现出衰减趋势呢？

​	这的确是个神奇的现象，源于高频振荡积分的渐近趋零性。具体来说，我们将内积写为：

![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228113715314.webp)

​	这样问题就变成了积分 $\int_0^1 e^{i(m-n)\theta_t}$ 的渐近估计问题了，使用 Mathematica 绘制积分图像，从图像中可以看出随着 m-n 增大，积分有明显衰减趋势：

![图片](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250228181006324.webp)

##### REFERENCE

[Transformer升级之路：Sinusoidal位置编码追根溯源](https://mp.weixin.qq.com/s/57iu8rPTXXG0jb2xxEVnTw)

#### Transformer 模型架构

Transformer由若干个编码器和解码器组成，如下图所示：

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301161007552.webp)

继续将Encoder和Decoder拆开，可以看到完整的结构，如下图所示：

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301161026893.webp)

​	可以看到Encoder包含一个Muti-Head Attention模块，是由多个Self-Attention组成，而Decoder包含两个Muti-Head Attention。Muti-Head Attention上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。

​	假设我们的输入包含两个单词，我们看一下Transformer的整体结构：

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301161202656.webp)

​	为了能够对Transformer的流程有个大致的了解，我们举一个简单的例子，将法语"Je suis etudiant"翻译成英文：

- **第一步**：获取输入句子的每一个单词的表示向量 $x$ ，$x$  由单词的Embedding和单词位置的Embedding 相加得到

  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301161331702.webp)

- **第二步**将单词向量矩阵传入Encoder模块，经过N个Encoder后得到句子所有单词的编码信息矩阵 ![\bm{C}](https://www.zhihu.com/equation?tex=%5Cbm%7BC%7D&consumer=ZHI_MENG) ，如下图。输入句子的单词向量矩阵用 $x \in R^{n \times d}$ 表示，其中 $n$ 是单词个数，$d$表示向量的维度（论文中 $d=512$)。每一个Encoder输出的矩阵维度与输入完全一致。

  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301161540229.webp)

- **第三步**：将Encoder输出的编码矩阵 ![\bm{C}](https://www.zhihu.com/equation?tex=%5Cbm%7BC%7D&consumer=ZHI_MENG) 传递到Decoder中，Decoder会根据当前翻译过的单词 $1~i$ 翻译下一个单词 $i+1$，如下图所示。
  
  ![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301161632406.webp)	上图Decoder接收了Encoder的编码矩阵，然后首先输入一个开始符 "<Begin>"，预测第一个单词，输出为"I"；然后输入翻译开始符 "<Begin>" 和单词 "I"，预测第二个单词，输出为"am"，以此类推。这是Transformer的大致流程，接下来介绍里面各个部分的细节。

##### Self-Attention

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301162834052.webp)

上图表示 Self-Attention 结构，$Q、K、V$ 矩阵通过输入矩阵 $X$ 和权重矩阵 $W^Q,W^K,W^V$ 相乘得到

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301163036604.webp)

$Q、K、V$ 通过如下计算得到 Self-Attention 的输出：

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250301163145498.webp)

- $W^Q$ 将 token 映射为与该 token 相关的一个 Query（Q），即询问与该 token 相关的其他 token 是什么
- $W^K$ 将 token 映射为上述 Query 的回答（K），同时 $Q \times K^T$ 表示查询与回答间的匹配度或者相似性，即注意力模式表。
- 除以 $\sqrt{d_k}$ 是为了使得一组 Query 的回答分布均值近似为0，避免趋于极值
- Softmax 过程对注意力模式表中一组 Query 对应的回答进行归一化操作，得到注意力权重（GPT 类生成式任务会在此过程进行掩码操作，从而避免后面的 token 影响前面的 token，具体为将掩码位置置为 $-00$ 再进行 Softmax）
- $W^V$ 将 token 映射为语义空间内的一个偏移值，该偏移值 value 与上述注意力权重的乘积表示注意力模式感知到的前面的 token 对当前 token 的语义影响，表示为在语义空间内前后语义之间的偏移向量。
- $ X + Softmax(Q \times K^T)*V$ 一方面表示残差结构，另一方面也可以解释为原始 token 的语义向量加上语义偏移向量

Multi-Head Attention

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303102729217.png)

多个感知头目标是希望学习到多种注意力模式
$$
MultiHead(Q,K,V)=Concat(head_1,…,head_h)W^O
$$
其中，
$$
head_i=Attention(QW_i^Q,KW^K_i,VW^V_i)
$$

 ##### 交叉注意力层

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303151113077.png)

- Q：解码器自注意力层的输出
- K：编码器的输出
- V：编码器的输出 

![请添加图片描述](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303153231659.png)

#### MHA （Multi-Head Attention，多头注意力机制）

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303153630817.png)	相当于多个单头注意力的拼接，在 LLAMA2-7b 中 $h = 4096, n=32, d_k=d_v=128$， LLAMA2-70b 中 $h = 8192, n=64,  d_k=d_v=128$

​	推理过程中，随着输入文本的不断增多，每次都需要计算历史上下文的$Q,K,V$矩阵，为了加速推理过程，优化思路为将历史$K,V$矩阵存储下来，减少重复运算，即 $KV cache$ 。

​	 $KV cache$ 也带来了新的问题，随着对话上下文的增加， $KV cache$ 占用的存储空间越来越大，于是便提出了 Multi-Query Attention（MQA）。

#### Multi-Query Attention（MQA）

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303154310746.png)

​	MQA的思路很简单，直接让所有Attention Head共享同一个K、V，用公式来说，就是取消MHA所有的$k,v$的上标$^{(s)}$。使用MQA的模型包括[PaLM](https://arxiv.org/pdf/2204.02311)、[StarCoder](https://papers.cool/arxiv/2305.06161)、[Gemini](https://papers.cool/arxiv/2312.11805)等。很明显，MQA直接将KV Cache减少到了原来的1/h1/h，这是非常可观的，单从节省显存角度看已经是天花板了。

​	效果方面，目前看来大部分任务的损失都比较有限，且MQA的支持者相信这部分损失可以通过进一步训练来弥补回。此外，注意到MQA由于共享了K、V，将会导致Attention的参数量减少了将近一半，而为了模型总参数量的不变，通常会相应地增大FFN/GLU的规模，这也能弥补一部分效果损失。

#### GQA（Grouped-Query Attention）

​	然而，也有人担心MQA对KV Cache的压缩太严重，以至于会影响模型的学习效率以及最终效果。为此，一MHA与MQA之间的过渡版本[《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》](https://papers.cool/arxiv/2305.13245)（**G**rouped-**Q**uery **A**ttention）应运而生。

​	GQA的思想也很朴素，它就是将所有Head分为g个组（g可以整除h），每组共享同一对K、V，用数学公式表示为：

![image-20250303155735914](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303155735985.png)

​	这里的$⌈⋅⌉$是上取整符号。GQA提供了MHA到MQA的自然过渡，当 $g=h$ 时就是MHA，$g=1$ 时就是MQA，当 $1<g<h$ 时，它只将KV Cache压缩到 $g/h$，压缩率不如MQA，但同时也提供了更大的自由度，效果上更有保证。GQA最知名的使用者，大概是Meta开源的[LLAMA2-70B](https://llama.meta.com/llama2/)，以及[LLAMA3](https://llama.meta.com/llama3/)全系列，此外使用GQA的模型还有[TigerBot](https://papers.cool/arxiv/2312.08688)、[DeepSeek-V1](https://papers.cool/arxiv/2401.02954)、[StarCoder2](https://papers.cool/arxiv/2402.19173)、[Yi](https://papers.cool/arxiv/2403.04652)、[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)、[ChatGLM3](https://github.com/THUDM/ChatGLM3)等，相比使用MQA的模型更多（ChatGLM虽然在它的介绍中说自己是MQA，但实际是 $g=2$ 的GQA）。

​	在llama2/3-70B中，GQA的 $g=8$，其他用了GQA的同体量模型基本上也保持了这个设置，这并非偶然，而是同样出于推理效率的考虑。我们知道，70B这个体量的模型，如果不进行极端的量化，那么不可能部署到单卡（A100/H100 80G）上。单卡不行，那么就能单机了，一般情况下一台机可以装8张卡，刚才我们说了，Attention的每个Head实际上是独立运算然后拼接起来的，当 $g=8$ 时，正好可以每张卡负责计算一组 $K、V$ 对应的Attention Head，这样可以在尽可能保证 $K、V$ 多样性的同时最大程度上减少卡间通信。

#### MLA（Multi-head Latent Attention）

![mla2](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303162943805.jpeg)	DeepSeek进一步优化，推出了多头潜在注意力机制（MLA）。MLA旨在进一步缩小KV缓存的大小，同时在性能上超越之前提到的注意力机制（包括MHA)。它通过将KV缓存压缩到低维潜在空间，成功将缓存大小减小了93.3% ！下面我们详细看看它是如何做到的。

1. **低秩键值联合压缩**：MLA不会像传统方式那样计算和存储每个令牌的键和值，而是使用下投影矩阵$W(DKV)$把它们压缩成潜在向量$C(KV)$。在推理时，再通过每个头的上投影矩阵$W(UK)$（用于键）和$W(UV)$（用于值）从这个潜在向量中重建KV对。为了降低计算成本，MLA还进行了巧妙的优化：把矩阵$W(UK)$合并到$W(Q)$中，这样就不用显式计算键$K(i)$了；把矩阵$W(UV)$合并到$W(O)$中，也就无需显式计算值$V(i)$了。

![img](https://segmentfault.com/img/remote/1460000046119622)

![img](https://segmentfault.com/img/remote/1460000046119623)

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164200123.png)

![img](https://segmentfault.com/img/remote/1460000046119625)

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164159842.png)

![img](https://segmentfault.com/img/remote/1460000046119627)

1. **查询的低秩压缩**：MLA对查询也进行了类似的压缩。

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164201285.png)

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164201965.png)

使用下投影矩阵$W(DQ)$将查询压缩成潜在表示$C(Q)$，需要时再用上投影矩阵$W(UQ)$进行重建。虽然这样做不会减少KV缓存的大小，但能降低训练期间的激活内存使用。（激活内存是训练过程中前向传播时用于存储中间激活的内存，反向传播计算梯度时会用到这些激活。）在使用MHA训练时，每一层都会在内存中显式计算和存储查询，且数量会随着层数线性增加。而在MLA中，只存储查询的压缩表示，减少了反向传播时存储的总激活量。不过要注意，在推理时，每个令牌计算一次查询后就会丢弃，不会存储用于反向传播的激活。所以，查询压缩主要是提高了训练效率，对推理性能没有影响。

![img](https://segmentfault.com/img/remote/1460000046119630)

研究人员尝试在MLA中使用旋转位置嵌入（RoPE）来加入令牌位置信息，

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164202691.png)

可这遇到了一些问题。在深入探讨之前，我们先来了解一下位置编码在大语言模型中的工作原理。

Transformer架构并行处理令牌，这虽然让它比RNN在计算上更有优势，但也导致它对令牌顺序不敏感。比如，“The cat sits on the mat.”和“The mat sites on the cat.”这两句话，对Transformer来说没什么区别。但在语言处理中，顺序很重要，所以需要添加位置信息。位置嵌入主要有两种类型：绝对位置嵌入，给每个令牌根据其位置分配唯一编码；相对位置嵌入，编码的是令牌之间的相对距离，而不是绝对位置。这两种嵌入又可以分为固定的（用数学函数计算）和可学习的（模型训练时通过反向传播更新参数）。在原始的Transformer论文中，作者使用的是固定的绝对位置嵌入，通过交替的正弦和余弦函数在偶数和奇数维度上计算位置嵌入$PE$，公式为：$PE(pos,2i)=sin(pos/10000^{2i/d(model)})$，$PE(pos,2i + 1)=cos(pos/10000^{2i/d(model)})$，其中$pos$是令牌索引，$i$是令牌嵌入维度的索引，$d(model)$是总令牌嵌入维度。这些位置嵌入和令牌嵌入维度相同，可以直接相加后再输入Transformer进行处理。

后来，2023年的一项研究提出了旋转位置嵌入（RoPE），这是一种在注意力机制中直接编码绝对和相对位置的新方法。RoPE不会像之前那样添加位置嵌入，而是根据令牌的位置旋转令牌嵌入。具体来说，对于位置$m$处维度为$d$的令牌嵌入$x(m)$，分别使用权重矩阵$W(q)$和$W(k)$将其转换为查询向量$q(m)$和键向量$k(n)$ 。在进行自注意力计算前，使用与位置相关的旋转矩阵$R(m)$对这些向量进行旋转。$R(m)$会独立作用于$q$和$k$中的每对维度。以二维向量为例，旋转矩阵$R(m)$定义为：$\begin{bmatrix}cos(m\theta)& -sin(m\theta)\\sin(m\theta)&cos(m\theta)\end{bmatrix}$，这个矩阵会将向量逆时针旋转，旋转角度与位置$m$成正比，为$m\theta$（$d = 2$时，$\theta = 1$ ）。对于更高维的向量（假设维度为偶数），会将相邻的维度两两配对，分别进行二维旋转。通过这种方式，RoPE可以让注意力分数编码令牌的相对位置，而且还能体现出相距较远的令牌之间联系的相对重要性低于较近的令牌。

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164202944.png)

### 为什么RoPE与MLA不兼容？

回到MLA，我们知道它通过创建键和值的潜在压缩表示$C(KV)$来减少内存使用和提高推理效率。而RoPE需要在计算注意力分数前，根据位置信息用旋转矩阵$R(m)$旋转查询和键。但由于MLA存储的是压缩的键值缓存，不是完整的键，所以如果应用RoPE，每次生成新令牌时都得重新计算所有之前的键，这就破坏了使用压缩KV表示带来的效率提升。另外，之前为了优化，MLA把键向上投影矩阵$W(UK)$合并到了$W(Q)$中，而RoPE的旋转操作会导致矩阵乘法不满足交换律，使得$W(UK)$无法像原来那样与$W(Q)$解耦和合并。

### 那么RoPE如何在MLA中使用呢？

在MLA中，研究人员引入了一种新方法——解耦旋转位置嵌入（Decoupled Rotary Position Embedding）。首先，计算两种类型的键：一种是之前讨论过的压缩键$K(C)$；另一种是位置敏感或解耦键$K(R)$，它是未压缩的键，用于存储应用RoPE所需的位置信息。查询也会进行类似计算，得到潜在查询$Q(C)$和用于RoPE的位置敏感或解耦查询$Q(R)$。这些计算是在推理时进行的，不会存储。这种方法既保留了低秩KV压缩的优势，又能通过单独存储$K(R)$来应用位置敏感变换，还不会影响RoPE的注意力计算。在这种方式下，每个令牌需要缓存$K(R)$和$C(KV)$，总共缓存$[d(c) + d(h)(R)]×L$个元素（$d(c)$是潜在密钥维度，$d(h)(R)$是解耦密钥的每个头维度，$L$是MLA中的层数） ，相比传统Transformer模型，效率大大提高。最后，使用压缩和位置敏感的查询和键来计算注意力分数，得到最终输出。

![img](https://raw.githubusercontent.com/VirtualCoder0/tuchuang/main/gongwei/20250303164203583.png)