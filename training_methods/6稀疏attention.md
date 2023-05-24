# 7 稀疏attention机制

在标准的Transformer中，每个token都需要和其他的所有token做运算，但是有研究人员观察到针对一个训练好的Transformer，其中的注意力矩阵通常非常稀疏。因此，通过结合结构偏差来限制每个Q关注的Q-K对的数量，可以降低计算复杂度。在该限制下，只需要根据预定义的模式计算Q-K对的相似度即可。

![](https://pic1.zhimg.com/v2-7c4fbe5ac6f80b06c882326d959a63d0_r.jpg)

上述公式得到的结果是一个非归一化的矩阵，在具体的实现中，矩阵中的一般不会被存储。

从另一个角度来看，标准的注意力可以看作是一个完整的二部图，其中每个Q接收来自所有存储节点的信息并更新其表示。稀疏注意可以看作是一个稀疏图，其中删除了节点之间的一些连接。我们将确定稀疏连接的度量分为两类：*基于位置的稀疏注意*和 *基于内容的稀疏注意* 。

### **基于位置的稀疏注意力**

在基于位置的稀疏注意力中，注意力矩阵根据一些预先定义的pattern进行限制。虽然这些稀疏模式有不同的形式，但本文发现其中一些可以分解为原子类型的稀疏pattern。本文首先确定一些原子类型的稀疏pattern，然后描述这些pattern是如何在一些现有的工作应用的。最后本文介绍了一些针对特定数据类型的扩展稀疏pattern。

**原子稀疏注意力（Atomic Sparse Attention）**

主要有五种原子稀疏注意模式，如下图所示。

* **Global Attention.** 为了缓解在稀疏注意中对长距离依赖性建模能力的退化，可以添加一些全局节点作为节点间信息传播的中心。这些全局节点可以attend到序列中的所有节点，并且整个序列也可以attend到这些全局节点，其中注意矩阵如图4（a）所示。
* **Band Attention.** 又称之为滑动窗口注意力或局部注意力。由于大多数数据都具有很强的局部性，因此很自然地会限制每个Q去关注其邻居节点。这种稀疏模式被广泛采用的一类是Band Attention，其中注意矩阵如图4（b）所示。
* **Dilated Attention.** 与扩张的CNN类似，通过使用具有间隙的扩张窗口，可以潜在地增加Band Attention的感受野，而不增加计算复杂度。其中注意矩阵如图4（c）所示。这可以很容易地扩展到跨越式的注意力机制，窗口大小不受限制，但 需要设置为更大的值。
* **Random Attention.** 为了增加非局部交互的能力，对每个Q随机抽取一些边，如图4（d）所示。这是基于随机图（Erdős–Rényi随机图）可以具有与完全图相似的谱性质，从而通过在随机图上的游走可以得到更加快速的mixing时间。
* **Block Local Attention.** 这类注意力机制将输入序列分割成若干个互不重叠的查询块，每个查询块与一个本地存储块相关联。查询块中的所有Q只涉及相应内存块中的K。图4（e）展示了存储器块与其对应的查询块。

![](https://pic2.zhimg.com/v2-959b0f59d856dd0cb164a3ab8b03b5bd_r.jpg)

**复合稀疏注意力（Compound Sparse Attention）**

现有的稀疏注意力通常由以上原子模式中的一种以上组成。图5显示出了一些代表性的复合稀疏注意模式。

![](https://pic4.zhimg.com/v2-8f3fd690e9ac14017ec9cbaadcadce8f_r.jpg)

* **Star Transformer**结合了Band Attention和Global Attention.。具体来看Star Transformer仅包括一个全局节点和一个宽度为3的Band Attention，其中任何一对非相邻节点通过一个共享的全局节点连接，相邻节点之间直接连接。这种稀疏模式可以在节点间形成星形图。
* **Longformer**结合了Band Attention和internal global-node attention，选择做分类的[CLS]token以及问答任务中的所有的问题tokens作为全局节点。此外还用扩大的Dilated Attention代替上层的一些Band Attention头，以增加感受野而不增加计算量。
* **ETC** （Extended Transformer Construction）作为和Longformer同时出现的工作，ETC结合了band attention和external global-node attention。ETC还使用了一种[MASK]机制，用于处理结构化输入和调整对比预测编码（CPC）以用于预训练。
* 除了band attention和external global-node attention之外，**Big bird**还使用额外的随机注意力来接近完全注意力。相应的理论分析表明，使用稀疏编码器和稀疏解码器可以模拟任何类型的图灵机，这同时解释了这些稀疏注意模型的有效性。
* **Sparse Transformer**使用了一种因式分解的注意力机制，其中针对不同类型的数据设计了不同的稀疏模式。对于具有周期性结构的数据（例如图像），它使用了band attention和strided attention的组合。而对于没有周期结构的数据（如文本），则采用block local attention与global attention相结合的组合，全局节点来自输入序列中的固定位置。

**扩展的稀疏注意力(Extended Sparse Attention)**

除了上述模式之外，一些现有的研究还探索了特定数据类型的扩展稀疏模式。

对于文本数据，**BP Transformer**构造了一个二叉树，其中所有标记都是叶节点，内部节点是包含许多标记的span节点。图中的边是这样构造的：每个叶节点都连接到它的邻居叶节点和更高级别的span节点，这些节点包含来自更长距离的token。这种方法可以看作是全局注意的一种扩展，其中全局节点是分层组织的，任何一对token都与二叉树中的路径相连接。图6（a）展示出了该方法的抽象视图。

对于视觉数据也有一些扩展。Image Transformer探索了两种类型的注意力模式：

* 按光栅扫描顺序展平图像像素，然后应用block local稀疏注意力;
* 2D block local注意力，其中查询块和存储块直接排列在2D板中，如图6(b)所示。作为视觉数据稀疏模式的另一个例子，Axial Transformer在图像的每个轴上应用独立的注意力模块。如图6（c）所示，每个注意模力块沿一个轴混合信息，同时保持沿另一个轴的信息独立。这可以理解为以光栅扫描水平和垂直展平的图像像素，然后分别使用图像宽度和高度的间隙来应用跨越式注意力。

![](https://pic4.zhimg.com/v2-2b679419024959352402563d99f17207_r.jpg)

### **基于内容的稀疏注意力**

另一个方向的工作是基于输入内容创建稀疏图，即构造输入中的稀疏连接时是有条件的。

构造基于内容的稀疏图的简单方法是选择那些可能与给定Q具有较大相似性分数的K。为了有效地构造稀疏图，可以将其看做Maximum Inner Product Search （MIPS）问题，即在不计算所有点积项的情况下，通过一个查询Q来寻找与其具有最大点积的K。**Routing Transformer**使用了k-means聚类——对查询queries和keys在同一簇质心向量集合上进行聚类。每个查询Q只关注与其属于同一簇内的keys。在训练的时候，会使用分配的向量指数移动平均数除以簇的数量来更新每个簇的质心向量。

![](https://pic3.zhimg.com/v2-88fb466c6144f2c872b47c25123fa786_r.jpg)

|μ| 表示的是 这个簇中包含的向量的数量， λ∈(0,1) 是一个可学习的超参数。

假设 Pi 表示第 i 个查询涉及的Key的索引集合，在Routing Transformer中的 Pi 表示为：

![](https://pic4.zhimg.com/80/v2-ab935c7f6d063e1087d8818f8d117b5f_1440w.webp)

**Reformer**使用了局部敏感哈希（LSH）算法来为每一个query选择对应的K-V对，LSH注意允许每个token只关注同一散列桶中的token。其基本思想是使用LSH函数将Query和Key散列到多个bucket中，相似的项有很高的概率落在同一个bucket中。

具体来看，他们使用随机矩阵方法作为LSH的函数。假设 b 表示bucket的数量，给定随机矩阵 R 大小为 [Dk,b/2] ，则LSH函数的计算公式为：

![](https://pic3.zhimg.com/v2-d6ccf8133acfa50a0182b9ab7a926ae2_r.jpg)

LSH允许Query只关注具有索引的K-V对：

![](https://pic3.zhimg.com/80/v2-14f3ac0cffdcb69fea13decfcc75fc4a_1440w.webp)

* **稀疏自适应连接** （Sparse Adaptive Connection，SAC）将输入序列视为一个图，并学习构造注意力边，以使用自适应稀疏连接提高特定任务的性能。SAC使用LSTM边缘预测器来构造token之间的边。在没有遍的情况下，采用强化学习的方法训练边缘预测器。
* **Sparse Sinkhorn Attention** （SSA）首先将查询Q和键K分成几个块，并为每个查询块分配一个键块。每个查询Q只允许关注分配给其相应键块中的键。键块的分配由排序网络控制，该网络是一个使用Sinkhorn归一化产生的双随机矩阵，该矩阵用作分配时的排列矩阵。SSA使用这种基于内容的block sparse attention以及上面中介绍的block local attention以增强模型的局部建模能力。

### 参考

[Transformer长大了，它的兄弟姐妹们呢？（含Transformers超细节知识点）](https://zhuanlan.zhihu.com/p/381899756)

[多种Attention之间的对比(下）](https://zhuanlan.zhihu.com/p/336484155)

[为节约而生：从标准Attention到稀疏Attention](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247498604&idx=1&sn=178bcb8827162a58a04d4ac131d03408&chksm=96ea24eca19dadfa27b5bedb58ddd0d6924d4b9e410bb523636e1a7834dd0f5cdb5ec7a2e45b&mpshare=1&scene=1&srcid=1211xlsVVTWl8oBM0HEjlwk6&sharer_sharetime=1607702340523&sharer_shareid=6de0f8e01e8c0625eda8ab4e997af088&key=ec5faba1391b624a81f012ed8c2d4d2434a6abbc0262b3172078a063493c8bf9482ca39875577a46da3b18bb8d4b315fc95b59b1d88414ef92efc739e62158b8f0392dec87c4a8d6d42f8512e548cb6aa9d9cd2f722275a30a40711f6aa8d0c170cd2d5c320dfaeb89575fca886e972106a20736944c8da7cb0e321515d86f91&ascene=1&uin=MjAwMzE0NjU0NQ%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=AUNZm11axeAFWi9%2BGO9u4lk%3D&pass_ticket=4YK02S64Ga8SiOQnwW2CZhb8lPVQDw9ZQhTUlcfnYY5e6IvbQAiTI%2FVap96j5Xy9&wx_header=0)

[线性self-attention的漫漫探索路（1）---稀疏Attention](https://zhuanlan.zhihu.com/p/469853664)
