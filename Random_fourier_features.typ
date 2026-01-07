#import "@preview/ori:0.2.3": *
本Jupyter notebook是一个用于求解高波数 Helmholtz 方程的混合深度学习框架。

以下是完整的代码结构介绍：

#theorem-box[
  === 1. 数值求解层：真值数据生成

==== `solve_helmholtz_fd(lam, mu, N)`

- *功能*：采用有限差分法生成高精度的真值数据。

- 实现细节

	- $lambda=205,mu=3$ , $u(-1)=0,u(1)=0$
	- 构建三对角稀疏矩阵求解边值问题 $u'' + lambda^2 u = sin(mu x)$。
	- 返回网格点 $x$ 和对应的数值解 $u$，用于后续神经网络的训练标签和验证基准。
]
#theorem-box[
  
=== 2. 通用求解器基类：训练框架

==== `Solver(nn.Module, ABC)`

- *功能*：抽象类，定义了物理信息神经网络的通用训练流程、数据管理和损失计算。

- 核心组件
	- *数据初始化*：自动调用 `solve_helmholtz_fd` 生成数据，并采样边界点和内部点。
	- *抽象方法 `net_u`*：定义网络前向传播接口，需要子类具体实现。
	- *物理残差 `net_f`*：利用 PyTorch 自动微分计算 PDE 残差。
	- *双优化器策略 `fit`*：实现动态概率切换机制，交替使用 `optimizer_u` (拟合数据) 和 `optimizer_f` (满足物理方程) 进行训练，避免了多尺度下PDE_loss和Data_loss之间的权重调节的麻烦。
	- *评估与可视化*：`predict` 方法用于推理，`plot` 方法用于绘制结果和损失曲线。
]

=== 3. 神经网络架构：具体模型实现

这一部分实现了三种不同的神经网络架构，均继承自 `Solver` 基类。
#definition-box[
  
==== A. 基础全连接网络

==== `DNNSolver(Solver)`

- *架构*：标准的深度神经网络。

- 结构

	- 输入层：接收坐标 $x$。
	- 隐藏层：由 `Linear` 层和 `Tanh` 激活函数交替组成。
	- *输入归一化*：在 `net_u` 中将输入坐标映射到 $[-1, 1]$ 区间。

]
#definition-box[
==== B. 多尺度傅里叶特征网络

==== `FourierFeatureSolver(Solver)`


- 核心组件



	- `MultiScaleFourierLayer`

		- 生成基础频率，并通过 $2^k$ 倍频机制生成多尺度频率。
		- 将原始坐标映射到高维傅里叶特征空间。
		- 引入可学习的权重 `beta` 和归一化尺度 `scale`。

- *整体结构*：傅里叶特征层 + 多层感知机 (ReLU 激活)。
]
#definition-box[==== C. 基于注意力机制的傅里叶网络

==== `AttentionFourierSolver(Solver)`


- 核心组件

	

	- *`MultiScaleFourierLayer`*：同上，作为特征提取前端。

	- `CrossAttentionResidualBlock`



		- *跨注意力机制*：将傅里叶特征重塑为 Token 序列。
		- *Query-Key 交互*：使用可学习的 Query 向量聚合所有特征 Token 的信息。
		- *残差更新*：通过多层注意力和非线性激活提炼特征。

- *整体结构*：傅里叶特征层 $->$ 注意力特征融合模块 $->$ 输出层。
]
#caution-box[
  由于本notebook中在不同条件下(如是否对原始数据尺度缩放,是否使用PDE_loss)进行了多组实验,这里我们仅就使用标准化输入和不使用PDE_loss训练的3个模型进行展示。
]
#v(-0.5em)
#grid(
  columns: 3,
  align(center, image("dnn1.png", width: 100%)),
  align(center, image("fourier1.png", width: 100%)),
  align(center, image("attention1.png", width: 100%)),
)
#grid(
  columns: 3,
  align(center, text("          图1: 基础全连接网络")),
  align(center, text("          图2: 多尺度傅里叶特征网络")),
  align(center, text("          图3: 基于注意力机制的傅里叶网络")),
)
从图上可以看出，基础全连接网络频谱偏差严重,而多尺度傅里叶特征网络和基于注意力机制的傅里叶网络均能较好地拟合高频信号,其中基于注意力机制的傅里叶网络拟合效果最佳。
#pagebreak()
以下代码反映了这三种模型的训练代码配置：
#corollary-box[
  ```python
layers=[1,16,32,32,16,1]
dnn_solver=DNNSolver(layers)
dnn_solver.if_scale_u=True 
dnn_solver.physical_information=False
history=dnn_solver.fit(10000)
``` 
]
#corollary-box[
  ```python
layers=[16,8,8,8,1]
base_feature_dim=32
num_scales=8
fourier_solver=FourierFeatureSolver(layers,base_feature_dim,num_scales)
fourier_solver.if_scale_u=True
fourier_solver.physical_information=False
history=fourier_solver.fit(10000)
```
]
#corollary-box[ ```python
attention_fourier_solver=AttentionFourierSolver(
    input_dim=1,
    base_feature_dim=32,
    num_scales=8,
    N_tok=32,
    d_q=8,
    num_attn_layers=4
)
attention_fourier_solver.physical_information=False
attention_fourier_solver.if_scale_u=True    
attention_fourier_solver.fit(10000)
```]
三种模型的Loss曲线如下图所示：
#grid(
  columns: 3,
  align(center, image("dnn2.png", width: 100%)),
  align(center, image("fourier2.png", width: 100%)),
  align(center, image("attention2.png", width: 100%)),
)
#grid(
  columns: 3,
  align(center, text("          图1: 基础全连接网络")),
  align(center, text("          图2: 多尺度傅里叶特征网络")),
  align(center, text("          图3: 基于注意力机制的傅里叶网络")),
)
从图上可以看出,随机傅里叶网络的Loss比基础全连接网络小一个数量级,而基于注意力机制的傅里叶网络的Loss又比随机傅里叶网络小二到三个数量级,说明引入傅里叶特征和注意力机制都能有效提升神经网络拟合高频信号的能力。