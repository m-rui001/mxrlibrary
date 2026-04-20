#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.1"
#import "@preview/fletcher:0.5.8" as fletcher: edge, node
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.4.1": *
#import cosmos.clouds: *
#import "@preview/wrap-it:0.1.1": *
#show: show-theorem
#let cetz-canvas = touying-reducer.with(
  reduce: cetz.canvas,
  cover: cetz.draw.hide.with(bounds: true),
)
#let fletcher-diagram = touying-reducer.with(
  reduce: fletcher.diagram,
  cover: fletcher.hide,
)
#show: university-theme.with(
  aspect-ratio: "16-9",
  // align: horizon,
  // config-common(handout: true),
  config-common(frozen-counters: (theorem-counter,)), // freeze theorem counter for animation
  config-info(
    title: [Overcoming Spectral Bias via Cross-Attention实验报告2 ],

  ),
)
#set par(first-line-indent: (amount: 2em, all: true))
#title-slide()

== 有限差分法和自动微分
由于在训练过程中,模型需要计算二阶导数,使用自动微分极大降低了训练效率,因此我们尝试使用有限差分法来近似计算二阶导数.

我们认为神经网络的输出近似为真实解$u(x)$，我们使用七点中心差分计算二阶导数:
$
   u''_i approx (2u_(i-3) - 27u_(i-2) + 270u_(i-1) - 490u_i + 270u_(i+1) - 27u_(i+2) + 2u_(i+3)) / (180h^2) 
$
误差项为
$(-h^6/560 u^((8))(xi))$
,在数量级上$u^((8))~(100pi)^8 sin(100pi x)$，不妨认为$abs(u^((4))(xi))<=(100pi)^4$.在[-1,1]上取batch_size=10000个点，此时$h=2/10000$,计算得
$abs((-h^6/560 u^((8))(xi)))<=1.1 times 10^(-5)$.

这个误差远小于$u(x)$低频项$sin(0.1 pi x)$的振幅$(0.1 pi)^2approx 0.099$.

实验显示，使用差分法后，模型的训练速度提升了一倍以上，Loss曲线和使用自动微分时基本一致，证明了有限差分法在该问题上的有效性。

然而对于二维高频Poisson方程的求解，使用差分法计算二阶导数时，达到相同精度需要的配点数是$10000^2$.实验中用10000个配点的训练中，loss无法有效下降，因此使用自动微分更合适。

下面我们解析以下代码:
```python    
def diff_operator(self, X, model):
        x = X[:, 0:1].requires_grad_(True)
        y = X[:, 1:2].requires_grad_(True)
        u = model(torch.cat([x,y], dim=1))  # (N, 1) 
        laplacian = torch.zeros_like(u)
        for i in range(X.shape[1]):
            grad_ui = torch.autograd.grad(
                u, [x, y][i],
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            hess_ii = torch.autograd.grad(
                grad_ui, [x, y][i],
                grad_outputs=torch.ones_like(grad_ui),
                create_graph=True,retain_graph=True)[0]
            laplacian += hess_ii
        return -laplacian  
```
输入 $bold(X) in bb(R)^(N times 2)$，输出$bold(u) in bb(R)^(N times 1)$,$bold(u)$关于$bold(X)_k$的雅可比矩阵为

$cal(J) in bb(R)^(N times 1 times N times 1)$,$cal(J)_(i,1,j,1)=(partial bold(u)^((i)))/(partial bold(X)_(k)^((j)))$(省略k,并且认为$bold(u)=bold(u)_1$),

$bold(u)_(i,1)$仅与$bold(x)_(i,k)$有关.
因此可以简化为
$
cal(J)_(i,1,j,1)=cases((partial bold(u)^((i)))/(partial bold(X)_(k)^((j)))\, quad i=j,0 \, quad i!=j)
$
$v=$torch.ones_like($U$) , $v in bb(R)^(N times 1)$,\
则torch.autograd.grad(u,[x, y][i],grad_outputs=torch.ones_like(u)) 的计算过程为:
$
  bold(g)^((j)) = bold(v)^top dot cal(J)= sum_(i,1) bold(v)_(1)^((i)) dot cal(J)_(i,1,j,1) = sum_i 1 dot  cal(J)_(i,1,j,1) 
  = (partial bold(u)^((j)))/(partial bold(X)_(k)^((j)))
$ 
其中$k=1,2, quad j = 1,2 dots N$.

$text("完整写出后")
bold(g)_k = vec((partial u(bold(X)^((1))))/(partial bold(X)_k^((1))),
(partial u(bold(X)^((2))))/(partial bold(X)_k^((2))),
dots.v,
(partial u(bold(X)^((N))))/(partial bold(X)_k^((N))),
)
$

$bold(g)_k in bb(R)^(N times 1)$ 对$bold(X)_k in bb(R)^(N times 1)$的雅可比矩阵为：
$ cal(J)_(i,1,j,1)=cases((partial bold(g)_k^((i)))/(partial bold(X)_k^((j)))\, quad i=j , 
0\, quad i!=j) 
$

则hess_ii的计算过程为:
$
  bold(h)_(k)^((j)) =bold(v)^top dot cal(J)= sum_(i) bold(v)^((i)) dot cal(J)_(i,1,j,1) = 1 dot cal(J)_(j,1,j,1) = (partial bold(g)_k)^((j))/(partial bold(X)_k^((j))) 
$
其中$k=1,2, quad j = 1,2 dots N$.
#align(center)[
$- Delta u = - sum_(k=1)^2 bold(h)_k$
]
#pagebreak()
== 采样策略
对于 $u=sin(50 x^2)+sin(50 y^2)$的回归任务， 
当采样方式为固定的网格点时，模型能部分学习到高频信息，而当采样方式变为动态从全求解域完全随机采样,batch为整个采样得到的训练集时，模型输出一个随训练而震荡的平面。

经过分析，这时由于对于任意一个随机batch的MSE Loss, 神经网络输出常数 $hat(u)=bb(E)(u_i)$ 时可以使Loss较小，而当该batch改变后, 模型仍然可以通过调整偏置项快速适应新的batch的均值，导致模型输出一个震荡的平面。

这本质上是由于，每次重采样都相当于改变了训练集，对于模型这也就成了另外一个回归任务，而目标$u$的频率过高，模型没有足够的能力在较短的epoch内学到可迁移到下一个batch的知识.
== 该方法的不足
=== 依赖先验频谱信息
$omega_(text("base"))~cal(N)(0,sigma^(-2) bb(I))$中$sigma$的选取依赖先验知识，该参数很大地影响了求解效果。当采样频率不能覆盖目标函数的主要频率时，几乎无法收敛。

当我们根据目标函数的频谱特征选择了合适的先验频率后，模型在训练前期能实现误差的快速下降，但训练后期误差出现震荡。论文中误差曲线一直有下降趋势是因为学习率衰减的缘故。

=== Cross Attention使得计算量激增
加入Cross Attention后，同一任务的训练时间为原来的五倍以上。论文令RFF-CA和RFF-NN网络训练同样的epoch数比较性能，这是不公平的。

pytorch有内置的MultiheadAttention模块，然而该模块中所有优化版本的注意力机制（Flash Attention 和 Memory Efficient Attention）都不支持高阶导数的计算，必须使用较慢的Math SDP。

=== 自适应性频率增强策略
自适应性频率增强策略提取后验频率依赖对网络输出做DFT，这本身就依赖神经网络已经部分学到后验频率。其次该方法依赖人工区分哪些是已有频率，哪些是需要增强的频率，不够智能。

在频谱增强任务中，离散傅里叶变换（DFT）的归一化功率谱可视为频率轴上的离散概率质量函数（PMF）。若直接对其实行高斯核密度估计（KDE）并采样，所得连续概率密度函数（PDF）可以作为后验概率的采样分布。
#align(center)[#image("assets/kde.png",height:200pt,width: 500pt,fit:"stretch")]
如果能够提出一种自适应排斥型平滑策略：在构建连续PDF时，将已有频率位置作为排斥中心，对标准KDE结果施加局部抑制，在已有频率邻域“凹陷”。使该PDF可直接用于重要性采样，引导新频率点自适应落入先验频谱未充分表征的区域。
=== 双网络优化困难
在实验中发现，使用双网络优化一些较简单的简单问题时，误差可能比单独使用其中一种要大.

论文中验证$alpha$设为可参数的有效性时，没有交代$alpha_text("opt")$的更新频率或$alpha_text("learnable")$的学习率.实验中发现$alpha$更新较快时，训练效果不好.
=== 超参数多
除去PINN的超参数，加入随机傅里叶特征的过程又引入了先验频率的分布参数$sigma$,num_scale,频率衰减率$beta$,以及是否设置$beta$为可学习。引入Cross Attention后又引入了注意力头数，注意力维度等超参数。引入双网络训练后又要决定$u_l$和$u_h$分别的学习率，网络层数和宽度，分别的激活函数，以及$alpha$的更新频率或学习率。这些超参数大多在论文中没有明确，调参变得非常困难。
