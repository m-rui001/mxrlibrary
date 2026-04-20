#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.1"
#import "@preview/fletcher:0.5.8" as fletcher: edge, node
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.4.1": *
#import cosmos.clouds: *
#import "ori.typ": *
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
    title: [PIPNNs: Physics-Informed Parameterized Neural Networks],
  ),
)
#set par(first-line-indent: (amount: 2em, all: true))
#title-slide()

== 相关工作
标准的时空 PINNs 将时间作为输入，但在所有时刻复用权重共享的单一网络，这就迫使同一组特征去表征差异显著的动力学行为。这种强耦合会降低精度，且在同时施加偏微分方程约束、边界约束与初始约束时，可能导致训练不稳定.

@dai2026tinnstimeinducedneuralnetworks 文提出了TINNs架构，它将网络权重参数化为时间的学习函数，使得有效的空间表征能够随时间演化，同时保留共享的结构。该文在多种含时偏微分方程上的实验表明：与 PINN 及其他强基线方法相比，TINN 的精度最高提升 4 倍，收敛速度最高加快 10 倍.

TINNs 的核心思想，是把PINNs的网络$u_theta (bold(x),t)$变为$u_theta(t) (bold(x))$，使得时间不进入网络输入，只进入参数空间.

考虑L-layer TINNs网络架构,

$ u_theta (bold(x)) = bf(W)_L sigma(bf(W)_(L-1)sigma(... sigma(bf(W)_1 bold(x) +bf(b)_1))+bf(b)_(L-1)) + bf(b)_L$

其中$bold(theta)(t) = {(bf(W)_cal(l),bf(b)_cal(l))}_(cal(l)=1)^L $.
$bf(W)_cal(l) in bb(R)^(l_(cal(l)) times l_(cal(l-1))),bf(b)_cal(l) in bb(R)^(l_(cal(l)))$,依赖 $t$ , $l_0$为输入层(空间)维度.总参数量为:

$ N_D = sum_(cal(l)=1)^L (l_(cal(l)) l_(cal(l-1)) + l_(cal(l))) $

一个自然的想法是使用MLP直接生成 $bold(theta)(t) in bb(R)^(N_D)$，但这样的方法会引入过多的参数.

另一个轻量的方法是令
$bold(theta)(t)=bold(omega)t+bold(b) $.

此处是一次性生成所有层的参数，优点是能充分利用gpu的并行计算能力，缺点是优化器的自适应学习率,weight decay无法针对每层参数进行调整.

如果设置第$i$层有四个独立的参数
$bf(A)_(2i-1),bf(A)_(2i),bf(B)_(2i-1),bf(B)_(2i)$,
分层生成权重如下:
$ bf(W)_i = t dot bf(A)_(2i-1)+ bf(B)_(2i-1)\
bf(b)_i = t dot bf(A)_(2i)+ bf(B)_(2i)
$
为了增强表达能力，论文中提出的方法是使用一个小神经网络输出
$ bold(cal(N)) in bb(R)^(2L)$,然后使用一个门控机制得到：
$ bold(Phi) (t) = (bold(1-alpha))t +bold(alpha dot.o)  bold(cal(N))(t) in bb(R)^(2L) $
$bold(alpha) in bb(R)^(2L) $,为可学习参数,
$bold(Phi)(t) = (Phi_1 (t),Phi_2 (t),...,Phi_(2L) (t))^top$.


$ bf(W)_i = Phi(t)_(2i-1) dot bf(A)_(2i-1)+ bf(B)_(2i-1)\
bf(b)_i = Phi(t)_(2i) dot bf(A)_(2i)+ bf(B)_(2i)
$

此外，我们预计残差形式
$ bold(Phi) (t) = t + bold(cal(N))(t) $
也能达到效果.


== 启发性实验

对于一维无粘性Burger方程，比较传统PINN和TINN的性能：
求解域: $x in [0, 1], t in [0, 0.3]$

初始条件: $u(x, 0) = sin(2pi x)$

边界条件: $u(0, t) = u(1, t) = 0$

PINN架构: 
输入层: 输入$vec(x,t)$,然后经过$W_text("in") in bb(R)^(2 times h)$.

使用2个残差块,每个残差块内部计算公式为：
$ y = x + sigma(W_1(sigma(W_0(x)+b_0))+b_1) $
其中 $W_0 in bb(R)^(h times 2h), W_1 in bb(R)^(2h times h),sigma = text("SiLU")$.

t-parametered PINN中, $W_0$和$W_1$被设计为时间依赖的权重，即$W_0(t) = W_0^0 + t W_0^1$和$W_1(t) = W_1^0 + t W_1^1$，并且只输入$x$.

由于t-parametered PINN 有约两倍的参数量，因此我们令t-parametered PINN的$h=24$，PINN的$h=34$,因为$34 approx sqrt(2)times 24$.

训练:

由于t-parametered PINN需要根据t决定权重，我们使用累积梯度方法.每个epoch内，使用Latin Hypercube Sampling 在时间域采样4个随机时间点.每个时间点采样128个随机空间点，再加上$4 times 2$个边界条件点，初始$t=0$采样256个点计算initial Loss，8个边界条件点参与计算boundary Loss.

普通PINN则直接在整个空间时间域内采样$4 times 128$个随机点计算pde Loss,再加上$128$个边界条件点计算boundary Loss,以及$256$个初始条件点计算initial Loss.

Loss等于三项平方损失的加权和.

训练1000个epoch.

使用Lax-Oleinik 公式计算Burger方程的解析解,每100个epoch,取时间点分别为[0.1,0.2,0.3]，在每个时间点Latin随机采样256个空间点计算相对L2误差,然后取平均，使用一个列表记录下来.

最后在一张图上，使用对数坐标，绘制出两种方法的相对L2误差随训练epoch的变化曲线.

训练完成后在[0,0.1,0.2,0.3]四个时间点上，使用2*2子图，绘制出两种方法的数值解和解析解的对比图.

分别记录两种方法训练所用的时间.

#bibliography("bib/IEEE Xplore Citation BibTeX Download 2026.3.8.20.9.42.bib")
