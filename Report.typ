#import "@preview/polylux:0.4.0": *
#import "@preview/touying:0.6.1"
#import "touying-template/theme.typ": *
#import "ori.typ": *
#import "@preview/cetz:0.4.2"
#import "@preview/wrap-it:0.1.1": *

#show: mytheme.with(
  aspect-ratio: "16-9",
  font-en: "Noto Sans",
  font-ja: "BIZ UDPGothic",
  font-math: "Noto Sans Math",
  config-info(
    title: [],
    subtitle: [Subtitle],
    author: [Author],
    institution: [Institution],
    header: [Conference\@Location (Date)]
  )
)

== 交叉注意力的计算原理@feng2025overcomingspectralbiascrossattention 
假设我们已有一个简化的傅里叶特征库:
$
  H(x) in bb(R)^(n times d) quad text("where") quad n=M/d
$
$
  H_(i,j)(x)=sin((i-1)d+j)x, quad i=1,2,...,n,quad j=1,2,...,d
$
使用坐标 $x$ 生成维度适合的 $Q_0(x)$ 后,Cross-Attention的计算公式为:
$
  text("CA")(Q_0(x),H(x))
  =text("softmax")((Q_0 W_q W_k^top H^top)/sqrt(d)) H W_v
$
把$text("softmax")$记为$s=(s_1,s_2,...,s_n)) in bb(R)^n$,而$W_v$可以看成下一层的权重矩阵从而省略，从而得到:
$
  text("CA")(Q_0(x),H(x))
  =(s_1 ,s_2, dots, s_n) dot H
$
#align(center)[
#cetz.canvas(
length: 2cm,  // 单位长度
{
let arrow-style = (mark: (end: "stealth", scale: 2.0), fill: black)
import cetz.draw: *

// 在指定位置显示内容
content((-2,3), [$s_1$])
content((-2,2), [$s_2$])
content((-2,1.4), [$dots.v$])
content((-2,1), [$dots.v$])
content((-2,0.6), [$dots.v$])
content((-2,0), [$s_n$])



content((1.5,1.4), [$dots.v$])
content((1.5,1), [$dots.v$])
content((1.5,0.6), [$dots.v$])

content((-1.5,3), [$times$])
content((-1.5,2), [$times$])
content((-1.5,0), [$times$])

content((1.5,3.0), [$[sin(x),sin(2x),dots,sin(d x)]$])
content((1.5,2.0), [$[sin((d + 1)x),dots,sin(2d x)]$])
content((1.5,0.0), [$[sin((n-1)d+1)x),dots,sin(n d x)]$])

line((4,3), (6,1.6), ..arrow-style)
line((4,2), (6,1.5), ..arrow-style)
line((4.5,0), (6,1.4), ..arrow-style)

content((6.5,1.5), [$text("CA")(x)$])
})]

但这也隐含了同一token内特征同样重要的假设($sin(x)$到$sin(d x)$的系数都是$s_1$).

并且输出时将不同尺度混在了一起，例如$sin(x)$输出时不和$sin(2x)$一组，反而和$sin((d+1)x)$一组，这打乱了输出结构.

我们期待提出一种方法，赋予每个token内部不同特征不同的权重，并且输出具有可解释的结构信息.

== KAFF
我们提出一种名为KAFF(Kernelized Attention for Fourier Features)的方法，大致思路如下：
- 将物理空间的多尺度RFF按「逐个分量频率递增，$d$整除最后一个分量的频率数」规则划分为token，每个token对应一段连续物理频带
- 每个token用输入自适应的一维高斯核生成token内单个频率的细粒度权重，token内聚合得到对应频带的输出
- 解决RFF-CA的两个问题：token内权重共享、输出跨尺度混合

具体如下:

$bold(x) in bb(R)^(d_text("in")
),quad text("FFN")_1(bold(x))=Q in bb(R)^(h), quad W_(mu)in bb(R)^(n times h),\ b_(mu) in bb(R)^(h),quad W_sigma in bb(R)^(n times h),quad b_sigma in bb(R)^(h),quad Omega in bb(R)^(n times d) $

下面为了简化，假设 $d_text("in") = 1$,从而频率 $omega$ 都是一维的.

$Omega=mat(omega_1, dots, omega_(d); dots.v,dots,dots.v; dots, dots , omega_(n d))$,逐行按照频率从低到高排列.

$Phi$和$Omega$同形状,$Phi_(i,j)=(2pi)/((i-1)d+j)$.
$ H[i,j]=cos(Omega[i,j]^T x+Phi[i,j]),
H in bb(R)^(n times d). $

- $ G = Omega $
然后对$G$按行(在每个token内)归一化到[-1, 1]，我们使用$G$来模拟同一token内频率间的距离.

- $ tanh(W_mu Q + b_mu) = mu $
- $ W_sigma Q + b_sigma = sigma $
#pagebreak()
以高斯核为例：
- $ S[i,j] = exp(-(G[i, j]-mu [i])^2 /(2 sigma[i]^2)) $
- $A = text("sum")(S dot.o H,dim=1) in bb(R)^n$ （0代表行，1代表列）
- $ text("Output") = text("FFN")_2(A)$
#v(-1em)
#align(center)[
  #image("assets/Figure_1.png")
]
#pagebreak()
下面考虑高维的情况：

取傅里叶特征频率为网格点(未必是均匀网格)，\
${bold(omega)_i = (omega_1, dots, omega_d_text("in"))|i=1,dots,n_1 times n_2 times ... times n_d}$,取$d$整除$n_d$，在划分token时使每个token内的频率除了最后一维的频率不同以外，其他维度的频率都相同，得到:
$Omega=mat(bold(omega)_1, dots, bold(omega)_(d); dots.v,dots,dots.v; dots, dots , bold(omega)_(n d)) in bb(R)^(n times d)$.

将上文的$G = Omega $修改为$G =mat(bold(omega)_1[-1], dots ,bold(omega)_(d)[-1]; dots.v,dots,dots.v;dots,dots,bold(omega)_(n d)[-1]) $,同一token内前面的频率相同,各个特征间的距离只和最后一个维度有关.

得到注意力输出结果后，再通过一个网络得到最终输出$text("FFN")_2(A)$.
#pagebreak()
使用残差神经网络拟合如下二维函数，除去输出和输入的线性变换，共使用4个残差块.分别设置4个残差块中有$1，2，3，4$个交叉注意力块，其余为普通残差块，普通残差块放置在注意力残差块的后面.

每个残差块的神经网络结构为：

[输入 →(pre_norm → CA → 残差连接)→pre_norm → 线性1(32→64) → SiLU激活 → 线性2(64→32) → 残差连接 → 输出
]

#wrap-content(
image("assets/output.png",height: 160pt),
[使用Adam优化器，学习率为1e-3，取65536个训练数据，单个epoch使用的mini batch size为8192，dot_ca方法内使用4头注意力,使用均方误差损失函数.

使用核化注意力的方法要比点积注意力的方法训练更慢，注意力层越多训练越慢.
]
)
#pagebreak()
#align(center)[

#image("assets/l2_error_curve.png",height: 200pt)

#three-line-table[
|注意力块数|dot_ca 最终误差|kernel_ca 最终误差|
|--|--|--|
|1|0.465|0.357|
|2|0.391|0.302|
|3|0.317|0.333|
|4|0.289|0.390|
]]
为什么注意力块数的增加没能带来KAFF误差的持续下降？
#pagebreak()
== 交叉注意力的缺点
- 1 计算复杂度，显存占用极高.
注意力方法本身计算复杂度就较高，在PINNs中还需要做高阶自动微分，傅里叶特征库$H$的每个分量都是由 $x$ 带入非线性函数 $cos$ 生成，不得不保持复杂的计算图.

- 2 训练较不稳定

KAFF方法引入了较强的非线性，这和通常的网络设计中追求可训练参数线性影响输出的原则@zhang2025fouriermulticomponentmultilayerneural 相悖，可能导致损失景观过度复杂.

#align(center)[
#image("assets/MMNNeg.pdf",height: 150pt)
]


#columns(2)[

#image("assets/LandscapeFCNN1Act1.pdf",height: 200pt)

#colbreak()
#image("assets/LandscapeMMNN1Act1.pdf",height: 200pt)
]
#pagebreak()
== TINNs@dai2026tinnstimeinducedneuralnetworks
标准的时空 PINNs 将时间作为输入，但在所有时刻复用权重共享的单一网络，这就迫使同一组特征去表征差异显著的动力学行为。这种强耦合会降低精度，且在同时施加偏微分方程约束、边界约束与初始约束时，可能导致训练不稳定.

@dai2026tinnstimeinducedneuralnetworks 文提出了TINNs架构，它将网络权重参数化为时间的学习函数，使得有效的空间表征能够随时间演化，同时保留共享的结构。该文在多种含时偏微分方程上的实验表明：与 PINN 及其他强基线方法相比，TINN 的精度最高提升 4 倍，收敛速度最高加快 10 倍.


#align(center)[
  #image("assets/structure_comparison.pdf",height: 160pt)

]
#pagebreak()
TINNs 的核心思想，是把PINNs的网络$u_theta (bold(x),t)$变为$u_theta(t) (bold(x))$，使得时间不进入网络输入，只进入参数空间.

考虑L-layer TINNs网络架构:\
$ u_theta (bold(x)) = bf(W)_L sigma(bf(W)_(L-1)sigma(... sigma(bf(W)_1 bold(x) +bf(b)_1))+bf(b)_(L-1)) + bf(b)_L$

其中$bold(theta)(t) = {(bf(W)_cal(l),bf(b)_cal(l))}_(cal(l)=1)^L $.
$bf(W)_cal(l) in bb(R)^(l_(cal(l)) times l_(cal(l-1))),bf(b)_cal(l) in bb(R)^(l_(cal(l)))$,依赖 $t$ , $l_0$为输入层(空间)维度.总参数量为:

$ N_D = sum_(cal(l)=1)^L (l_(cal(l)) l_(cal(l-1)) + l_(cal(l))) $


我们需要用 $t$ 生成这$N_D$个参数.
#pagebreak()
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
#pagebreak()
为了增强表达能力，论文中提出的方法是使用一个小神经网络输出
$ bold(cal(N)) in bb(R)^(2L)$,然后使用一个门控机制得到：
$ bold(Phi) (t) = (bold(1-alpha))t +bold(alpha dot.o)  bold(cal(N))(t) in bb(R)^(2L) $
$bold(alpha) in bb(R)^(2L) $,为可学习参数,
$bold(Phi)(t) = (Phi_1 (t),Phi_2 (t),...,Phi_(2L) (t))^top$.


$ bf(W)_i = Phi(t)_(2i-1) dot bf(A)_(2i-1)+ bf(B)_(2i-1)\
bf(b)_i = Phi(t)_(2i) dot bf(A)_(2i)+ bf(B)_(2i)
$



#pagebreak()

能否推广TINNs 的方法， 在隐藏层嵌入空间坐标信息$(x_1, x_2, ..., x_d)$?

主要方向如下：

- 1 每层都嵌入所有物理坐标$(t, x_1, x_2, ..., x_d)$，
与HyperPINN@belbuteperes2021hyperpinnlearningparameterizeddifferential 使用方程参数（如粘性系数）作为超网络的输入不同，用$(t,x_1, x_2, ..., x_d)$作为超网络的输入，让超网络生成 $bold(Phi) (t)$，然后把$Phi_i(t)$经过仿射变换得到一个参数矩阵.

- 2 分层嵌入坐标信息，例如在第$i$层输入$x_i$，物理意义更明确.

预计该方法能够在具有复杂空间结构的偏微分方程上取得更好的效果.
#pagebreak()
#bibliography("bib/IEEE Xplore Citation BibTeX Download 2026.3.8.20.9.42.bib")
