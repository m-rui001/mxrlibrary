#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.1"
#import "@preview/fletcher:0.5.8" as fletcher: edge, node
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.4.1": *
#import cosmos.clouds: *
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
    title: [随机傅里叶特征（Random Fourier Features）原理],
    institution: [上海师范大学数理学院],
  ),
)


#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))

= 为什么需要RFF?

== 核技巧

#theorem-box[
  在机器学习中，核技巧（Kernel Trick） 是解决非线性问题的核心工具之一。它通过隐式地将原始低维数据映射到高维特征空间，让原本线性不可分的数据变得可分，进而用线性模型解决非线性问题。支持向量机（SVM）、高斯过程回归（GPR）等经典算法都依赖核技巧实现强大的非线性拟合能力。
]
#place(
  bottom+left,
  image("assets/before_kernel.webp",
    height:auto,
    width:10em,),
);

#place(
  bottom+right,
  image("assets/kernel.webp",
    height:auto,
    width:15em,),
);

#pagebreak()
=== 核技巧的核心公式
#definition-box[
核函数的本质是高维特征空间中内积的 “替身”，其定义为：
$
  k(bold(x),bold(y)) =  chevron.l phi(bold(x)) , phi(bold(y)) chevron.r
$
]
#v(-1.2em)
#corollary-box[
- $k(bold(x),bold(y))$ 是核函数，表示输入数据;点 $bold(x)$ 和 $bold(y)$ 在高维特征空间中的相似度;
- $phi(dot):bb(R)^d -> bb(R)^infinity$ 是#text(fill: blue, "隐式")特征映射（通常映射到无穷维空间）;
- 其中，$chevron.l dot chevron.r$ 表示高维空间中的内积操作。
]
#pagebreak()
=== 核技巧的致命缺陷：大数据下的计算爆炸
当数据集规模较大时(如样本数$N=10000$)，使用核技巧需要计算核矩阵（Kernel Matrix）$K in bb(R)^(N times N)$,此时会面临两个严重问题:
- 时间复杂度：计算核矩阵需要$O(N^2)$次核函数运算,计算量巨大；
- 空间复杂度：储存核矩阵需要$O(N^2)$的内存,对于大规模数据集,内存需求难以满足。
因此，核技巧在非线性核函数 + 大数据集的场景下几乎不可用。而随机傅里叶特征（Random Fourier Features, RFF）正是为解决这个问题而生 —— 它通过显式的低维映射，用 $O(N dot D)$ 的复杂度（$D$ 为映射后维度，通常 $D lt.double N$)近似核函数，让大数据下的非线性建模成为可能。
== RFF 的核心思想
#theorem-box[
  RFF 的核心目标是：找到一个#text(fill: blue, "显式")的低维映射$Z(dot):bb(R)^d -> bb(R)^D$,使得：$k(bold(x),bold(y)) approx Z(bold(x))^T dot Z(bold(y))$.
]
这样一来，我们可以先通过 $Z(dot)$ 将原始数据映射到$D$维特征空间,然后利用线性模型替代非线性核方法，可以大幅降低计算复杂度。
#caution-box[RFF 仅适用于移位不变核（Shift-Invariant Kernel），这类核函数的输出只与两个数据点的差值有关，形式为：$k(bold(x),bold(y)) = k(bold(x) - bold(y))$.常见的移位不变核包括径向基函数（RBF）核、多项式核等。]


== RFF 的数学推导
#theorem-box[
RFF 的推导基于傅里叶变换和波赫纳定理（Bochner's Theorem），核心是将核函数表示为傅里叶积分，再通过随机采样近似该积分，最终得到低维映射。]
=== 第一步：核函数的傅里叶表示（波赫纳定理）
#corollary-box[
  好吧，到这里就看不懂了。
]
