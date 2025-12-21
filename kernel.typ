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
#set math.vec(delim: "[")
#show: university-theme.with(
  aspect-ratio: "16-9",
  // align: horizon,
  // config-common(handout: true),
  config-common(frozen-counters: (theorem-counter,)), // freeze theorem counter for animation
  config-info(
    title: [核方法],
    institution: [上海师范大学数理学院],
  ),
)

#title-slide()

在基本线性回归中，模型$h_theta (x)=theta x + theta_0$假设数据满足线性关系，但实际中的数据间的关系往往是非线性的。例如:
$
  h_theta (x)=theta_3 x^3 + theta_2 x^2 + theta_1 x + theta_0
$
,可以看出，$h_theta$ 对于$x$是非线性的，但是它对于参数$theta_i$是线性的。因此，我们可以通过对输入数据进行非线性变换，将其映射到一个高维特征空间中，在该空间中，数据可能满足线性关系，从而可以使用线性回归模型进行拟合。
#definition-box[
  定义函数$phi:bb(R) -> bb(R)^4$,形式为：$phi(x)=[1, x, x^2, x^3]^T$，有：
  $
    h_theta (x)=[theta_0, theta_1, theta_2, theta_3] vec(1,x,x^2,x^3)=theta^T phi(x)
  $
  这样，我们得到了新的数据集${(phi^(1),y^(1)), (phi^(2),y^(2)), ..., (phi^(n),y^(n))}$\
  $phi$ 
 被称为特征提取函数（feature function / feature extractor）\
 $phi(x)$被称为特征（features），\
 而$x$被称为属性（attribute）或原始特征（raw feature）。
]

在新数据集上做线性回归，并使用梯度下降(GD)进行优化，学习率为$alpha$，此时损失函数和参数
的更新过程分别为：
$
text("loss")=1/2 sum_(i=1)^n (y^((i)) - theta^T phi(x^((i))))^2\
theta:=theta + alpha sum_(i=1)^n (y^((i)) - theta^T phi(x^((i)))) phi(x^((i)))
$
,可以看出，计算损失函数和参数更新时，都涉及到$phi(x^((i)))$的计算。如果$phi$将数据映射到一个非常高维的空间中，计算$phi(x^((i)))$的开销以及做内积的开销会非常大，甚至不可行。
#theorem-box[
  若$theta^0=0$,那么$theta$恒能表示为features的线性组合，即$theta=sum_(i=1)^n beta_i phi(x^((i)))$，其中$beta_i in bb(R)$
] 
#proof[
  通过数学归纳法证明：\
  (1)当迭代次数$k=0$时，$theta^0=0$，命题成立。\
  (2)假设当迭代次数$k=m$时，命题成立，即存在$beta_i^(m)$使得$theta^m=sum_(i=1)^n beta_i^(m) phi(x^((i)))$。\
  (3)当迭代次数$k=m+1$时，有：
  $
    theta^(m+1)=theta^m + alpha sum_(i=1)^n (y^((i)) - (theta^m)^T phi(x^((i)))) phi(x^((i)))\
    =sum_(i=1)^n beta_i^(m) phi(x^((i))) + alpha sum_(i=1)^n (y^((i)) - (sum_(j=1)^n beta_j^(m) phi(x^((j))))^T phi(x^((i)))) phi(x^((i)))\
    =sum_(i=1)^n [beta_i^(m) + alpha (y^((i)) - sum_(j=1)^n beta_j^(m)phi(x^((j)))^T phi(x^((i))) )] phi(x^((i)))
  $
因此，命题对于$k=m+1$也成立。\
  综上所述，命题对所有非负整数$k$均成立。
]\
这样，我们就不必每次迭代都计算高维的$theta$，而是通过学习$beta_i$来间接表示$theta$。将$theta^m=sum_(i=1)^n beta_i^(m) phi(x^((i)))$代入参数更新公式，有：
$
  theta^(m+1)=sum_(i=1)^n beta_i^(m) phi(x^((i))) + alpha sum_(i=1)^n (y^((i)) - (sum_(j=1)^n beta_j^(m) phi(x^((j))))^T phi(x^((i)))) phi(x^((i)))\
  =sum_(i=1)^n underbrace([beta_i^(m) + alpha (y
^((i)) - sum_(j=1)^n beta_j^(m) phi(x^((j)))^T phi(x^((i))) )],beta_i^(m+1)) phi(x^((i)))
$
#v(-1em)
$
text("因此")
beta_i^(m+1) = beta_i^(m) + alpha (y^((i)) - sum_(j=1)^n beta_j^(m) k(x^((j)), x^((i))) )
$
然而，计算内积$phi(x^((j)))^T phi(x^((i)))$仍然是一个高维问题，我们注意到：
- $phi(x^((j)))^T phi(x^((i)))$在整个过程中只需计算一次，可以预先计算并存储下来；
- 计算高维特征 $phi(x)$ 和 $phi(y)$ 的内积，不需要显式构造高维特征，直接用原始低维向量 $x,y$ 的内积就能计算,例如:
$
  chevron.l phi(x), phi(y) chevron.r = [1,x, x^2, x^3] vec(1,y,y^2,y^3) \ 
  =1+ x y + x^2 y^2 + x^3 y^3 = chevron.l 1,1 chevron.r + chevron.l x,y chevron.r + chevron.l x, y chevron.r^2 + chevron.l x, y chevron.r^3
$
#theorem-box[
定义核函数（kernel function）$K(x,y)=chevron.l phi(x), phi(y) chevron.r$
]
在这个例子中，$K(x,y)=chevron.l 1,1 chevron.r + chevron.l x,y chevron.r + chevron.l x, y chevron.r^2 + chevron.l x, y chevron.r^3$,而原空间的内积 $chevron.l x,y chevron.r$ 只需要计算一次。当这里不是4维而100维时，这无疑大大降低了计算量。\
若$phi:bb(R)^d -> bb(R)^p$,数据量为$n$,则单次迭代$theta$的时间复杂度由$O(n p)$变为$O(n^2)$，当$n<<p$时，降幅显著。\
#pagebreak()
考虑核函数$
K(x,y)=sum_(i=0)^p chevron.l x,y chevron.r^k
$当$p$极大时，直接计算$K(x,y)$的开销仍然极大，但是如果令$
K(x,y)=sum_(i=0)^p 1/(k!) chevron.l x,y chevron.r^k
$,然后令$p -> +oo$,则有：$K(x,y)=e^(chevron.l x,y chevron.r)$ ,这样我们就不必显示地计算多次低维空间内积的幂次，而是通过计算一次指数函数来间接得到结果。并且数据的特征被映射到了无穷维空间。\
然而，这个核函数并不常用，一个显著的问题是当$chevron.l x,y chevron.r$较大时，$e^(chevron.l x,y chevron.r)$会变得非常大，导致数值不稳定。另外，这个核函数的几何意义并不如下文将引入的高斯核清晰。\
不知道$phi(dot)$的时候，我们怎么知道一个函数（例如$e^(chevron.l x,y chevron.r)$）是否是某个$phi(dot)$对应的核函数呢？还好Mercer 条件给出了判定方法：
#theorem-box[
  Mercer 条件：关于两个变量对称的函数$K(dot,dot)$是某个特征映射$phi(dot)$对应的核函数的充分必要条件是，对于任意有限集合${x^(1), x^(2), ..., x^(n)}$，由$K(x^(i), x^(j))$构成的矩阵$K in bb(R)^(n times n)$是半正定的。
]
#pagebreak()

#theorem-box[
  在分类问题中，核方法 是解决非线性问题的核心工具之一。它通过隐式地将原始低维数据映射到高维特征空间，让原本线性不可分的数据变得可分，进而用线性模型解决非线性问题。支持向量机（SVM）、高斯过程回归（GPR）等经典算法都依赖核方法实现强大的非线性拟合能力。
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
在SVM中，给定训练数据集${(x^(i), y^(i)) | i=1,2,...,n}$，其中$x^(i) in bb(R)^d$为输入特征，$y^(i) in {+1, -1}$为类别标签。SVM的目标是找到一个最优超平面，将不同类别的数据点分开。对于线性可分的数据，SVM通过求解以下优化问题来找到最优超平面：
#v(-1em)
$
cases(
min_alpha quad 1/2 sum_(i = 1)^n sum_(j = 1)^n alpha_i alpha_j y_i y_j bold(x)_i^T bold(x)_j - sum_(i = 1)^n alpha_i ,  sum_(i = 1)^n alpha_i y_i = 0 ,
 alpha_i >= 0 quad (1<=i<=n)
)
$
这里的$bold(x)_i^T bold(x)_j $就是输入特征的内积。当数据线性不可分时，我们可以引入核函数$K(bold(x), bold(y))$，将输入特征映射到高维空间，从而使得数据在高维空间中线性可分。\
此时，优化问题变为：
$
cases(
min_alpha quad 1/2 sum_(i = 1)^n sum_(j = 1)^n alpha_i alpha_j y_i y_j K(bold(x)_i, bold(x)_j) - sum_(i = 1)^n alpha_i ,  sum_(i = 1)^n alpha_i y_i = 0 ,
 alpha_i >= 0 quad (1<=i<=n)
)
$
我们希望核函数有以下性质：
- 隐式映射：核函数应能隐式地将数据映射到高维空间
- 数值稳定：如核函数的值域为[0,1]
- 位移不变:也就是说$K(bold(x), bold(y)) = K(bold(x) - bold(y))$，核函数的输出只与两个数据点的距离有关。这是因为在分类问题中，我们倾向于将距离较近的数据点归为同一类，距离较远的数据点归为不同类。而和这两点是否靠近原点无关。
#pagebreak()
其中最常用的核函数之一便是高斯核：
#v(-1em)
$
  K(bold(x), bold(y)) = exp(-(||bold(x) - bold(y)||^2)/(2 sigma^2))
$
#v(-1em)
它将两个数据点之间的几何距离，巧妙地转化为了一个介于 0 和 1 之间的“相似度”分数：
- 分数接近 1 意味着两个点在空间上离得非常近，它们“高度相似”。
- 分数接近 0 ：意味着两个点在空间上离得非常远，它们“毫不相关”
从几何角度看，高斯核还可以表示为每个数据点在周围产生了一个能量场，距离越近，影响力越大；距离越远，影响力越小。通过调整参数$sigma$，我们可以控制这种影响力的扩散程度，从而灵活地适应不同的数据分布和分类需求。
#pagebreak()
可视化高斯核的能量场如下所示：
#place(
  bottom+left,
  image("assets/Figure_1.png",
    height:auto,),
);
#pagebreak()
=== 核方法的致命缺陷：大数据下的计算爆炸
当数据集规模较大时(如样本数$N=10000$)，使用核方法需要计算核矩阵（Kernel Matrix）$K in bb(R)^(N times N)$,此时会面临两个严重问题:
- 时间复杂度：计算核矩阵需要$O(N^2)$次核函数运算,计算量巨大；
- 空间复杂度：储存核矩阵需要$O(N^2)$的内存,对于大规模数据集,内存需求难以满足。
因此，核方法在非线性核函数 + 大数据集的场景下几乎不可用。而随机傅里叶特征（Random Fourier Features, RFF）正是为解决这个问题而生 —— 它通过显式的低维映射，用 $O(N dot D)$ 的复杂度（$D$ 为映射后维度，通常 $D lt.double N$)近似核函数，让大数据下的非线性建模成为可能。
== RFF 的核心思想
#theorem-box[
  RFF 的核心目标是：找到一个#text(fill: blue, "显式")的低维映射$Z(dot):bb(R)^d -> bb(R)^D$,使得：$k(bold(x),bold(y)) approx Z(bold(x))^T dot Z(bold(y))$.
]
这样一来，我们可以先通过 $Z(dot)$ 将原始数据映射到$D$维特征空间,然后利用线性模型替代非线性核方法，可以大幅降低计算复杂度。
#caution-box[RFF 仅适用于移位不变核。]

#theorem-box[
RFF 核心是将核函数表示为傅里叶积分，再通过随机采样近似该积分，最终得到低维映射。RFF极大降低计算量的能力，显示出了概率和近似思想的强大威力。
]
