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
    title: [Overcoming Spectral Bias via Cross-Attention实验报告],

  ),
)
#set par(first-line-indent: (amount: 2em, all: true))
#title-slide()
== 1D Poisson方程求解的失败
对于1D Poisson方程的例子:
$
  - Delta u(x) = f(x) , quad x in [-1,1]
$
满足Dirichlet边界条件,真实解 $u(x)$ 为:
$
   sin(0.1 pi x) + 0.2 sin(pi x) + 0.4 sin(100/3 pi x) +0.6 sin(200/3 pi x) +sin(100 pi x)
$
这个例子的求解难点在于,$f(x)$中,频率为$100pi$的振幅为$(100 pi)^2$,而频率为$0.1 pi$的振幅为$(0.1 pi)^2$,相差了$10^6$数量级,导致训练过程中,高频部分的误差主导了损失函数,使得低频部分的误差无法得到有效优化,最终导致模型难以收敛到真实解。

使用NN-CA单网络，傅里叶特征频率满足$omega ~ cal(N)(0,sigma^(-2))$, $sigma=0.006$总特征数为4096，单个token包含的特征维数$d_q=256$,隐藏层层数6，使用Adam优化器训练5000 epoch,训练结果见右下图.
#align(center)[
  #columns(2)[
#figure(image("assets/image-1.png",height:250pt))
#image("assets/image.png",height:250pt,width: 400pt,fit:"stretch")]
]

可以看出，模型较好的捕捉了高频部分的特征，但在低频部分表现较差，难以收敛到真实解。

论文使用了一种双网络方法,在训练过程中,一个网络$u_l$专注于捕捉低频特征,另一个网络$u_h$专注于捕捉高频特征,最终预测为$u_h+ alpha u_l$. 文中比较了$alpha$恒定0或1,可学习参数,和Loss函数对$alpha$求导为0的极小值点$alpha_text("opt")$.其中$alpha_text("opt")$是每次迭代中用一些样本点计算得到的近似值，论文比较了四种方法得到的结果,发现可学习参数和$alpha_text("opt")$方法表现较好,而恒定0或1的方法表现较差。

然而在复现中，$u_l$的输出总是在很少几次训练后就恒为0，导致双网络架构失去意义。经过多次实验，观察到$u_l$收敛于0的倾向在多组超参数下都存在,包括不同的学习率、不同的频率分布参数$sigma$,不同的网络结构等。
#wrap-content(image("assets/image-4.png",height:259pt,width: 300pt,fit:"stretch"),[经过分析，这可能是由于训练过程中的边界条件在损失函数中加强太大导致的。论文中损失函数$L=L_f + lambda L_b$，其中无论是$L_f$还是$L_b$,都混合了$u_h$和$u_l$，由于$L_f$的量级很大，论文中设置了$lambda$为1e3或1e4,原文中让$u_h$在边界处和满足原方程的Dirichlet条件，$u_l$在边界满足输出为0的Dirichlet条件，这本身就让$u_l$倾向于输出0.而我们本来期待$u_l$学习的低频部分，如$sin(0.1pi x)$,在二阶导后的量级为$(0.1pi)^2$,远小于高频部分的量级$(100pi)^2$,因此模型倾向于忽略拟合低频部分，让$u_l$的输出恒为0.])

为了解决这个问题，有以下方法可以尝试:
- 解耦$L_b$中$u_h$和$u_l$的部分,让两部分有不同的权重,例如$L_b= lambda_h L_b(u_h) + lambda_l L_b(u_l)$,并且设置$lambda_l$远小于$lambda_h$,以减轻边界条件对$u_l$的影响。
- 不再让$u_l$在边界满足输出为0的Dirichlet条件,而是让$u_h + alpha u_l$整体满足原始边界条件,这样可以让$u_l$有更多的自由度来学习低频部分。

由于第二种方法超参数少而且似乎更加自然,我们首先尝试了第二种方法,并且在这种情景下重新推导了$alpha_text("opt")$的计算公式.结果证明了这个方法能解决$u_l$输出恒为0的问题.
#align(center)[
  #columns(2)[
#image("assets/output2.png")
#image("assets/output5.png")]
左图：令$alpha$为$alpha_text("opt")$,  右图:令$alpha$为可学习参数.]

然而该方法仍未解决模型在低频难以收敛到真实解的问题.另外，当使用原有公式计算$alpha_text("opt")$时，$alpha$的值很快就趋近于0且无法回升，双网络没有效果；当使用新边界条件推导的公式计算$alpha_text("opt")$时，$alpha$的值虽然不再趋近于0，但训练表现不如设置$alpha$为可学习参数的方法，这点和论文中$alpha_text("opt")$方法表现最好不一致。
#pagebreak()
#align(top)[
#image("assets/output3.png",height:200pt,width: 700pt,fit:"stretch")
]
#align(center)[$alpha=alpha_text("opt")$时的高低频分量]
根据上图我们可以推测，在这种情况下，$alpha u_b$的作用可能也只是和$u_h$相加满足边界条件，并没有真正起到分离高低频特征的作用，因此模型仍然难以收敛到真实解。

和论文中平稳下降的Error曲线不同，我们的复现中Error曲线初期下降后几乎就不再下降了，这也说明模型在低频部分的拟合没有得到有效优化，无法进一步收敛到真实解。

后面两张图分别为论文和复现的结果。
#image("assets/image-6.png",height:400pt,width: 700pt,fit:"stretch")
#image("assets/image-5.png",height:400pt,width: 700pt,fit:"stretch")

== 使用极简RFF神经网络求解
观察到上面的例子中的解是由简单正弦波叠加而成的，于是使用一个没有任何隐藏层的极简RFF神经网络来拟合这个解,即
$
  u_text("pred")(x) = sum_(i=1)^N a_i sin(omega_i x + phi_i) + text("bias")
$
$a_i$是模型权重(weights),设置$omega,phi$均为可学习参数，总特征维数为512,每次训练使用64个配点，采用PINN的损失函数，用Adam优化器训练1000 epoch后训练结果见下图.
#v(-1em)
#image("assets/image-8.png",height:120pt,width: 500pt,fit:"stretch")
#image("assets/image-7.png")
尽管这个模型非常简单,但它却非常快地在精度上超越了之前的复杂模型.然而，当我们尝试使用4层,宽64,激活函数为ReLU的MLP神经网络拟合该网络输出解和真实解的pde残差,使用Adam优化器训练1000 epoch后,训练结果见下图.
#align(center)[#image("assets/image-10.png",height:230pt,width: 500pt,fit:"stretch")]
#align(center)[#image("assets/image-11.png",height:230pt,width: 500pt,fit:"stretch")]
甚至彻底放弃边界条件，仍然无法拟合低频部分。

总结，该问题无法有效解决的原因是低频部分二阶导后的振幅过小,单靠物理约束的损失函数无法有效优化低频部分的拟合.

== 使用差分法代替自动微分
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