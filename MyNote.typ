#import "ori.typ":*
#import "@preview/cetz:0.4.2": canvas, draw
#set heading(numbering:numbly("{1:一}、",default:"1.1  "))
#show: ori.with(
  title: "坟头草的数学笔记",
  author: "作者：坟头草",
  maketitle: true,
  makeoutline: true,
  first-line-indent: auto,
  outline-depth: 3,
)
#set page(header:none,margin:4em,)
#set-inherited-levels(3)
#v(20em)
#align(center)[
  #set text(size: 24pt, weight: "bold")
  面对我们的骨灰，高尚的人们将洒下热泪。
]
#align(right)[  #set text(size: 22pt)
  —— 卡尔·马克思  《青年选择职业的考虑》]
#pagebreak()
= 数学分析
== 渐近分析
=== Laplace方法
#idea[
Laplace方法基本思想：估计$I(x)=integral_a^b f(t) e^(phi.alt(t) x) dif t$时，如果$phi.alt$取得最大值$phi.alt(c),f(c)!=0$,那么$c$的邻域这部分积分贡献了$I(x)$在$x	-> +oo$时的主部.

基本方法：取$(c-delta,c+delta)$,先利用单调性、有界性等性质估计出积分$I_1(x)=integral_(c-delta)^(c+delta) f(t) e^(phi.alt x) dif t$的渐近表达式，然后估计$integral_a^(c-delta)+integral_(c+delta)^b$,结果往往为0，再对$integral_(c-delta)^(c+delta)$进行估阶（如Taylor展开，分部积分等）.
]
#example[
1、设$x_n=sum_(k=1)^n e^(k^2)/k,y_n=integral_0^n e^(x^2) dif x$，问
$
  lim_(n -> +oo) x_n/y_n=?
$

]
解：使用Stolz定理，有,
$
  (x_n-x_(n-1))/(y_n-y_(n-1))=e^(n^2)/(n integral_0^1 e^((x+n-1)^2)dif x)=1/(n integral_0^1 e^((x-1)^2) e^(2n(x-1)) dif x)
$
注意到当$x lt 1$时,令$n->+oo,e^(2n(x-1))	-> 0$，最大值点在$x=1$，因此使用Laplace方法估计分母部分，
$
  abs(n integral_0^(1-delta) e^((x-1)^2) e^(2n(x-1)) dif x) <= n integral_0^(1-delta) e dot e^(-2n delta) dif x = n(1-delta)e dot e^(-2n delta)	-> 0 quad (n -> +oo)
$
再考虑积分主部，直接根据单调性放缩成可积的形式
$
  n integral_(1-delta)^1 e^((x-1)^2) e^(2n(x-1)) dif x <= n integral_(1-delta)^1 e^(delta^2)e^(2n(x-1)) dif x=e^(delta^2)/2 e^(2n(x-1))|_(x=1-delta)^(x=1)=e^(delta^2)/2(1-e^(-2n delta))	-> e^(delta^2)/2
$
$
  n integral_(1-delta)^1 e^((x-1)^2) e^(2n(x-1)) dif x >= n integral_(1-delta)^1e^(2n(x-1)) dif x=1/(1-e^(-2n delta))	-> 1/2
$
因此，
$
  1/2 <= lim_(n -> +oo) n integral_(1-delta)^1 e^((x-1)^2) e^(2n(x-1)) dif x <= e^(delta^2)/2
$
令$delta -> 0$，可得
$
  lim_(n -> +oo) n integral_(1-delta)^1 e^((x-1)^2) e^(2n(x-1)) dif x=1/2
$
综上所述，
$
  lim_(n -> +oo) x_n/y_n=2
$
#definition[

    2、计算 $
           lim_(x -> +oo) x integral_0^10 (1+t)^(-1) e^(-x t) dif t
         $
  
]
解：最大值在$t=0$，取$delta$,
$
  x integral_delta^10 e^(-x t)/(1+t) dif t <= x integral_delta^10 e^(-x t)/(1+ delta) dif t= (e^(-delta x)-e^(-10 x))/(1+ delta) -> 0 quad (x -> +oo)
$
然后用分部积分估计主要部分:
$
  x integral_0^delta e^(-x t)/(1+t) dif t =-e^(- x t)/(1+t) | _0^delta - integral_0^delta e^(-x t)/(1+t)^2 dif t
  = (1- e^(-x delta)/(1+ delta)) - integral_0^delta e^(-x t)/(1+t)^2 dif t
\
  abs(integral_0^delta e^(-x t)/(1+t)^2 dif t)<=integral_0^delta e^(-x t) dif t = (1-e^(-x delta))/x ->0 quad (x->+oo)
\
  x integral_0^delta e^(-x t)/(1+t) dif t->1 quad (x->+oo)
$
从而结果为1.
#example[
    3、计算
    $
      lim_(x -> +oo) x integral _0 ^(pi/2) e^(-x tan t) dif t
    $
]

解：$t=0$为最大值点,先估计积分为0的部分：
$
  x integral _delta ^(pi/2) e^(-x tan t) dif t<= (pi/2 - delta) x e^(-x tan delta) ->0 quad (x -> +oo)
$
再研究最大值附近情况，尝试分部积分：
$
  x integral_0^delta e^(-x tan t) dif t=x t e^(-x tan t)|^(t=delta)_0 + x^2 integral_0^delta t e^(-x tan t) / (cos^2 t) dif t
$
看起来我们的做法增加了复杂度，于是回到一开始的式子，做Taylor展开：
$
  x integral_0^delta e^(-x tan t) dif t &= x integral _0 ^delta e^(-x (t+O(t^3))) dif t\
  &=x integral_0^delta e^(-x t) e^(O(t^3)) dif t \
  &=x integral_0^delta e^(-x t) (1+O(t^3)) dif t\
$
存在C>0,使得：
$
  abs(integral_0^delta e^(-x t) O(t^3) dif t )&<=integral_0^delta e^(-x t) C t^3 dif t  
$
令 $u=x t$ 那么 :
$
C integral_0^delta e^(-x t) t^3 dif t = C/x^4 integral_0^(x delta) e^(-u) u^3 dif u
$
$
  lim_(x -> +oo) x abs(integral_0^delta e^(-x t) O(t^3) dif t ) &<=lim_(x-> +oo) C/x^3 integral_0^(x delta) e^(-u) u^3 dif u \
  &=0 dot Gamma (4)=0
$
$
  lim_(x -> +oo) x integral_0^delta e^(-x tan t) dif t
  = lim_(x -> +oo) x integral_0^delta e^(-x t) dif t
  = lim_(x -> +oo) (1-e^(-x delta))=1
$
因此结果为1.
#idea[
以上的例子都显示含有e,对于没有e的情况，我们可以人为构造e来使用Laplace方法.
]
#example[
  
    4、求证：对于非负整数n,考虑积分
    $
      I_n=integral_0^(pi/2) sin^(2n)x dif x,text("则")
    $
    $
      lim_(n ->+oo) sqrt(n) I_n= sqrt(pi)/2
    $
  
]

#proof[
用Taylor展开，有
$
  ln sin^2 x= -(x- pi/2)^2 + o((x- pi/2)^2) quad (x -> pi/2)
$
对于任意$epsilon in (0,1)$,我们知道存在$delta>0$使得当$x in [pi/2 -delta,pi/2]$时，有
$
  -(1+epsilon)(x- pi/2)^2 <= ln sin^2 x <= -(1-epsilon)(x- pi/2)^2
$
现在一方面，
$
  integral_0^(pi/2) sin^(2n)x dif x&<= integral_(pi/2 -delta)^(pi/2) sin^(2n)x dif x + integral_0^(pi/2 -delta) sin^(2n)x dif x \
  &<= integral_(pi/2 -delta)^(pi/2) e^(-(1-epsilon) n (x- pi/2)^2) dif x + (pi/2-delta)(cos delta)^(2n) 
$
令$t=sqrt(n(1-epsilon)) (x-pi/2)$,得:
$
  integral_(pi/2 -delta)^(pi/2) e^(-(1-epsilon) n (x- pi/2)^2) dif x = 1/sqrt(n(1-epsilon))integral_(-sqrt(n(1-epsilon)) delta)^0 e^(-t^2) dif t
$
$
  limsup_(n -> +oo) sqrt(n) I_n <= 1/sqrt(1-epsilon) dot integral_(-oo)^0 e^(-t^2) dif t = sqrt(pi)/(2 sqrt(1-epsilon))
$
另一方面，
$
  integral_0^(pi/2) sin^(2n)x dif x&>= integral_(pi/2 -delta)^(pi/2) sin^(2n)x dif x \
  &>= integral_(pi/2 -delta)^(pi/2) e^(-(1+epsilon) n (x- pi/2)^2) dif x
$
同理令$t=sqrt(n(1+epsilon)) (x-pi/2)$,得:
$
  integral_(pi/2 -delta)^(pi/2) e^(-(1+epsilon) n (x- pi/2)^2) dif x = 1/sqrt(n(1+epsilon))integral_(-sqrt(n(1+epsilon)) delta)^0 e^(-t^2) dif t
$
$
  liminf_(n -> +oo) sqrt(n) I_n >= 1/sqrt(1+epsilon) dot integral_(-oo)^0 e^(-t^2) dif t = sqrt(pi)/(2 sqrt(1+epsilon))
$
综上所述，
$
  sqrt(pi)/(2 sqrt(1+epsilon)) <= liminf_(n -> +oo) sqrt(n) I_n <= limsup_(n -> +oo) sqrt(n) I_n <= sqrt(pi)/(2 sqrt(1-epsilon))
$
令$epsilon -> 0$，可得所需结论.]
#theorem[
  根据点火公式，有
$
  I_n=integral_0^(pi/2) sin^(2n)x dif x= pi/2 dot (2n-1)!!/(2n)!!
$
  因此我们也得到了斯特林公式的一个推论：
$
  (2n)!!/(2n-1)!! ~ sqrt(pi n) quad (n -> +oo)
$
]
== Taylor定理
#idea[伸缩步长,两边夹方法]
#example[
设 $f in C[0, +infinity) inter D^2(0, +infinity)$ 且存在常数 $C > 0$ 使得 $f''(x) <= C/x^2$ 对所有 $x > 0$ 成立.则有
$
  lim_(x -> 0^+) x f'(x) = 0
$
]
#proof[
对任意固定的 $eta in (0, 1)$ 和 $x > 0$，我们使用 Taylor 中值定理.

首先，在区间 $(x - eta x, x)$ 上，存在 $theta in (x - eta x, x)$ 使得：
$
  f(x - eta x) = f(x) - eta x f'(x) + (eta x)^2 f''(theta)/2
$

从中解出 $f'(x)$：
$
  f'(x) = (f(x) - f(x - eta x))/(eta x) + (eta x) f''(theta)/2
$

由于 $f''(theta) <= C/theta^2$ 且 $theta > x - eta x = x(1 - eta)$，有 $theta^2 > x^2(1 - eta)^2$，因此：
$
  f'(x) <= (f(x) - f(x - eta x))/(eta x) + (eta x) C/(2 theta^2)
$

两边乘以 $x$：
$
  x f'(x) &= (f(x) - f(x - eta x))/eta + (eta x^2) C/(2 theta^2) \
          &<= (f(x) - f(x - eta x))/eta + C eta/2 dot 1/(1 - eta)^2 \
          &= (f(x) - f(x - eta x))/eta + C eta/(2 (1 - eta)^2)
$

其次，在区间 $(x, x + eta x)$ 上，存在 $theta.alt in (x, x + eta x)$ 使得：
$
  f(x + eta x) = f(x) + eta x f'(x) + (eta x)^2 f''(theta.alt)/2
$

从中解出 $f'(x)$：
$
  f'(x) = (f(x + eta x) - f(x))/(eta x) - (eta x) f''(theta.alt)/2
$

由于 $f''(theta.alt) <= C/theta.alt^2$ 且 $theta.alt > x$，有 $theta.alt^2 > x^2$，因此：
$
  f'(x) >= (f(x + eta x) - f(x))/(eta x) - (eta x) C/(2 x^2)
$

两边乘以 $x$：
$
  x f'(x) >= (f(x + eta x) - f(x))/eta - C eta/2
$

综合上述两个不等式，我们得到：
$
  (f(x + eta x) - f(x))/eta - C eta/2 <= x f'(x) <= (f(x) - f(x - eta x))/eta + C eta/(2 (1 - eta)^2)
$

现在考虑极限行为.由于 $f in C[0, +infinity)$，当 $x -> 0^+$ 时：
$
  lim_(x -> 0^+) (f(x + eta x) - f(x))/eta = 0, quad lim_(x -> 0^+) (f(x) - f(x - eta x))/eta = 0
$

设 $L = lim inf_(x -> 0^+) x f'(x)$ 和 $M = lim sup_(x -> 0^+) x f'(x)$.由上述不等式，取 $x -> 0^+$，我们有：
$
  - C eta/2 <= L <= M <= C eta/(2 (1 - eta)^2)
$

由于这对任意 $eta in (0, 1)$ 都成立，令 $eta -> 0^+$，得到：
$
  0 <= L <= M <= 0
$

因此 $L = M = 0$，即 $lim_(x -> 0^+) x f'(x) = 0$.
]

为什么我们的条件看起来不是对称的呢？事实上，对于 $f''(x) >= - C/x^2$ 的情况，使用几乎完全相同的方法也能得到相同的结论.这表明，*无论是上界还是下界的二阶导数控制，都足以确保* $lim_(x -> 0^+) x f'(x) = 0$.

#remark[这个证明的深刻之处在于：*单侧控制足以扼杀震荡*.表面看，条件 $f''(x) <= C/x^2$ 只限制了 $f$ 向上弯曲的程度，似乎允许 $f$ 自由地向下剧烈震荡.但实际并非如此.函数要产生高频震荡（如振幅不衰减的振荡），其二阶导数必须在正负两个方向都具有足够大的振幅，以完成完整的波峰-波谷-波峰循环.单侧限制破坏了这种对称性，使震荡无法持续.

函数 $f(x) = x sin(1/x)$（定义 $f(0) = 0$）是一个绝佳的反例：它在 $x -> 0^+$ 时连续，但 $x f'(x) = x sin(1/x) - cos(1/x)$ 不趋于 0.计算可得其二阶导数的主项为 $f''(x) ~ cos(1/x)/x^3$，其振幅以 $1/x^3$ 增长，正负交替，*突破了*无论是上界还是下界的二阶导数控制.这正说明 $C/x^2$ 是临界尺度——弱于此尺度的控制无法抑制震荡，而达到此尺度的单侧控制已足以确保 $x f'(x) -> 0$.
]

#example[
  设$Omega subset bb(R)^m$是凸域，$
  f in C^2(Omega,bb(R))$且$f(bold(x))>= f(bold(x)_0) + text("D")f(bold(x)_0) quad (forall bold(x),bold(x)_0 in Omega)
$，
求证：$f$的Hessian矩阵$H_f$在$Omega$上半正定.
]
#proof[
$
  f(bold(x))=f(bold(x)_0 + (bold(x)-bold(x)_0))\
  =f(bold(x)_0) + text("D")f(bold(x)_0)(bold(x)-bold(x)_0) + 1/2 (bold(x)-bold(x)_0)^ top H_f (bold(x)_0)(bold(x)-bold(x)_0) + o(|bold(x)-bold(x)_0|^2)
$
由题设条件$f(bold(x))>= f(bold(x)_0) + text("D")f(bold(x)_0)(bold(x)-bold(x)_0) quad (forall bold(x),bold(x)_0 in Omega)$，可得
$
  1/2 (bold(x)-bold(x)_0)^ top H_f (bold(x)_0)(bold(x)-bold(x)_0) + o(|bold(x)-bold(x)_0|^2) >=0
$
但是这里没有消去$o$项，我们陷入了困难.

为了解决这个问题，我们引入动态步长的方法，即令$bold(x)=bold(x)_0 + t bold(h),t>0$，其中$bold(h)$是任意固定的向量.代入上式，得到
$
  1/2 t^2 bold(h)^ top H_f (bold(x)_0) bold(h) + o(t^2 |bold(h)|^2) >=0\
  bold(h)^ top H_f (bold(x)_0) bold(h) + (o(t^2 |bold(h)|^2))/t^2 \
  =>^(t->0^+) bold(h)^ top H_f (bold(x)_0) bold(h) >=0\
$
]
== 一致收敛，一致连续
=== 一致收敛
#theorem[
  $f(x)$在$(0,+infinity)$上连续，那么$integral_0^infinity e^(-alpha x)f(x)dif x$关于$alpha$在$(0,+infinity)$上一致收敛$<==>integral_0^infinity f(x) dif x$收敛
  ]

#tip-box[$<==$可以使用Able判别法，而$==>$可以先在有限区间研究.]
由一致收敛的Cauchy收敛准则可知：
$forall epsilon >0 , exists A>0 , s.t. forall N_2>N_1>A text("和任意") forall alpha >0 , abs(integral_(N_1)^(N_2) e^(-alpha x)f(x) dif x) < epsilon$

$F(alpha)=integral_(N_1)^(N_2) e^(-alpha x)f(x) dif x$是连续函数，因此:

$
lim_(alpha->0^+) abs(integral_(N_1)^(N_2) e^(-alpha x)f(x) dif x) =abs(integral_(N_1)^(N_2) f(x) dif x)<= epsilon
$
由Cuachy，$integral_1^infinity f(x) dif x$收敛.

类似的，对$forall epsilon >0,forall delta>0,text("考虑任意的"),n_1,n_2 in (0,delta),n_2>n_1$，使用一致收敛的Cauchy收敛准则，$lim_(alpha->0^+) |integral_(n_1)^(n_2) e^(-alpha x)f(x) dif x|=|integral_(n_1)^(n_2) f(x) dif x|<=epsilon$,可以证明$integral_0^1 f(x) dif x$收敛. 

#example[
  $
    sum_(n=1)^infinity sin(n x)/n,
  $在$[0,2 pi]$不一致收敛
]
#proof[
  1、$sum_(n=1)^infinity sin(n x)/n->$Sawtooth,即级数在$[0,2 pi]$上点态收敛于$f(x)= (pi - x)/2$，但在$x=0,2 pi$处Sawtooth不连续，所以不一致收敛.
  
  2、假设一致收敛，对$epsilon=sin(1)dot ln 2,exists N>0,forall x in [0,2 pi]$
  $
    abs(sum_(n=N)^(2N) sin(n x)/n ) < epsilon
  $
  取$x=1/N$，则
  $
    abs(sum_(n=N)^(2N) sin(n/N)/n) >= sin 1 dot sum_(n=N)^(2N)1/n\
    > sin 1 dot integral_N^(2N) 1/x dif x = sin 1 dot ln 2 = epsilon
  $
]
#remark[以上两题都用了Cauchy收敛准则,并且都体现了一定的换序思想.]
=== 一致连续
#example[
  设 $f:[1,infinity)->bb(R)$满足对某个$L>0$成立
  $
    |f(x_1)-f(x_2)|<= L|x_1 - x_2| quad (forall x_1,x_2 in [1,+infinity))
  $
  证明: $f(x)/(x+ln(1+1/x))$在$[1,+infinity)$上一致连续.
]
#proof[
  $ h(x) eq.delta x+ln(1+ 1/x)\
    forall x>=1, quad x< h(x) 
  $
  并且
  $
    0<h'(x)= 1 - 1/(x^2 + x)<1 
  $
  $
    abs(f(x)-f(1))<=L abs(x-1)=L |x-1|\
    => abs(f(x))<= L |x-1| + abs(f(1))<= L x + abs(f(1))\
  $
  $
    forall x_1 ,x_2 in [1,+infinity)\ 
    &abs(f(x_1)/h(x_1) - f(x_2)/h(x_2))\
    &= abs((f(x_1) h(x_2) - f(x_2) h(x_1))/(h(x_1) h(x_2)))\
    &= abs([f(x_1) - f(x_2)] h(x_2) + f(x_2) [h(x_2) - h(x_1)])/(h(x_1) h(x_2))\
    &<= L abs(x_2- x_1)/h(x_1) + |f(x_2)|abs((h(x_2) - h(x_1)))/(h(x_1) h(x_2))\
    &(text("利用")    0<1/h(x)<1/x<=1 )\
    &<= L abs(x_2- x_1)+ abs(f(x_2))/x_2 abs(integral_(x_1)^(x_2) h'(x) dif x)\
    &<= (L+(L x_2 + abs(f(1)))/ x_2) abs(x_2 - x_1)\
    &<= (2L + abs(f(1))) abs(x_2 - x_1)\
  $

]

#pagebreak()
== Abel变换
#align(center)[
#canvas({
  import draw: *
  let size=10pt
  let x1=4
  let x2=8
  let y1=3
  let y2=5.5
  let x3=15
  let y3=8
  rect((0,0),(x1,y1),fill:yellow,name:"under1")
  rect((x1,0),(x2,y2),fill:purple,name:"under2")
  rect((x2,0),(x3,y3),fill:purple,name:"under3")
  rect((0,y1),(x1,y2),fill:yellow,name:"over1")
  rect((0,y2),(x2,y3),fill:blue,name:"over2")
set-style(
    mark:(
        symbol:none,
        end:"straight"
    )
)
line((0,0),(16,0))
line((0,0),(0,9))
content((0,-0.35),[#box[#set text(20pt);$o$]])
content((x1,-0.35),[#box[#set text(20pt);$b_1$]])
content((x2,-0.35),[#box[#set text(20pt);$b_2$]])
content((x3,-0.35),[#box[#set text(20pt);$b_3$]])
content((-0.35,y1),[#box[#set text(20pt);$a_1$]])
content((-0.35,y2),[#box[#set text(20pt);$a_2$]])
content((-0.35,y3),[#box[#set text(20pt);$a_3$]])
content("under1",[#box[#set text(20pt);$a_1 b_1$]])
content("under2",[#box[#set text(20pt);$a_2 (b_2- b_1)$]])
content("under3",[#box[#set text(20pt);$a_3 (b_3 - b_2)$]])
content("over1",[#box[#set text(20pt);$b_1 (a_2 - a_1)$]])
content("over2",[#box[#set text(20pt);$b_2 (a_3 - a_2)$]])
})
]
如图，在公式
  $
  sum_(n=p)^(q)a_n (b_n-b_(n-1))=a_q b_q - a_p b_(p-1)- sum_(n=p)^(q-1)(a_(n+1)-a_n) b_n
  $
中令$p=2,q=3$,那么

#align(center)[
  #set text(size:20pt)
  #text($sum_(n=2)^(3)a_n (b_n-b_(n-1))$,fill:purple)=$underbrace(a_3 b_3,"全部面积")$-#text($a_2b_1$,fill:yellow)-#text($b_2(a_3 - a_2)$,fill:blue)
]

#pagebreak()
== 换序法求积分
#example[
设$f(x)=integral_0^x integral_t^x e^(-s^2)dif s dif t$,求$f(x)$]
$e^(-s^2)$的原函数不能被初等函数表示，考虑用换序法：
$
  f(x)=integral_0^x integral_0^s e^(-s^2) dif t dif s= integral_0^x t e^(-t^2) dif t= -1/2 e^(-x^2)+1/2
$
或者，我们另辟蹊径:
设$G(x,t)=integral_t^x e^(-s^2)dif s $,则$f(x)=integral_0^x G(x,t) dif t$，由Leibniz公式可得:
$
  f'(x)= G(x,x) +integral_0^x (partial G)/( partial x )dif t =0 + integral_0^x e^(-x^2) dif t = x e^(-x^2)\
  f(0)=0\
  f(x)=integral_0^x t e^(-t^2) dif t = -1/2 e^(-x^2)+1/2
$
#example[利用求导的方法，计算
$
  sum_(k=0)^infinity cos(k x) /k^2
$
并令$x=0$，求出$sum_(k=1)^infinity 1/k^2$
]
#tip-box[
  解决这个问题需要的必要知识见@SawtoothFunction
]
#example[
  $
  integral_0^1 (ln x)/(1-x) dif x
  $
]
$
  integral_0^1 (ln x)/(1-x) dif x = integral_0^1 ln x (sum_(n=0)^infinity x^n) dif x = sum_(n=0)^infinity integral_0^1 x^n ln x dif x \
  = sum_(n=0)^infinity [x^(n+1)/(n+1) ln x - x^(n+1)/(n+1)^2]_0^1 = - sum_(n=1)^infinity 1/n^2 = - pi^2/6
$
其中 $sum_0^infinity x^n ln x$在(0,1)上内闭一致收敛.
$forall delta in (0,1/2)$,当$x in [delta,1-delta]$时,
由Abel判别法:
$sum_0^infinity x^n$显然一致收敛，而$ln x$在$[delta,1-delta]$上与$n$无关，可认为单调一致有界，因此$sum_0^infinity x^n ln x$在$[delta,1-delta]$上一致收敛,即在$(0,1)$内闭一致收敛.
#remark[本题中级数展开的方法似乎还有推广空间,展开为Fourier级数行不行?]
#idea[方型区域上单独的$x$和$y$可以互换]
#example[设$D=[0,1]times[0,1],f text("和") g$都是$[0,1]$上的连续函数，求证：
$
  integral.double_D f(x)+g(y) dif x dif y = (integral_0^1 f(x)+g(x) dif x)
$]
证明很容易.
#example[设$D=[0,1]times[0,1]$
求证： 
$
  1<=integral.double_D sin(x^2)+cos(y^2) dif x dif y<=sqrt(2)
$
]
== 特殊积分
#example[
  计算积分 $
  I=integral_0^(1) ln(Gamma (x)) dif x
  $
]
$
  2 I=integral_0^(1) (ln(Gamma (x)) + ln(Gamma (1-x))) dif x \
  =integral_0^(1) ln(pi/sin(pi x)) dif x \
  =ln pi - integral_0^(1) ln(sin(pi x)) dif x \
  =ln pi - integral_0^(pi) ln(sin t) dif t / pi \
  =ln pi - 2/pi dot (-pi/2 ln 2)
  =ln 2 pi\
  I=1/2 ln(2 pi)=ln sqrt(2 pi)
$
#example[
  计算积分 $
  I=integral_0^(pi/2) sin^7 x cos^(1/2) x dif x
  $
]
$
  I&=1/2 B((7+1)/2,(1/2 +1)/2)=1/2 B(4,3/4)=1/2 (Gamma(4) Gamma(3/4))/Gamma(4+3/4)\ &=1/2 (Gamma(4) Gamma(3/4))/(Gamma(3/4)dot (3/4+1) dot (3/4+2) dot (3/4+3) dot 3/4)=256/1125
$
== 复积分
#idea[看到$sin,cos$可以利用欧拉公式，使用复积分方法求实积分]
#example[
  $
  &integral_0^1 x^y sin(ln 1/x)dif x
  =integral_0^1 x^y Im(e^(i ln 1/x)) dif x
  =Im(integral_0^1 x^(y - i) dif x)\
  &=Im(1/(y - i +1) (x^(y - i +1))|_0^1)
  =Im(1/(y +1 - i))
  =1/((y +1)^2 +1) quad(y>-1)
  $
]
#example[
  $
    I=integral_0^(+infinity)cos(x^2)dif x\

  $
]<FresnelIntegral>
$
    &I =Re(integral_0^(+infinity) e^(i x^2) dif x)\
    &=Re(integral_0^(+infinity) e^(i x^2) e^(-i pi/4) dif (x dot e^(i pi/4) ) )\
    &=Re(e^(-i pi/4) integral_0^(+infinity e^(i pi/4)) e^(-t^2) dif t)\
    &=Re(e^(-i pi/4) dot sqrt(pi)/2)\
    &=sqrt(pi)/2 dot cos(pi/4)\
    &=sqrt(pi)/2 dot sqrt(2)/2 = sqrt(2pi)/4
$
#warning-box()[这里不完全严谨.]
== 凸分析
#theorem[$(a,b)$上的下凸函数$f$一定内闭Lipschitz连续]
#proof[

$text("任意")[c,d] subset(a,b) text("对于") forall x_0 in [c,d] 
$
, 取$c<=x_0<d$，取$a<e<c quad d<h<b$ 由下凸函数的斜率比较定理知：对于足够小的$Delta x>0$ 使得$a<e<c<=x_0<x_0+Delta x<=d<h<b$有
$
  (f(c)-f(e))/(c-e)<=(f(x_0+Delta x)-f(x_0))/(Delta x)<=(f(h)-f(d))/(h-d)
$

令$M=max{|(f(c)-f(e))/(c-e)|,|(f(h)-f(d))/(h-d)|}$
，有
$
  |f(x_0+Delta x)-f(x_0)|<=M Delta x
$
对于任意的$x_1,x_2 in [c,d]$，不妨设$x_1<x_2$，取$x_0=x_1,Delta x=x_2 - x_1$，则有
$
  |f(x_2)-f(x_1)|<=M |x_2 - x_1|
$
]

#example[ 
设 $f: [0, +infinity) -> [0, +infinity) $ 是一个非负函数，满足：
- $f(x) = 0 $ 当且仅当$x = 0$；
- $f $ 是*严格上凸*函数.
可以推出：
$f $ 在 $(0, +infinity) $ 上严格递增；]<上凸函数的单调性>
#proof[*反证*：假设存在 $0 < a < b $使得 $f(a) >= f(b) $.

由于 $f $ 严格上凸，对任意 $t > b $，将 $b $ 表示为 $a $ 与 $t $ 的凸组合：
$b = lambda a + (1 - lambda)t, quad lambda = (t - b)/(t - a) in (0,1). $

由严格上凸性，
$
  f(b) > lambda f(a) + (1 - lambda)f(t) >= lambda f(b) + (1 - lambda)f(t),
$
因为 $ f(a) >= f(b) $整理得
$
  f(b) > lambda f(b) + (1 - lambda)f(t) arrow.r (1-lambda)f(b) > (1-lambda)f(t).
$
由于 $ 1-lambda > 0 $我们得到 $ f(b) > f(t) $这意味着对任意 $t > b $， $f(t) < f(b) $

现在考虑割线斜率.固定 $b $，对于 $t > b $，由严格上凸性，割线斜率 $ (f(t)-f(b))/(t-b) $ 关于 $t $ 是严格递减的.由于 $f(t) < f(b) $，该斜率恒为负.

取$s > t > b $，则有
$
  (f(s)-f(b))/(s-b) < (f(t)-f(b))/(t-b) < 0.
$
设
$ m = (f(t)-f(b))/(t-b) < 0. $
于是
$ f(s) < f(b) + m(s-b). $

由于 $m < 0 $，当 $s $ 足够大时，右端为负，即 $f(s) < 0 $，与 $f(x) >= 0 $ 矛盾.

因此，假设不成立，从而对任意 $0 < a < b $ 必有 $f(a) < f(b) $，即 $f $ 在 $(0,+infinity) $ 上严格递增.]
== 拓扑与度量空间

#theorem[设 $f: [0, +infinity) -> [0, +infinity) $ 是一个非负函数，满足：
- $f(x) = 0 $ 当且仅当$x = 0$；
- $f $ 是*严格上凸*函数.
并且在$bb(R)$上定义函数$d(x_1,x_2)=f(|x_1-x_2|)$,则$d$是$bb(R)$上的度量]
#proof[任取 $x, y, z in bb(R) $,令$a = | x - z |, quad b = | z - y |, quad c = | x - y |$
由实数绝对值的三角不等式，有  
$c <= a + b $
由@上凸函数的单调性， $f $ 非负、严格递增，因此
$f(c) <= f(a + b) .$
若 $a = 0 $ 或 $b = 0 $，则不等式显然成立.
假设 $b>=a>0$,于是由割线的斜率递减：$ f(a + b) - f(b) < f(a) - f(0)
$
$ f(a + b) - f(b) < f(a) quad => quad f(a + b) < f(a) + f(b)$.
因此，
$ d(x, y) = f(c) <= f(a + b) < f(a) + f(b) = d(x, z) + d(z, y) .
$]

#theorem[
设 $(X_1, d_1)$ 和 $(X_2, d_2)$ 为两个度量空间.在乘积空间 $X_1 times X_2$ 上定义函数
$
  d((x_1,x_2),(y_1,y_2)) = sqrt(d_1(x_1,y_1)^2 + d_2(x_2,y_2)^2).
$
我们将证明 $d$ 满足三角不等式.]
#proof[
任取三点 $(x_1,x_2), (z_1,z_2), (y_1,y_2) in X_1 times X_2$，记：
$
  a_1 &= d_1(x_1, z_1), & a_2 &= d_2(x_2, z_2), \
  b_1 &= d_1(z_1, y_1), & b_2 &= d_2(z_2, y_2), \
  c_1 &= d_1(x_1, y_1), & c_2 &= d_2(x_2, y_2).
$

由于 $d_1$ 和 $d_2$ 均为度量，由三角不等式得：
$
  c_1 <= a_1 + b_1, quad c_2 <= a_2 + b_2.
$

两边平方并相加：
$
  c_1^2 + c_2^2 <= (a_1 + b_1)^2 + (a_2 + b_2)^2.
$

开平方：
$
  sqrt(c_1^2 + c_2^2) <= sqrt((a_1 + b_1)^2 + (a_2 + b_2)^2).
$

在 $RR^2$ 中，对向量 $(a_1, a_2)$ 和 $(b_1, b_2)$ 应用闵可夫斯基不等式（即欧几里得范数的三角不等式）：
$
  sqrt((a_1 + b_1)^2 + (a_2 + b_2)^2) <= sqrt(a_1^2 + a_2^2) + sqrt(b_1^2 + b_2^2).
$

将上述结果链式组合：
$
  sqrt(c_1^2 + c_2^2) <= sqrt(a_1^2 + a_2^2) + sqrt(b_1^2 + b_2^2).
$

根据 $d$ 的定义，这等价于：
$
  d((x_1,x_2), (y_1,y_2)) <= d((x_1,x_2), (z_1,z_2)) + d((z_1,z_2), (y_1,y_2)).
$

因此，$d$ 满足三角不等式.
]
#remark[
如果 $q >= 1$， $d((x_1,x_2),(y_1,y_2)) = (d_1(x_1,y_1)^q + d_2(x_2,y_2)^q)^(1/q)$ 也是度量，令 $q = infinity$ 此时 $d = op("max") {d_1(x_1,y_1), d_2(x_2,y_2)}$.
]

#pagebreak()
#align(center)[
  #v(20em)
  #set text(size: 19pt, weight: "bold")
  卡拉玛佐夫即将死去，卡拉玛佐夫就要自杀。\ 人们会记住这一天的，\ 必须对得起他这颗诗人的灵魂,\ 也不枉他把自己的生命蜡烛是从两头儿一起点燃烧光的。
]
#align(right)[  #set text(size: 18pt)
  ——陀思妥耶夫斯基   《卡拉马佐夫兄弟》]
#pagebreak()
= 傅里叶分析
== 基础知识
$f:bb(R)->bb(R)$为$T$周期函数,$omega_0= (2 pi)/T$,它的Fourier级数为:
$
  c_0 + sum_(n=1)^infinity 2 abs(c_n) cos(n omega_0 x + arg(c_n))
$
其中
$
  c_n=1/T integral_0^T f(x) e^(-i n omega_0 x) dif x
$
== 傅里叶变换求积分
#example[
  计算积分 $
  I=integral_(-infinity)^(+infinity) (sin x)/x (sin x\/3)/(x\/3) dif x
  $
]
$
  I=cal(F)[(sin x)/x dot (sin x\/3)/(x\/3)](0)=1/(2 pi)(cal(F)[(sin x)/x] * cal(F)[(sin x\/3)/(x\/3)])(0)= pi 
$
== 傅里叶级数的收敛
#theorem[$
  sum_(k=1)^infinity sin(k t)/k =cases((pi -  t)/2 quad (0<t<2 pi) , 0 quad (t=0))
$]<SawtoothFunction>
#proof[
  Sawtooth 函数可以 Dirichlet 核的积分得到:
  $
    I_n (t)=integral_pi^t 1 + 2 cos x + 2 cos 2x + dots + 2 cos (n x) dif x\ = (t - pi) + 2(sin t + (sin 2t)/2 + dots + (sin (n t))/n)
  $
  $
    sum_(k=1)^n (sin (k t))/k = 1/2 I_n (t) + (pi - t)/2
  $
  #remark[从$pi$开始积分而不是从0开始是为了构造$pi - t$,非常技巧.这样只要证明$I_n (t)->0$.]
  另一方面
  $
    I_n (t)=integral_pi^t (sin((n + 1/2) x))/(sin(x/2)) dif x
  $
  $
    = - cos((n+1/2) t)/((n+1/2) sin(t/2)) + 1/((n+1/2)) integral_pi^t cos((n+1/2) x) (1/sin(x/2))' dif x
  $
  不把导数算出,是因为后续放缩我们要放掉符号不定的$cos((n+1/2) x) $.
  在$t in (0, 2 pi)$上，我们知道$sin(x/2)>0$,它的单调性和$t$和$pi$的大小关系有关.无论如何，对$integral$放缩时积分上界必须大于下界.
  $
    abs(I_n (t))<=1/((n+1/2) sin(t/2)) +1/((n+1/2)) integral_(min(t,pi))^(max(t,pi)) sign(t-pi) (1/sin(x/2))' dif x 
  $
  应当注意,$sign(t-pi) (1/sin(x/2))'$是在积分区域恒为正的.
  $
    abs(I_n (t))
    &<= 1/((n+1/2) sin(t/2)) + 1/((n+1/2)) (1/sin(t/2) -1)\
    &<=2/((n+1/2) sin(t/2))->0 quad (n->+infinity,forall t in (0,2 pi))
  $
  #remark[这个证明防御性很强,没有关于Sawtooth 函数可以 Dirichlet 核的关系的知识是不可能想到的.即使想到，谁能想到利用两个积分相等呢？而且你需要提前知道结果才可能想到从$pi$开始积分,才能证的比较顺.后面的放缩也精彩.]
]

关于Sawtooth 函数的傅里叶级数的Gibbs现象，有以下方法:
$
    
  sum_(k=1)^n (sin (k t))/k &= 1/2 I_n (t) + (pi - t)/2\
  R_n (t)=sum_(k=1)^n (sin (k t))/k - (pi - t)/2 &= 1/2 integral_pi^t (sin((n + 1/2) x))/(sin(x/2)) dif x, quad t in (0,2 pi)\
  R_n^' (t)&= 1/2 (sin((n + 1/2) t))/(sin(t/2))\
$
注意到第一个正的极值点位于$t_N=pi/(n+1\/2)$
$
  R_n (t_N)&=1/2 integral_pi^(t_N) (sin((n + 1/2) x))/(sin(x/2)) dif x\
  &=1/2 (integral_0^t_N - integral_0^pi) (sin((n + 1/2) x))/(sin(x/2)) dif x\
$
$
  integral_0^pi (sin((n + 1/2) x))/(sin(x/2)) dif x=integral_0^pi 1+ 2 cos x + 2 cos 2x + dots + 2 cos (n x) dif x=pi\

  R_n (t_N)=1/2 integral_0^t_N (sin((n + 1/2) x))/(sin(x/2)) dif x - pi/2\

  text("并且"),quad sin((n + 1/2) x)/sin(x/2)=sin((n+1/2)) dot 1/sin(x/2) \
  =sin((n+1/2)x) dot (sin(x\/2)/x + 1- sin(x\/2)/x)/sin(x/2)\
  =sin((n+1/2)x) dot (1/x + (x-sin(x\/2))/(x sin(x\/2)))\
$
$
  R_n (t_N)= 1/2 integral_0^t_N sin((n+1/2) x) / x dif x + 1/2 integral_0^t_N sin((n+1/2) x) (x - sin(x\/2))/(x sin(x\/2)) dif x - pi/2\
$
#important-box[但是 $x-sin(x\/2)$是不好的，它的无穷小的阶数是1，但$x-2 sin(x\/2)$的无穷小阶数是3.我们希望把积分拆为两部分，一部分是主项，一部分是无穷小阶数高的项.]
$
  sin((n + 1/2) x)/sin(x/2)=sin((n+1/2)) dot 1/sin(x/2) \
  =sin((n+1/2)x) dot (sin(x\/2)/(x\/2) + 1- sin(x\/2)/(x\/2))/sin(x/2)\
  =sin((n+1/2)x) dot (2/x + (x-2 sin(x\/2))/(x sin(x\/2)))\
  R_n (t_N)=  integral_0^t_N sin((n+1/2) x) / x dif x + 1/2 integral_0^t_N (sin((n+1/2) x) dot (x-2 sin(x\/2)))/(x sin(x\/2)) dif x - pi/2\
  =integral_0^pi (sin u) / u dif u + 1/2 integral_0^t_N (sin((n+1/2) x) dot (x-2 sin(x\/2)))/(x sin(x\/2)) dif x - pi/2\
  =text("Si")(pi) -pi/2 + 1/2 integral_0^t_N (sin((n+1/2) x) dot (x-2 sin(x\/2)))/(x sin(x\/2)) dif x \
$
注意到$lim_(x->0) (x-2 sin(x\/2))/(x sin(x\/2))=0$，而$N->infinity ,quad t_N->0$
，所以第二个积分趋于0，因此
$
  lim_(n->infinity) R_n (t_N)=text("Si")(pi) -pi/2 approx 0.0895 dot pi
$
这里的$pi$是总跳变值.
#pagebreak()
#align(center)[
  #v(20em)
  #set text(size: 25pt, weight: "bold")
  要相信啊：你的诞生绝非枉然！\
  你的生存和磨难绝非徒然！\
]
#align(right)[  #set text(size: 20pt)
  ——马勒   《第二交响曲：复活》]
#pagebreak()
= 线性代数
== 线性无关性
#idea[利用空间维度]
#example[
  设$p_0,p_1,dots,p_m$是$cal(P)_m (bold(text("F")))$中的$m+1$个多项式，且$p_j (2)=0$，证明：$p_0,p_1,dots,p_m$线性相关.
]
#proof[
  $p_j =(x-2)q_j$ , 其中$q_j in cal(P)_(m-1) (bold(text("F")))$
  ,所以$q_0,q_1,dots,q_m$是$cal(P)_(m-1) (bold(text("F")))$中的$m+1$个多项式，而$dim cal(P)_(m-1) (bold(text("F")))=m$，所以$q_0,q_1,dots,q_m$线性相关，进而$p_0,p_1,dots,p_m$线性相关.
]
== 多项式的根与整除
#example[
  设$f_1(x),f_2(x) in text("K")[x].$若$
  (x^2+x+1)|(f_1(x^3)+x f_2(x^3)),
  $求证:$(x-1)|f_1(x),(x-1)|f_2(x)$
]
#proof[
  设$epsilon_k=e^(k (2 pi) / 3), k=1,2$
，则$epsilon_1,epsilon_2$是$x^2+x+1$的两个根.
$f(epsilon_k^3)=f(1)$.\

$
  cases(f_1(1)+epsilon_1f_2(1)=0\, ,
  f_1(1)+epsilon_2f_2(1)=0\,)
$
$
  mat(delim:"|",1 , epsilon_1;1 , epsilon_2)=epsilon_2-epsilon_1!=0
$
$
  =>f_1(1)=0,f_2(1)=0
$

== 谱定理
#idea[对于半正定对角矩阵成立的定理，由于谱定理，往往对一般的实对称矩阵也成立.]

#example[
  证明：二次型$f(x,y)=A x^2 + B x y +B y x +C y^2$在单位球面上的最大最小值分别为其对应实对称矩阵的最大和最小特征值.
]
#proof[
  设$
     Phi = mat(A , B ; B , C)
   $
   由谱定理，存在正交矩阵$Q$使得
   $
     Q^top Phi Q = mat(lambda_1 , 0 ; 0 , lambda_2)
   $

    其中$lambda_1 , lambda_2$为$Phi$的特征值.
    令 $X = vec(x,y)$,则$f(x,y)= X^top Phi X$.
    
    令$Y = Q^top X$,则$X = Q Y$.
    $ f(x,y)= Y^top (Q^top Phi Q) Y = Y^top mat(lambda_1 , 0 ; 0 , lambda_2) Y = lambda_1 y_1^2 + lambda_2 y_2^2 $

    由于$Y^top Y =X^top Q Q^top X = X^top X=x_1^2 +x_2^2 =1，$
    
    此时题目结论是显然的.
   ]
]
#pagebreak()
#align(center)[
  #v(20em)
  #set text(size: 22pt, weight: "bold")
  向之所欣，俯仰之间，已为陈迹，犹不能不以之兴怀
]
#align(right)[  #set text(size: 18pt)
  ——王羲之   《兰亭集序》]
#pagebreak()
= 概率论与数理统计
== 大数定律和中心极限定理
#idea[用大数定律证明依概率收敛]
#example[
  设${X_n}$为一独立同分布随机变量序列，$Var(X_n)=sigma^2<+infinity$，
  
  令$S_n^2=1/n sum_(i=1)^n (X_i - overline(X))^2$,
  求证：
  $S_n^2 stretch(->)^P sigma^2$
]
#proof[
  $
    S_n^2=1/n sum_(i=1)^n X_i^2 - overline(X)^2
  $
  由于$X_n$独立同分布，$X_n^2$独立同分布，且
  $ bb(E)(X_n^2) = sigma^2 + mu^2 $

由大数定律知：
$
  1/n sum_(i=1)^n X_i^2 stretch(->)^P sigma^2 + mu^2
$
又由大数定律知：
$
  overline(X) stretch(->)^P mu
$
因此，
$
  S_n^2 stretch(->)^P (sigma^2 + mu^2) - mu^2 = sigma^2
$]
== 依概率收敛
#example[
  试证:$X_n stretch(->)^P X <==> n-> infinity text("时，") bb(E)((|X_n-X|)/(1+|X_n-X|))->0$
]
#proof[
  $<==:$令$f(x)=x\/(1+x),x>0$则$f$单调递增，因此
$
{|X_n-X|>epsilon} subset { (|X_n-X|)/(1+|X_n-X|) > epsilon/(1+epsilon)}.
$
由于$(|X_n-X|)/(1+|X_n-X|)$显然非负，使用马尔可夫不等式，有
$
  bb(P)(|X_n-X|>epsilon) <= bb(P)((|X_n-X|)/(1+|X_n-X|) > epsilon/(1+epsilon)) <= (1+epsilon)/epsilon bb(E)((|X_n-X|)/(1+|X_n-X|))
$
由题设知右端趋于0，因此左端趋于0，$X_n stretch(->)^P X$.

$==>:$ 对任意$epsilon>0$，令$A_epsilon ={|X_n-X|>epsilon}$，因为$X_n stretch(->)^P X$，
存在$N$，使得当
$n>=N$时，有$P(A_epsilon)<epsilon$，于是
$
  &bb(E)((|X_n-X|)/(1+|X_n-X|)) \ &= bb(E)(((|X_n-X|)/(1+|X_n-X|)) bb(I)_A_epsilon) + bb(E)(((|X_n-X|)/(1+|X_n-X|)) bb(I)_(A_epsilon^c)) \
  &<= bb(E)(1 dot bb(I)_A_epsilon) +  bb(E)(|X_n-X| bb(I)_(A_epsilon^c)) \
  &<= P(A_epsilon) + epsilon bb(E)(bb(I)_(A_epsilon^c)) \
  &<= P(A_epsilon) + epsilon P(A_epsilon^C)< 2 epsilon
$
]
#remark[
  $==>$的证明体现了分段估计的思想，把全部积分分成两部分，一部分事件本身概率小，另一部分随机变量值小，从而控制期望值.

  精彩，不过技巧性太强，还看不到本质.
]
#example[
  将$n$个编号为$1,2,dots,n$的信封随机分配给$n$个人，每个人只能拿一个信封，
  $
    X_i=cases(1 text("第") i text("个人拿到自己的信封"),0 text("其它情况"))
  $
  $S_n=sum_(i=1)^n X_i$  ,求证:$(S_n-bb(E)(S_n))/n stretch(->)^P 0$.
]
#proof[
  $P(X_i=1)=1/n$, 因此$bb(E)(X_i)=1/n$, Var$(X_i)=(n-1)/n^2$，注意到$X_i$不是相互独立的，（例如$n-1$个人拿到自己的信封时，第$n$个人一定拿到的自己是的信封）,我们无法使用大数定律直接得到结论.
  
  联想到切比雪夫不等式，$ P(abs((S_n-bb(E)(S_n))/n )>= epsilon)<=Var(S_n-bb(E)(S_n))/(n^2 epsilon^2)$
，因此我们需要计算$Var(S_n)$，而$Var(S_n)=sum_(i=1)^n Var(X_i)+2 sum_(1<=i<j<=n)Cov(X_i,X_j)$，所以我们还需要计算协方差.
  $
    Cov(X_i,X_j)=bb(E)(X_i X_j)-bb(E)(X_i) bb(E)(X_j)
  $
  $P(X_i=1,X_j=1)=1/(n(n-1)) ,i != j$，$P(X_i X_j=1)=1/(n(n-1)) ,i!=j$
  $
    bb(E)(X_i X_j)=1/(n(n-1)),quad bb(E)(X_i) bb(E)(X_j)=1/n^2
  $
  因此$Cov(X_i,X_j)=1/(n(n-1)) - 1/n^2=1/(n^2(n-1))$
  $
    Var(S_n)=n (n-1)/n^2 + 2 (n(n-1)/2) 1/(n^2(n-1)) = (n-1)/n + 1/n =1
  $
  因此
$
    P(abs((S_n-bb(E)(S_n))/n) >= epsilon) <= 1/(n^2 epsilon^2)
$
  结论成立.
]
#remark[
  即使我们不知道切比雪夫不等式，也可以用马尔可夫不等式导出它：
$  P(abs((S_n-bb(E)(S_n))/n) >= epsilon)=  P((S_n-bb(E)(S_n))^2/n^2 >= epsilon^2) <= bb(E)((S_n-bb(E)(S_n))^2)/ (n^2 epsilon^2) = Var(S_n)/(n^2 epsilon^2)
$
或者
$
  P(abs((S_n-bb(E)(S_n))/n) >= epsilon)<= bb(E)(abs((S_n-bb(E)(S_n))/n))/epsilon
$
由于$x^2$是上凸函数，使用Jensen不等式，有
$
  bb(E)(abs((S_n-bb(E)(S_n))/n)) <= sqrt(bb(E)((S_n-bb(E)(S_n))^2)/n^2) = sqrt(Var(S_n))/n
$
因此
$
  P(abs((S_n-bb(E)(S_n))/n) >= epsilon)<= sqrt(Var(S_n))/(n epsilon)
$
这比切比雪夫不等式更弱.
]
== 概率方法在其它领域的应用
#example("中心极限定理在数学分析中的应用")[
求证：
$
  lim_(n->infinity) (1+n+n^2/2! + dots + n^n/n!)e^(-n) = 1/2.
$]
#proof[
  设$X_i ~ text("Poisson")(1),$独立同分布，
  $
    Y_n = sum_(i=1)^n X_i ~ text("Poisson")(n)
  $
  $Y_n$的分布率为：
  $
    P(Y_n =k)= e^(-n) n^k/k!, quad k=0,1,2,dots
  $
  $
    P(Y_n <= n)=sum_(k=0)^n e^(-n) n^k/k! = (1+n+n^2/2! + dots + n^n/n!)e^(-n)
  $
  由中心极限定理知：
  $
    (Y_n-n)/sqrt(n) ->^d text("N")(0,1) , quad (n->infinity)
  $
  因此
  $
    1/2=P(Z<=0)=lim_(n->infinity) P((Y_n - n)/sqrt(n) <=0)=lim_(n->infinity) P(Y_n <= n)\
    =lim_(n->infinity) (1+n+n^2/2! + dots + n^n/n!)e^(-n)
  $
  #tip-box[使用其它离散或连续可加分布，可以做更多尝试.]
]
#pagebreak()
#align(center)[
  #v(20em)
  #set text(size: 25pt, weight: "bold")
我终将遗忘梦境中的那些路径，山峦与田野，\
遗忘那些永远不能实现的梦。
]
#align(right)[  #set text(size: 22pt)
  ——马塞尔·普鲁斯特   《追忆似水年华》]
#pagebreak()
= 微分方程
#pagebreak()
#align(center)[
  #v(20em)
  #set text(size:28pt)
  朝搴阰之木兰兮，夕揽洲之宿莽。

  日月忽其不淹兮，春与秋其代序。
]
#align(right)[  #set text(size: 22pt)
  ——屈原   《离骚》]
#pagebreak()
= 抽象代数
#pagebreak()
#align(center)[
  #v(20em)
  #set text(size:28pt, weight: "bold")
 桂棹兮兰桨，击空明兮溯流光.\ 渺渺兮予怀，望美人兮天一方。
 ]
 #align(right)[  #set text(size: 22pt)
 ——苏轼   《赤壁赋》]
#pagebreak()
= 统计学习
#pagebreak()
#align(center)[
  #v(20em)
  #set text(size:28pt, weight: "bold")
悟已往之不谏，知来者之可追。
 ]
 #align(right)[  #set text(size: 22pt)
 ——陶渊明   《归去来兮辞》]


#pagebreak()
#align(center)[
  #v(20em)
  #set text(size:28pt, weight: "bold")
 心无挂碍，无有恐怖，\ 远离颠倒梦想，究竟涅磐。 
 ]
 #align(right)[  #set text(size: 22pt)
 ——   《心经》]

#pagebreak()
#align(center)[
  #v(20em)
  #set text(size:25pt, weight: "bold")
蜀道之难，难于上青天，使人听此凋朱颜！

连峰去天不盈尺，枯松倒挂倚绝壁。

飞湍瀑流争喧豗，砯崖转石万壑雷。]
#align(right)[  #set text(size: 22pt)
  ——李白   《蜀道难》]
#pagebreak()
#align(center)[
#v(20em)
#set text(size:25pt, weight: "bold")
发愤忘食，乐以忘忧，不知老之将至云尔。]
#align(right)[  #set text(size: 22pt)
  ——孔子   《论语》]
