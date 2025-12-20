#import "@preview/ori:0.2.3": *
#set heading(numbering: numbly("{1:一}、", default: "1.1  "))
#show: ori.with(
  title: "数学分析中的Laplace方法",
  author: "上海师范大学数理学院",
  maketitle: true,
  first-line-indent: auto,
  outline-depth: 3,
)
#theorem-box[
Laplace方法基本思想：估计$I(x)=integral_a^b f(t) e^(phi.alt(t) x) dif t$时，如果$phi.alt$取得最大值$phi.alt(c),f(c)!=0$,那么$c$的邻域这部分积分贡献了$I(x)$在$x	-> +oo$时的主部。
基本方法：取$(c-delta,c+delta)$,先利用单调性、有界性等性质估计出积分$I_1(x)=integral_(c-delta)^(c+delta) f(t) e^(phi.alt x) dif t$的渐近表达式，然后估计$integral_a^(c-delta)+integral_(c+delta)^b$,结果往往为0，再对$integral_(c-delta)^(c+delta)$进行估阶（如Taylor展开，分部积分等）。
]
#definition-box[
#example[1、设$x_n=sum_(k=1)^n e^(k^2)/k,y_n=integral_0^n e^(x^2) dif x$，问
$
  lim_(n -> +oo) x_n/y_n=?
$
]
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
#definition-box[
  #example[
    2、计算 $
           lim_(x -> +oo) x integral_0^10 (1+t)^(-1) e^(-x t) dif t
         $
  ]
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
#definition-box[
  #example[
    3、计算
    $
      lim_(x -> +oo) x integral _0 ^(pi/2) e^(-x tan t) dif t
    $
  ]
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
#caution-box[
以上的例子都显示含有e,对于没有e的情况，我们可以人为构造e来使用Laplace方法。
]
#definition-box[
  #example[
    4、求证：对于非负整数n,考虑积分
    $
      I_n=integral_0^(pi/2) sin^(2n)x dif x
    $，则
    $
      lim_(n ->+oo) sqrt(n) I_n= sqrt(pi)/2
    $
  ]
]
证明：
另一方面，利用Taylor展开，有
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
令$epsilon -> 0$，可得所需结论.
#corollary-box[
  根据点火公式，有
$
  I_n=integral_0^(pi/2) sin^(2n)x dif x= pi/2 dot (2n-1)!!/(2n)!!
$
  因此我们也得到了斯特林公式的一个推论：
$
  (2n)!!/(2n-1)!! ~ sqrt(pi n) quad (n -> +oo)
$
]
