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
    title: [$section$5 三重积分],
    institution: [上海师范大学数理学院],
  ),
)

#title-slide()

类似于第一型曲线积分，求一个空间立体 $V$ 的质量 $M$ 就可导出三重积分。设密度函数为 $f(x, y, z)$，为了求 $V$ 的质量，我们把 $V$ 分割成 $n$ 个小块 $V_1, V_2, dots.c, V_n$，在每个小块 $V_i$ 上任取一点 $(xi_i, eta_i, zeta_i)$，则
#theorem-box[
$
M = lim_(bar.v.double T bar.v.double -> 0) sum_(i = 1)^n f(xi_i, eta_i, zeta_i) Delta V_i,
$
其中 $Delta V_i$ 为小块 $V_i$ 的体积，$bar.v.double T bar.v.double = max_(1 <= i <= n) {V_i " 的直径"}$。
]
#pagebreak()
#corollary-box[
设 $f(x, y, z)$ 是定义在三维空间可求体积的有界闭区域 $V$ 上的有界函数。现用若干光滑曲面所组成的曲面网 $T$ 来分割 $V$，它把 $V$ 分成 $n$ 个小区域 $V_1, V_2, dots.c, V_n$，记 $V_i$ 的体积为 $Delta V_i (i = 1, 2, dots.c, n)$，$bar.v.double T bar.v.double = max_(1 <= i <= n) {V_i " 的直径"}$。在每个 $V_i$ 中任取一点 $(xi_i, eta_i, zeta_i)$，作积分和
$
sum_(i = 1)^n f(xi_i, eta_i, zeta_i) Delta V_i .
$
]
#definition-box[设 $f(x, y, z)$ 为定义在三维空间可求体积的有界闭区域 $V$ 上的函数，$J$ 是一个确定的数。若对任给的正数 $epsilon$，总存在某一正数 $delta$，使得对于 $V$ 的任何分割 $T$，只要 $bar.v.double T bar.v.double < delta$，属于分割 $T$ 的所有积分和都有
$lr(|sum_(i = 1)^n f(xi_i, eta_i, zeta_i) Delta V_i - J|) < epsilon$
则称 $f(x, y, z)$ 在 $V$ 上可积，数 $J$ 称为函数 $f(x, y, z)$ 在 $V$ 上的三重积分，记作
$
J = integral.triple_V f(x, y, z) d V quad "或" quad J = integral.triple_V f(x, y, z) d x d y d z
$
]<definition>
#pagebreak()
#theorem-box[
其中 $f(x, y, z)$ 称为被积函数，$x$、$y$、$z$ 称为积分变量，$V$ 称为积分区域。

当 $f(x, y, z) equiv 1$ 时，$integral.triple_V d V$ 在几何上表示 $V$ 的体积。
]
三重积分具有与二重积分相应的可积条件和有关性质（参见\S1），这里不一一详述了。例如，类似于二重积分，有：

    - 有界闭区域 $V$ 上的连续函数必可积；\
    - 如果有界闭区域 $V$ 上的有界函数 $f(x, y, z)$ 的间断点集中在有限多个零体积（可类似于零面积那样来定义）的曲面上，则 $f(x, y, z)$ 在 $V$ 上必可积。




若函数 $f(x, y, z)$ 在长方体 $V = [a, b] times [c, d] times [e, h]$ 上的三重积分存在，且对任意 $(x, y) in D = [a, b] times [c, d]$，$g(x, y) = integral_e^h f(x, y, z) d z$ 存在，积分 $integral.double_D g(x, y) d x d y$ 也存在，则
$
integral.triple_V f(x, y, z) d x d y d z = integral.double_D d x d y integral_e^h f(x, y, z) d z .
$


 用平行于坐标平面的平面作分割 $T$，它把 $V$ 分成有限多个小长方体
$
V_(i j k) = [x_(i - 1), x_i ] times [y_(j - 1), y_j ] times [z_(k - 1), z_k ] .
$
设 $M_(i j k)$、$m_(i j k)$ 分别是 $f(x, y, z)$ 在 $V_(i j k)$ 上的上确界和下确界。对任意 $(xi_i, eta_j) in [x_(i - 1), x_i ] times [y_(j - 1), y_j ]$，有
$
m_(i j k) Delta z_k <= integral_(z_(k - 1))^(z_k) f(xi_i, eta_j, z) d z <= M_(i j k) Delta z_k .
$
现按下标 $k$ 相加，有
$
sum_k integral_(z_(k - 1))^(z_k) f(xi_i, eta_j, z) d z = integral_e^h f(xi_i, eta_j, z) d z = g(xi_i, eta_j)
$
以及
$
sum_(i, j, k) m_(i j k) Delta x_i Delta y_j Delta z_k <= sum_(i, j) g(xi_i, eta_j) Delta x_i Delta y_j <= sum_(i, j, k) M_(i j k) Delta x_i Delta y_j Delta z_k .
$
上述不等式两边是分割 $T$ 的下和与上和。由 $f(x, y, z)$ 在 $V$ 上可积，当 $bar.v.double T bar.v.double -> 0$ 时，下和与上和具有相同的极限，所以由上式得 $g(x, y)$ 关于 $D$ 对应 $T$ 的直线网格分割的下和与上和具有相同的极限。由定理21.4得 $g(x, y)$ 在 $D$ 上可积，且
$
integral.double_D g(x, y) d x d y = integral.triple_V f(x, y, z) d x d y d z .
$


若 $V = {(x, y, z) divides(x, y) in D, z_1 (x, y) <= z <= z_2 (x, y)}$，其中 $D$ 为 $V$ 在 $O x y$ 平面上的投影，$z_1 (x, y)$、$z_2 (x, y)$ 是 $D$ 上的连续函数，函数 $f(x, y, z)$ 在 $V$ 上的三重积分存在，且对任意 $(x, y) in D$，
$
G(x, y) = integral_(z_1 (x, y))^(z_2 (x, y)) f(x, y, z) d z
$
亦存在，则积分 $integral.double_D G(x, y) d x d y$ 存在，且
$
integral.triple_V f(x, y, z) d x d y d z = integral.double_D G(x, y) d x d y = integral.double_D d x d y integral_(z_1 (x, y))^(z_2 (x, y)) f(x, y, z) d z 
$
（见图21-31）


 设 $V_0 = [a, b] times [c, d] times [e, h]$ 是包含 $V$ 的长方体，定义
$
F(x, y, z) = cases(f(x comma y comma z) comma &(x comma y comma z) in V comma, 0 comma &(x comma y comma z) in V_0 without V .)
$
对 $F(x, y, z)$ 用定理21.15，则有
$
integral.triple_V f(x, y, z) d x d y d z &= integral.triple_(V_0) F(x, y, z) d x d y d z \ &= integral.double_([a, b] times [c, d]) d x d y integral_e^h F(x, y, z) d z \ &= integral.double_D d x d y integral_(z_1 (x, y))^(z_2 (x, y)) f(x, y, z) d z .
$


计算 $integral.triple_V (d x d y d z)/(x^2 + y^2)$，其中 $V$ 为由平面 $x = 1$、$x = 2$、$z = 0$、$y = x$ 与 $z = y$ 所围区域（见图21-32）。


 设 $V$ 在 $O x y$ 平面上投影为 $D$，则 $V = {(x, y, z) divides z_1 (x, y) <= z <= z_2 (x, y),(x, y) in D}$，其中 $D = {(x, y) divides 0 <= y <= x, 1 <= x <= 2}$ 是 $x$ 型区域，$z_1 (x, y) = 0$，$z_2 (x, y) = y$。于是
$
integral.triple_V (d x d y d z)/(x^2 + y^2) &= integral.double_D d x d y integral_0^y (d z)/(x^2 + y^2) \ &= integral.double_D y/(x^2 + y^2) d x d y = integral_1^2 d x integral_0^x (y d y)/(x^2 + y^2) \ &= lr(integral_1^2 1/2 ln(x^2 + y^2)|)_0^x d x = integral_1^2 1/2 ln 2 d x \ &= 1/2 ln 2 .
$


计算 $integral.triple_V (x^2 + y^2 + z) d x d y d z$，其中 $V$ 是由 $cases(z = y, x = 0)$ 绕 $z$ 轴旋转一周而成的曲面与 $z = 1$ 所围的区域。


 旋转面方程为 $z = sqrt(x^2 + y^2)$。$V$ 在 $O x y$ 平面上投影为 $D = {(x, y) divides x^2 + y^2 <= 1}$，$z_1 (x, y) = sqrt(x^2 + y^2)$，$z_2 (x, y) = 1$。于是
$
integral.triple_V (x^2 + y^2 + z) d x d y d z &= integral.double_D d x d y integral_sqrt(x^2 + y^2)^1 (x^2 + y^2 + z) d z \ &= integral.double_D (x^2 + y^2)(1 - sqrt(x^2 + y^2)) d x d y + 1/2 integral.double_D [1 -(x^2 + y^2)] d x d y \ &= integral_0^(2 pi) d theta integral_0^1 r^2 (1 - r) r d r + 1/2 integral_0^(2 pi) d theta integral_0^1 (1 - r^2) r d r \ &= pi/10 + pi/4 = (7 pi)/20 .
$

类似分析可得到以下定理和推论：


若函数 $f(x, y, z)$ 在长方体 $V = [a, b] times [c, d] times [e, h]$ 上的三重积分存在，且对任何 $z in [e, h]$，二重积分
$
I(z) = integral.double_D f(x, y, z) d x d y
$
存在（其中 $D = [a, b] times [c, d]$），则积分
$
integral_e^h d z integral.double_D f(x, y, z) d x d y
$
也存在，且
$
integral.triple_V f(x, y, z) d x d y d z = integral_e^h d z integral.double_D f(x, y, z) d x d y .
$



若 $V subset [a, b] times [c, d] times [e, h]$，函数 $f(x, y, z)$ 在 $V$ 上的三重积分存在，且对任意固定的 $z in [e, h]$，积分 $phi(z) = integral.double_(D_z) f(x, y, z) d x d y$ 存在（其中 $D_z = {(x, y) divides(x, y, z) in V}$），则 $integral_e^h phi(z) d z$ 存在，且
$
integral.triple_V f(x, y, z) d x d y d z = integral_e^h phi(z) d z = integral_e^h d z integral.double_(D_z) f(x, y, z) d x d y .
$
（见图21-33）


 类似于对二重积分转化为累次积分的几何解释，对定理21.15和定理21.16及其推论所给出的方法也可以从几何上做出解释。在定理21.15及其推论中，把积分区域视为柱体的一部分（见图21-31），投影（柱体底面）区域为 $D$，柱体上、下曲面分别为 $z = z_2 (x, y)$、$z = z_1 (x, y)$（$(x, y) in D$），三重积分相应转化为先对 $z$ 再对 $x, y$ 的累次积分形式（先一后二）。在定理21.16及其推论中，把积分区域视为介于两平行平面 $z = e$、$z = h$ 之间的立体（见图21-33），用垂直于 $z$ 轴的平面截积分区域所得截面为 $D_z$，此时三重积分相应转化为先对 $x, y$ 变量在平面区域 $D_z$ 上的含参二重积分，再对 $z in [e, h]$ 求定积分的形式（先二后一）。具体采用哪一种形式可根据实际问题中积分区域与被积函数的特性加以选择。


求 $I = integral.triple_V ((x^2)/(a^2) + (y^2)/(b^2) + (z^2)/(c^2)) d x d y d z$，其中 $V$ 是椭球体 $(x^2)/(a^2) + (y^2)/(b^2) + (z^2)/(c^2) <= 1$。


 由于
$
I = integral.triple_V (x^2)/(a^2) d x d y d z + integral.triple_V (y^2)/(b^2) d x d y d z + integral.triple_V (z^2)/(c^2) d x d y d z,
$
其中 $integral.triple_V (x^2)/(a^2) d x d y d z = integral_(-a)^a (x^2)/(a^2) d x integral.double_(V_x) d y d z$，这里 $V_x$ 表示截面
$
(y^2)/(b^2) + (z^2)/(c^2) <= 1 - (x^2)/(a^2) quad "或" quad (y^2)/(b^2 (1 - (x^2)/(a^2))) + (z^2)/(c^2 (1 - (x^2)/(a^2))) <= 1 .
$
它的面积为
$
pi(b sqrt(1 - (x^2)/(a^2)))(c sqrt(1 - (x^2)/(a^2))) = pi b c(1 - (x^2)/(a^2)) .
$
于是
$
integral.triple_V (x^2)/(a^2) d x d y d z = integral_(-a)^a (pi b c)/(a^2) x^2 (1 - (x^2)/(a^2)) d x = 4/15 pi a b c .
$
同理可得
$
integral.triple_V (y^2)/(b^2) d x d y d z = 4/15 pi a b c, quad integral.triple_V (z^2)/(c^2) d x d y d z = 4/15 pi a b c .
$
所以
$
I = 3(4/15 pi a b c) = 4/5 pi a b c .
$

下面用定理21.16的推论计算例1：

例1中的区域 $V = {(x, y, z) divides 1 <= x <= 2,(y, z) in D_x }$，其中 $D_x = {(y, z) divides 0 <= y <= x, 0 <= z <= y}$（见图21-34）。于是
$
integral.triple_V (d x d y d z)/(x^2 + y^2) &= integral_1^2 d x integral.double_(D_x) (d y d z)/(x^2 + y^2) \ &= integral_1^2 d x integral_0^x d y integral_0^y (d z)/(x^2 + y^2) \ &= integral_1^2 d x integral_0^x (y d y)/(x^2 + y^2) = 1/2 ln 2 .
$


和二重积分一样，某些类型的三重积分作适当的变量变换后能使计算方便。

设变换 $T : x = x(u, v, w)$，$y = y(u, v, w)$，$z = z(u, v, w)$，把 $u v w$ 空间中的区域 $V'$ 一对一地映成 $x y z$ 空间中的区域 $V$，并设函数 $x(u, v, w)$、$y(u, v, w)$、$z(u, v, w)$ 及它们的一阶偏导数在 $V'$ 内连续且函数行列式
$
J(u, v, w) = mat(delim: "|", (diff x)/(diff u),(diff x)/(diff v),(diff x)/(diff w);(diff y)/(diff u),(diff y)/(diff v),(diff y)/(diff w);(diff z)/(diff u),(diff z)/(diff v),(diff z)/(diff w)) != 0, quad(u, v, w) in V',
$
于是与二重积分换元法一样，可以证明（用本章\S9中证明二重积分类似的方法）成立下面的三重积分换元公式：
$
& integral.triple_V f(x, y, z) d x d y d z \ &= integral.triple_(V') f(x(u, v, w), y(u, v, w), z(u, v, w))|J(u, v, w)|d u d v d w, 
$
其中 $f(x, y, z)$ 在 $V$ 上可积。

下面介绍几个常用的变换公式：


$
T : cases(x = r cos theta comma & 0 <= r < + infinity comma, y = r sin theta comma & 0 <= theta <= 2 pi comma, z = z comma & - infinity < z < + infinity .)
$
由于变换 $T$ 的函数行列式
$
J(r, theta, z) = mat(delim: "|", cos theta, - r sin theta, 0; sin theta, r cos theta, 0; 0, 0, 1) = r,
$
按(4)式，三重积分的柱面坐标换元公式为
$
integral.triple_V f(x, y, z) d x d y d z = integral.triple_(V') f(r cos theta, r sin theta, z) r d r d theta d z
$
这里 $V'$ 为 $V$ 在柱面坐标变换下的原象。

与极坐标变换一样，柱面坐标变换并非是一对一的，并且当 $r = 0$ 时，$J(r, theta, z) = 0$，但我们仍可证明(5)式成立。

在柱面坐标系中，用 $r = "常数"$、$theta = "常数"$、$z = "常数"$ 的平面分割 $V'$ 时，变换后在 $x y z$ 直角坐标系中，$r = "常数"$ 是以 $z$ 轴为中心轴的圆柱面，$theta = "常数"$ 是过 $z$ 轴的半平面，$z = "常数"$ 是垂直于 $z$ 轴的平面（图21-35）。

用柱面坐标计算三重积分，通常是找出 $V$ 在 $O x y$ 平面上的投影区域 $D$，即当
$
V = {(x, y, z) divides z_1 (x, y) <= z <= z_2 (x, y),(x, y) in D}
$
时，
$
integral.triple_V f(x, y, z) d x d y d z = integral.double_D d x d y integral_(z_1 (x, y))^(z_2 (x, y)) f(x, y, z) d z,
$
其中二重积分部分应用极坐标计算。


计算 $integral.triple_V (x^2 + y^2) d x d y d z$，其中 $V$ 是由曲面 $2(x^2 + y^2) = z$ 与 $z = 4$ 为界面的区域（图21-36）。


 $V$ 在 $O x y$ 平面上的投影区域 $D$ 为 $x^2 + y^2 <= 2$。按柱坐标变换，区域 $V'$ 可表为
$
V' = {(r, theta, z) divides 2 r^2 <= z <= 4, 0 <= r <= sqrt(2), 0 <= theta <= 2 pi} .
$
所以由公式(5)，有
$
integral.triple_V (x^2 + y^2) d x d y d z &= integral.triple_(V') r^3 d r d theta d z \ &= integral_0^(2 pi) d theta integral_0^sqrt(2) d r integral_(2 r^2)^4 r^3 d z = (8 pi)/3 .
$


$
T : cases(x = r sin phi cos theta comma & 0 <= r < + infinity comma, y = r sin phi sin theta comma & 0 <= phi <= pi comma, z = r cos phi comma & 0 <= theta <= 2 pi .)
$
由于
$
J(r, phi, theta) &= mat(delim: "|", sin phi cos theta, r cos phi cos theta, - r sin phi sin theta; sin phi sin theta, r cos phi sin theta, r sin phi cos theta; cos phi, - r sin phi, 0) \ &= r^2 sin phi,
$
当 $phi$ 在 $[0, pi]$ 上取值时，$sin phi >= 0$，所以在球坐标变换下，按公式(4)，三重积分的球坐标换元公式为
$
& integral.triple_V f(x, y, z) d x d y d z \ &= integral.triple_(V') f(r sin phi cos theta, r sin phi sin theta, r cos phi) r^2 sin phi d r d phi d theta, 
$
这里 $V'$ 为 $V$ 在球坐标变换 $T$ 下的原象。

类似地，球坐标变换并不是一对一的，并且当 $r = 0$ 或 $phi = 0$ 或 $pi$ 时，$J(r, phi, theta) = 0$，但我们仍然可以证明(6)式成立。

在球坐标系中，用 $r = "常数"$、$phi = "常数"$、$theta = "常数"$ 的平面分割 $V'$ 时，变换后在 $x y z$ 直角坐标系中，$r = "常数"$ 是以原点为心的球面，$phi = "常数"$ 是以原点为顶点、$z$ 轴为中心轴的半圆锥面，$theta = "常数"$ 是过 $z$ 轴的半平面（图21-37）。

在球坐标系下，当区域 $V'$ 为集合
$
V' = {(r, phi, theta) divides r_1 (phi, theta) <= r <= r_2 (phi, theta), phi_1 (theta) <= phi <= phi_2 (theta), theta_1 <= theta <= theta_2 }
$
时，(6)式可化为累次积分
$
& integral.triple_V f(x, y, z) d x d y d z \ &= integral_(theta_1)^(theta_2) d theta integral_(phi_1 (theta))^(phi_2 (theta)) d phi integral_(r_1 (phi, theta))^(r_2 (phi, theta)) f(r sin phi cos theta, r sin phi sin theta, r cos phi) r^2 sin phi d r .
$


求由圆锥体 $z >= sqrt(x^2 + y^2) cot beta$ 和球体 $x^2 + y^2 +(z - a)^2 <= a^2$ 所确定的立体体积（图21-38，其中 $beta in(0, pi/2)$ 和 $a > 0$ 为常数）。


 在球坐标变换下，球面方程 $x^2 + y^2 +(z - a)^2 = a^2$ 可表示成 $r = 2 a cos phi$，锥面方程 $z = sqrt(x^2 + y^2) cot beta$ 可表示成 $phi = beta$。因此
$
V' = {(r, phi, theta) divides 0 <= r <= 2 a cos phi, 0 <= phi <= beta, 0 <= theta <= 2 pi} .
$
由公式(7)求得 $V$ 的体积为
$
integral.triple_V d V = integral_0^(2 pi) d theta integral_0^beta d phi integral_0^(2 a cos phi) r^2 sin phi d r = 4/3 pi a^3 (1 - cos^4 beta) .
$

除上述介绍的两种变换外，下面我们再举一个例子，进一步说明如何根据被积函数或积分区域的特点来选择其他不同的变换。


求 $I = integral.triple_V z d x d y d z$，其中 $V$ 为由 $(x^2)/(a^2) + (y^2)/(b^2) + (z^2)/(c^2) <= 1$ 与 $z >= 0$ 所交区域。


 作广义球坐标变换
$
T : cases(x = a r sin phi cos theta comma, y = b r sin phi sin theta comma, z = c r cos phi comma)
$
于是 $J = a b c r^2 sin phi$。在上述广义球坐标变换下，$V$ 的原象为
$
V' = {(r, phi, theta) divides 0 <= r <= 1, 0 <= phi <= pi/2, 0 <= theta <= 2 pi} .
$
由公式(7)，有
$
integral.triple_V z d x d y d z &= integral.triple_(V') a b c^2 r^3 sin phi cos phi d r d phi d theta \ &= integral_0^(2 pi) d theta integral_0^(pi/2) d phi integral_0^1 a b c^2 r^3 sin phi cos phi d r \ &= (pi a b c^2)/2 integral_0^(pi/2) sin phi cos phi d phi = (pi a b c^2)/4 .
$
