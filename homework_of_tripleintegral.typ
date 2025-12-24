求由两个曲面$z=x^2+y^2$与$z=x+y$所围成的立体体积。\
首先，求两个曲面的交线：
$
  x^2+y^2=x+y\
  x^2-x+y^2-y=0\
  (x-1/2)^2+(y-1/2)^2=1/2
$
这是一个以$(1/2,1/2)$为圆心，$sqrt(2)/2$为半径的圆。
令$u=x-1/2$，$v=y-1/2$。
那么积分区域为$D={(u,v)|u^2+v^2 <= 1/2}$。
$
  V&=integral.double_D<= (x+y)-(x^2+y^2) dif x dif y\
  &=integral.double_D<= (u+1/2+v+1/2)-((u+1/2)^2+(v+1/2)^2) dif u dif v\
  &=integral.double_D<= -(u^2+v^2) +1/2 quad dif u dif v=integral.double_D<= (-r^2 +1/2) quad r dif r dif theta\
  &=integral_0^(2pi) integral_0^(sqrt(2)/2) (-r^2 +1/2) quad r dif r dif theta=integral_0^(2pi) [(-1/4 r^4 +1/4 r^2)]_0^(sqrt(2)/2) dif theta\
  &=integral_0^(2pi) (1/16) dif theta= (1/16)(2pi-0)=pi/8
$
求曲面$z^2=x^2 /4 +y^2 /9$和$2 z=x^2 /4 + y^2/9$所围成的立体V体积。
联立方程：
$
  z^2=2z  => z(z-2)=0  => z=0 or z=2
$
所以积分区域为$D={(x,y)| x^2 /4 +y^2 /9 <=4}$。
$
  V=integral.double_D<= sqrt(x^2 /4 +y^2 /9) - 1/2(x^2 /4 +y^2 /9) dif x dif y\
$
令$x=2r cos(theta)$,$y=3r sin(theta)$，则$dif x dif y=6 r dif r dif theta$。
积分区域变为$D={(r,theta)|0 <= r <=1,0 <= theta <= 2pi}$。
$
  V=integral_0^(2pi) integral_0^2 (r - 1/2 r^2) 6 r dif r dif theta=integral_0^(2 pi) integral_0^2 [6 r^2-3r^3]dif r dif theta\
  =integral_0^(2 pi) [2 r^3 - 3/4 r^4]_0^2 dif theta=integral_0^(2 pi) (16 - 12) dif theta=integral_0^(2 pi) (4) dif theta= (4)(2 pi - 0)=8pi
$
== 将下列积分化成累次积分
$
  integral_0^2 dif x integral_(1-x)^(2-x) f(x,y) dif y
$
令$u=x+y$，$v=x-y$，
则$J=|partial(x,y)/partial(u,v)|=1/2$\

$cases((u+v)/2 in [0,2],(u-v)/2 in [1-(u+v)/2,2-(u+v)/2])$$=>u in [1,2],v in [-u,4-u]$
$
  integral_1^2 dif u integral_(-u)^(4-u) f((u+v)/2,(u-v)/2) (1/2) dif v
$

// (2)
 $ integral.double_D  f(x,y) dif x dif y $ 令 $ x = u cos^4 v, y = u sin^4 v $，则
 $ J = |[mat(cos^4 v, -4u cos^3 v sin v; sin^4 v, 4u sin^3 v cos v)] = 4u sin^3 v cos^3 v $ 积分区域边界：
 $ sqrt(x) + sqrt(y) = sqrt(u)(cos^2 v + sin^2 v) <= sqrt(a) => 0 <= u <= a $  $ x >= 0, y >= 0 => 0 <= v <= pi/2 $  $ = integral_0^(pi/2) dif v integral_0^a f(u cos^4 v, u sin^4 v) dot 4u sin^3 v cos^3 v dif u $ 
// (3)
 $ integral.double_D  f(x,y) dif x dif y $ 令 $ u = x+y, v = y/(x+y) $，即 $ y = u v, x = u - y = u(1-v) $.
则 $ J = |[mat(1-v, -u; v, u)] = u $ 积分区域边界：
 $ 0 <= x+y <= a => 0 <= u <= a $  $ x >= 0, y >= 0 => cases(v <= 1, v >= 0) => 0 <= v <= 1 $  $ = integral_0^1 dif v integral_0^a f(u(1-v), u v) dot u dif u $ 

 
$
  integral.double_D sin(sqrt(x^2 + y^2)) dif x dif y
$
令 $x = r cos(theta)$，$y = r sin(theta)$。
则 $J = r$，积分限为 $pi <= r <= 2 pi$，$0<= theta <=2 pi$。
$
  &= integral_0^(2 pi) integral_pi^(2 pi) (sin r) dot r dif r dif theta \
  &= integral_0^(2 pi) dif theta dot integral_pi^(2 pi) r sin r dif r = [theta]_0^(2 pi) dot ([-r cos r]_pi^(2 pi) + integral_pi^(2 pi) cos r dif r) \
  &= 2 pi dot (-2 pi cos 2 pi + pi cos pi + [sin r]_pi^(2 pi))=-6 pi^2 
$

// (2)
$
  integral.double_D (x + y) dif x dif y
$
积分区域 $D$ 由 $x^2 + y^2 <= x+y$  围成。
令 $x = r cos(theta)$，$y = r sin(theta)$。
边界方程 $r^2 <=r(cos(theta)+sin(theta)) => r <= cos(theta)+sin(theta)$。
积分限为 $-pi/4 <= theta <=(3 \pi) / 4$，$0 <= r <= cos(theta)+sin(theta)$。
$
&= integral_(-pi/4)^((3 pi) / 4) integral_0^(cos(theta)+sin(theta)) (r cos(theta) + r sin(theta)) dot r dif r dif theta \
  &= integral_(-pi/4)^((3 pi) / 4) (cos(theta) + sin(theta)) [r^3 / 3]_0^(cos(theta)+sin(theta)) dif theta \
  &= integral_(-pi/4)^((3 pi) / 4) (cos(theta) + sin(theta))^4 / 3 dif theta \
  &= (1/3) integral_(-pi/4)^((3 pi) / 4) (1 +  sin(2 theta))^2 dif theta = (1/3) integral_(-pi/4)^((3 pi) / 4) (1 + 2 sin(2 theta) +  sin^2(2 theta)) dif theta \
  &= (1/3) [theta - cos(2 theta) + (2 theta - sin(4 theta)) / 4]_(-pi/4)^((3 pi) / 4) = (1/3) (pi -0+(2pi -0)/4)=pi/2
$
== $D={(x,y)|x^2+y^2<=a^2}$ 求$integral.double_D |x y| dif x dif y$
$

  &=4 integral_0^(pi/2)dif theta integral_0^a r cos theta dot r sin theta dot r dif r \
  &=4 integral_0^(pi/2) cos theta sin theta dif theta integral_0^a r^3 dif r \
  &=4 [1 / 2] dot [r^4 / 4]_0^a = a^4/2
$
== $D={(x,y)|x^2+y^2<=R^2}$求$integral.double_D f'(x^2+y^2) dif x dif y$
$
  integral.double_D f'(x^2+y^2) dif x dif y &= integral_0^(2 pi) dif theta integral_0^R f'(r^2) dot r dif r \
  &= 2 pi [1/2 f(r^2)]_0^R = pi (f(R^2) - f(0))
$