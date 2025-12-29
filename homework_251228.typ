= 1
设总体$X$服从二项分布$b(m,p)$，其中$m,p$未知，$x_1,dots,x_n$为来自总体$X$的样本。
求参数$m$和$p$的矩估计量。

$
E[X]=m p
$
$
  E[X^2]=text("Var(X)")+(E[X])^2=m p (1-p)+(m p)^2=m p (1-p+m p)=m p +m(m-1)p^2
$
样本一二阶矩分别为：
$
mu_1=1/n sum_(i=1)^n x_i\
mu_2=1/n sum_(i=1)^n x_i^2
$
联立
$
  cases(1/n sum_(i=1)^n x_i=m p,1/n sum_(i=1)^n x_i^2=m p +m(m-1)p^2) 
$
解得$p=mu_1/m$，带入第二个方程得
$
  mu_2=mu_1+mu_1^2-1/m mu_1^2\
  m=(mu_1^2)/(mu_1-mu_2+mu_1^2)
$
$
  p=(mu_1-mu_2+mu_1^2)/mu_1
$
$m$和$p$的矩估计量分别为
$
  hat(m)=[( (1/n sum_(i=1)^n x_i)^2 )/( (1/n sum_(i=1)^n x_i)-(1/n sum_(i=1)^n x_i^2)+( (1/n sum_(i=1)^n x_i)^2 ) )]， quad text("中括号表示取整")\
  hat(p)=( (1/n sum_(i=1)^n x_i)-(1/n sum_(i=1)^n x_i^2)+( (1/n sum_(i=1)^n x_i)^2 ) )/(1/n sum_(i=1)^n x_i)
$
= 2
螺纹面的方程为：
$
  cases(x=r cos phi,y=r sin phi,z=b phi),quad (0<=r<=a,0<=phi<=2 pi)
$
求曲面面积
$
  bold(r)=(r cos phi, r sin phi, b phi)\
  bold(r)_r=(cos phi,sin phi,0)\
  bold(r)_phi=(-r sin phi,r cos phi,b)\
  bold(r)_r times bold(r)_phi=mat(delim:"|",bold(i),bold(j),bold(k);cos phi,sin phi,0;-r sin phi,r cos phi,b)= (b cos phi, -b sin phi, r)\
  ||bold(r)_r times bold(r)_phi||=sqrt(b^2+r^2)\
  S=integral.double_(D)sqrt(b^2+r^2)dif r dif phi, quad D={(r,phi)|0<=r<=a,0<=phi<=2 pi}\
$
$
    S=integral_0^(2 pi) dif phi integral_0^a sqrt(b^2+r^2) dif r=2 pi [integral_0^a sqrt(b^2+r^2) dif r]\
    =2 pi[r/2 sqrt(r^2+b^2)+b^2/2 ln(r+sqrt(r^2+b^2))]^a_0\
    =2 pi[a/2 sqrt(a^2+b^2)+b^2/2 ln(a+sqrt(a^2+b^2))-b^2/2 ln b]\
    =pi a sqrt(a^2+b^2)+pi b^2 ln((a+sqrt(a^2+b^2))/b)
$
= 3
计算$a z=x y$包含在圆柱面$x^2+y^2=a^2$内部分的面积

$
  S=integral.double_(D)sqrt(1+z_x^2+z_y^2) dif x dif y, quad D={(x,y)|x^2+y^2<=a^2}\
  =integral.double_(D)sqrt(1+(y/a)^2+(x/a)^2) dif x dif y\
  =integral.double_(D)sqrt(1/(a^2)(x^2+y^2)+1) dif x dif y\
  =1/a integral.double_(D)sqrt( r^2+a^2) r dif r dif theta\
  =1/a integral_0^(2 pi) dif theta integral_0^a sqrt(r^2+a^2) r dif r\
  =2 pi/a [1/3 (r^2+a^2)^(3/2)]^a_0\
  =2 pi/a [1/3 (2 a^2)^(3/2)-1/3 a^3]\
  =2 pi/a [ (2^(3/2)/3) a^3 -1/3 a^3 ]\
  =2 pi a^2/3 (2^(3/2)-1)
  
$