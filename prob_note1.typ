#import "@preview/ori:0.2.3": *
#set heading(numbering: numbly("{1:一}、", default: "1.1  "))
#theorem-box[强大数定律：

  设$X_1,X_2,dots$为一独立同分布的随机变量序列，公共均值为$mu$.则
$
  P{lim_(n->infinity) (X_1+X_2+...+X_n)/n=mu}=1
$
记为
$
  (X_1+X_2+...+X_n)/n stretch(->)^(text(a.s.) )mu quad (n->infinity) 
$
]
设$X_1,X_2,dots$为一独立同分布的随机变量序列，定义
$
  F_n (x)=1/n sum_(i=1)^n I{X_i <= x}
$
为经验分布函数.
我们有：
$
  bb(E)(I(X_1 <= x))=1 dot P(X_1 <= x)+ 0 dot P(X_1 > x)=F(x)
$
由强大数定律知：
$
  P{lim_(n->infinity) F_n (x)=F(x)}=1
$
#definition-box[
$X_1,dots,X_n stretch(~)^text("iid") N(mu,sigma^2)$,求证
$
  ((n-1) s^2)/sigma^2=1/sigma^2 sum_(i=1)^n (X_i-overline(X))^2~chi^2(n-1)
$
]
令
$
  Z_i=(X_i-mu)/sigma => Z_i ~N(0,1),text("独立")
$
那么
$
overline(Z)=1/n sum_(i=1)^n Z_i
$
且
$
  X_i-overline(X)=(X_i-mu)-(overline(X)-mu)=sigma (Z_i-overline(Z))
$
所以
$
  (X_i-overline(X)^2)=sigma^2(Z_i-overline(Z))^2\
  sum_(i=1)^n (X_i-overline(X))^2=sigma^2 sum_(i=1)^n (Z_i-overline(Z))^2\
  => 1/sigma^2 sum_(i=1)^n (X_i-overline(X))^2 =sum_(i=1)^n (Z_i-overline(Z))^2
$
又因为
$
  sum_(i=1)^n (Z_i-overline(Z))^2= sum_(i=1)^n Z_i^2 - n overline(Z)^2
$
我们知道：
$
  sum_(i=1)^n Z_i^2 ~ chi^2(n)
$
又有
$
  overline(Z) ~ N(0,1/n)\
  n overline(Z)^2=(sqrt(n) overline(Z))^2~chi^2(1)
$
由于正态分布样本均值 $overline(Z)$ 和样本偏差 $Z_i - overline(Z)$ 独立,那么$n overline(Z)^2$ 和 $sum_(i=1)^n (Z_i-overline(Z))^2$ 独立.
由卡方分布的可加性,我们有：
$
  sum_(i=1)^n (Z_i-overline(Z))^2 ~ chi^2(n-1)
$

#theorem-box[t分布：

若$Z~N(0,1)$,$chi^2(k)$相互独立，则
$
  T=Z/sqrt(chi^2 (k)\/k) ~ t(k)
$
]
如果$X_1,X_2,dots,X_n stretch(~)^text("iid") N(mu,sigma^2)$, 
$sigma$未知，样本标准差为 $s$ ,Gosset发现统计量
$
  t=(overline(X)-mu)/(s\/sqrt(n))
$
不服从正态分布，而是服从t分布:
$
  t=((overline(X)-mu)\/(sigma\/sqrt(n)))/sqrt((n-1)s^2\/(n-1)sigma^2)=Z/(sqrt(chi^2(n-1)\/(n-1)))~t(n-1)
$
这里用到了正态分布的样本均值与样本方差独立性.
