1.设$f in C[0,1]$,使得
$
  integral_0^1 f(x) x^n dif x=0,forall n=0,1,2,dots
$
求证$f(x)=0,forall x in [0,1]$
证明:
对于任意$epsilon >0$由Weistrass逼近定理,存在多项式$p(x)$使得
$
  |f(x)-p(x)|<epsilon,forall x in [0,1]
$
由题目条件知  
$
  integral_0^1 f(x) p(x) dif x=0\
  integral_0^1 f^2(x) dif x=integral_0^1 f(x)(f(x)-p(x)) dif x<=integral_0^1 |f(x)| dot |f(x)-p(x)| dif x<epsilon integral_0^1 |f(x)| dif x
$
由$epsilon$的任意性，可得
$
  integral_0^1 f^2(x) dif x=0
  =>f(x)=0,forall x in [0,1]
$
