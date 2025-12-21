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
    title: [],
    institution: [上海师范大学数理学院],
  ),
)
#theorem-box[$(a,b)$上的下凸函数$f$一定内闭Lipschitz连续]
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
