#import "@preview/polylux:0.4.0": *
#import "@preview/touying:0.6.1"
#import "touying-template/theme.typ": *
#import "ori.typ": *
#import "@preview/cetz:0.4.2"
#import "@preview/wrap-it:0.1.1": *
#show: mytheme.with(
  aspect-ratio: "16-9",
  font-en: "Noto Sans",
  font-ja: "BIZ UDPGothic",
  font-math: "Noto Sans Math",
  config-info(
    title: [圣彼得堡悖论与比例效用理论 @XAJD201706002],
    subtitle: [],
    author: [孟祥瑞, 高冰鑫, 汤皓亮],
    institution: [上海师范大学数理学院],
    header: [ #set text(size:25pt) 
    @XAJD201706002 王首元. 圣彼得堡悖论新解——比例效用理论溯源经典]
  )
)
#title-slide()

== 圣彼得堡游戏@XBSW200405002

圣彼得堡悖论是尼古拉・伯努利(1713)提出的一个概率期望值悖论,它来自于一种掷币游戏即圣彼得堡游戏。设定掷出正面或者反面为成功,游戏者如果第一次投掷成功,得奖金1元,游戏结束；第一次若不成功,继续投掷,第二次成功得奖金2元,游戏结束;这样游戏者如果投掷不成功就反复继续投掷,直到成功游戏结束。如果第 n 次投掷成功,得奖金$2^(n-1)$元,游戏结束。
#align(center)[
#image("assets/stpetersburg.pdf", width: 23cm)
]
==  期望收益
计算该游戏的期望收益:
$ 
  bb(E) = sum_(n=1)^infinity (1/2)^n dot 2^(n-1) = sum_(n=1)^infinity 1 /2= infinity
$
因此,按照期望收益理论,任何一个理性的游戏者都应该愿意支付任意高的金额来参与这个游戏.

然而在实验中,大多数人愿意支付的金额却极低，根据@Hayden_Platt_2009 的实验结果显示,大多数人愿意支付的金额在1到2美元之间,中位数仅为1.5美元,这与期望收益理论的预测相差甚远.

丹尼尔・伯努利在1738年发表论文,引入了期望效用的概念,部分解释了这个悖论,开创了现代效用理论的先河。

== 自然对数效用

丹尼尔·伯努利提出的反比例法则认为：个人的比例效用增量是财富变化量与个人初始财富数量的比值,即预想比例(PRU)效用:
$ 
  text("PRU")_(W -> W + Delta W) = (Delta W)/W
$
如果人们足够细心地评价增量财富的每一个微小变化，那么财富由初始

$W$ 变化到 $W + Delta W$ 时的效用增量为:
$ 
  text("PRU")_(W -> W + Delta W) = integral_(W)^(W + Delta W) (1/x) dif x = ln(W + Delta W) - ln(W) = ln((W + Delta W)/W)
$
此时,圣彼得堡游戏的自然对数期望效用(NLU)为:
#v(-1em)
$ 
  text("NLU")_(W -> (2^(n-1),(1/2)^n))= sum_(n=1)^infinity (1/2)^n dot ln((2^(n-1))/W) < infinity
$

== 比例效用
伯努利的自然对数效用过于理想化,因为它假设人们能够细心地评价增量财富的每一个微小变化,但在现实中,效用应该是离散的,即:
$ 
  text("PRU")_(W -> W + Delta W) = (Delta W)/W
$
这一形式的问题是, $Delta u$是关于$Delta W$的线性函数,这不符合财富增加时边际效用递减的规律.

托德亨特提出应该使用财富变化量和最终财富的比值来计算效用(真实比例效用),即:
$ 
  text("ARU")_(W -> W + Delta W) = (Delta W)/(W + Delta W)
$
== 效用函数的图像
#align(center)[
#image("assets/image.png", width: 25cm)]

== 效用函数的性质
根据NLU和ARU具有以下性质:
- 边际效用递减:
$ 
  text("ARU")_(W -> W + Delta W) = (Delta W)/(W + Delta W) > (Delta W)/ (W + Delta W' + Delta W) = text("ARU")_(W + Delta W' -> W + Delta W' + Delta W)
$
- 损失厌恶
$ 
abs(text("ARU")_(W -> W - Delta W)) = abs((-Delta W)/(W - Delta W)) > (Delta W)/(W + Delta W) = text("ARU")_(W -> W + Delta W)
$
- ARU上限为1下限为$-infinity$,
- NLU上限为$infinity$下限为$-infinity$.

== 风险资产和报价
假设有风险资产 $(x_i,p_i)$,$sum_i p_i =1$,假设投资者付出$P$价格获得风险资产.

当投资者按照真实比例效用函数衡量投资时,该资产的期望效用为:
$
text("ARU")_(W -> (W -P + x_i,p_i)) = sum_i p_i dot ((-P+x_i)/(W -P + x_i))
$

当投资者按照自然对数效用函数衡量投资时,该资产的期望效用为:
$
text("NLU")_(W -> (W -P + x_i,p_i)) = sum_i p_i dot ln((W -P + x_i)/W)
$
报价与效用水平反方向变动，所以对决策者
而言，理论上存在一个报价的上限，该报价上限恰好使得效用水平为零，分别记为$P_(text("AR"))$和 $P_(text("N L"))$，则有:
$
text("ARU")_(W -> (W -P_(text("AR")) + x_i,p_i)) = 0
,quad
text("NLU")_(W -> (W -P_(text("N L")) + x_i,p_i)) = 0
$
== 风险溢价
$ 
text("ARU")_(W -> (W -P_(text("AR")) + x_i,p_i)) = 0
$
$
sum_i p_i dot ((-P_(text("AR"))+x_i)/(W -P_(text("AR")) + x_i))=0
$
$
1- W dot bb(E)(1/(W - P_(text("AR"))+x_i))=0
$

$
1/W = bb(E)(1/(W - P_(text("AR"))+x_i)) >= 1/(W - P_(text("AR"))+bb(E)(x_i)) 
$

$
P_(text("AR")) <= bb(E)(x_i)
$
这解释了为何投资者的报价上限小于风险资产的期望收益来购买风险资产,即存在风险溢价.
== 报价上限和初始资产
根据ARU和NLU的定义,可以得出报价上限与初始资产的关系:
$
(dif P_(text("AR")))/(dif W) = -(partial text("ARU")\/ partial W)/(partial text("ARU")\/ partial P_(text("AR"))) > 0
$
$ 
(dif P_(text("N L")))/(dif W) = -(partial text("NLU")\/ partial W)/(partial text("NLU")\/ partial P_(text("N L"))) > 0
$
这解释了为什么富人投资风险资产的意愿更强.

== 独立投资评估
当投资者独立评估某项风险资产时，比较理性的选择是即将其他资产与该投资进行隔离。例如，当决策者有初始财富$W$，为风险资产$(x_i,p_i)$的报价为$P$时，从独立评估的角度而言，剩余财富$W - P$与该风险资产无关。因而，独立评估风险资产时,报价等于初始财富额，以$P_(text("AR"))$与$P_(text("N L"))$表示独立评估风险资产时的报价上限，如下所示：
$
text("ARU")_(P_(text("AR")) -> (x_i,p_i)) = 0
,quad
text("NLU")_(P_(text("N L")) -> (x_i,p_i)) = 0
$
计算得:
$
P_(text("AR")) = 1/(bb(E)(1/x_i)), quad P_(text("N L")) = product_i x_i^(p_i)
$
== 结果
带入圣彼得堡游戏的风险资产$(2^(n-1),(1/2)^(n))$,可以得出独立评估风险资产时的报价上限为:
$
P_(text("AR")) = 1/(bb(E)(1/2^(n-1))) = 1.5, quad P_(text("N L")) = product_(n=1)^infinity (2^(n-1))^((1/2)^n) = 2
$
其中使用ARU计算的报价上限为1.5美元，这一定量理论计算结果与Hayden@Hayden_Platt_2009 等进行的实验结果相符，这为该理论提供了有力的支持.

#pagebreak()
#bibliography("bib/IEEE Xplore Citation BibTeX Download 2026.3.8.20.9.42.bib")

