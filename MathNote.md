# 卓里奇第二卷补遗

## 第九章

### 1.度量空间

#### 由严格上凸函数诱导的度量满足三角不等式

设 $ f: [0, \infty) \to [0, \infty) $ 是一个非负函数，满足：

- $ f(x) = 0 $ 当且仅当 $ x = 0 $；
- $ f $ 是**严格上凸**函数。

在 $ \mathbb{R} $ 上定义函数  
$$
d(x_1, x_2) = f(|x_1 - x_2|)
$$
可以推出：

**$ f $ 在 $ (0, \infty) $ 上严格递增**；

**反证**：假设存在 $ 0 < a < b $ 使得 $ f(a) \geq f(b) $。

由于 $ f $ 严格上凸，对任意 $ t > b $，将 $ b $ 表示为 $ a $ 与 $ t $ 的凸组合：
$$
b = \lambda a + (1 - \lambda)t,\quad \lambda = \frac{t - b}{t - a} \in (0,1).
$$
由严格上凸性，
$$
f(b) > \lambda f(a) + (1 - \lambda)f(t) \geq \lambda f(b) + (1 - \lambda)f(t),
$$
因为 $ f(a) \geq f(b) $。整理得
$$
f(b) > \lambda f(b) + (1 - \lambda)f(t) \quad \Rightarrow \quad (1-\lambda)f(b) > (1-\lambda)f(t).
$$
由于 $ 1-\lambda > 0 $，我们得到 $ f(b) > f(t) $。这意味着对任意 $ t > b $，有 $ f(t) < f(b) $。

现在考虑割线斜率。固定 $ b $，对于 $ t > b $，由严格上凸性，割线斜率 $ \frac{f(t)-f(b)}{t-b} $ 关于 $ t $ 是严格递减的。由于 $ f(t) < f(b) $，该斜率恒为负。

取 $ s > t > b $，则有
$$
\frac{f(s)-f(b)}{s-b} < \frac{f(t)-f(b)}{t-b} < 0.
$$
设
$$
 m = \frac{f(t)-f(b)}{t-b} < 0.
$$
于是
$$
f(s) < f(b) + m(s-b).
$$
由于 $ m < 0 $，当 $ s $ 足够大时，右端为负，即 $ f(s) < 0 $，与 $ f(x) \ge 0 $ 矛盾。

因此，假设不成立，从而对任意 $ 0 < a < b $ 必有 $ f(a) < f(b) $，即 $ f $ 在 $ (0,\infty) $ 上严格递增。

**验证三角不等式:**

任取 $ x, y, z \in \mathbb{R} $，令  
$$
a = |x - z|, \quad b = |z - y|, \quad c = |x - y|.
$$
由实数绝对值的三角不等式，有  
$$
c \leq a + b.
$$

由于 $ f $ 非负、严格递增，我们有  
$$
f(c) \leq f(a + b).
$$

现在利用步骤 2 中的关键不等式。若 $ a = 0 $ 或 $ b = 0 $，结论显然成立（例如 $ a = 0 \Rightarrow x = z $，则 $ d(x, y) = d(z, y) \leq d(x, z) + d(z, y) = 0 + d(z, y) $）。  
假设 $ a > 0 $ 且 $ b > 0 $。不妨设 $ a \leq b $（否则交换 $ a, b $，因 $ f $ 单调，不影响结果）。则 $ 0 < a \leq b $，于是由割线的斜率递减：
$$
f(a + b) - f(b) < f(a) - f(0)
$$

$$
f(a + b) - f(b) < f(a) \quad \Rightarrow \quad f(a + b) < f(a) + f(b).
$$

因此，
$$
d(x, y) = f(c) \leq f(a + b) < f(a) + f(b) = d(x, z) + d(z, y).
$$

即使 $ c = a + b $（三点共线且 $ z $ 在 $ x, y $ 之间），上述严格不等式仍成立；若 $ c < a + b $，由于 $ f $ 递增，仍有 $ f(c) \leq f(a + b) < f(a) + f(b) $。

综上，对任意 $ x, y, z \in \mathbb{R} $，  
$$
d(x, y) \leq d(x, z) + d(z, y),
$$
即三角不等式成立。

> 注:将$d(x_1, x_2) = f(|x_1 - x_2|)$改为$d(x_1, x_2) = f(d'(x_1 - x_2))$,$d'$是另外一个度量，上述证明依旧成立。



#### 直积度量满足三角不等式

设 $(X_1, d_1)$ 和 $(X_2, d_2)$ 为两个度量空间。在乘积空间 $X_1 \times X_2$ 上定义函数  
$$
d\big((x_1,x_2),(y_1,y_2)\big) = \sqrt{d_1^2(x_1,y_1) + d_2^2(x_2,y_2)}.
$$
我们将证明 $d$ 满足三角不等式。

任取三点 $(x_1,x_2), (z_1,z_2), (y_1,y_2) \in X_1 \times X_2$，记：
$$
\begin{aligned}
a_1 &= d_1(x_1, z_1), & a_2 &= d_2(x_2, z_2), \\
b_1 &= d_1(z_1, y_1), & b_2 &= d_2(z_2, y_2), \\
c_1 &= d_1(x_1, y_1), & c_2 &= d_2(x_2, y_2).
\end{aligned}
$$

由于 $d_1$ 和 $d_2$ 均为度量，由三角不等式得：
$$
c_1 \le a_1 + b_1, \qquad c_2 \le a_2 + b_2.
$$

两边平方并相加：
$$
c_1^2 + c_2^2 \le (a_1 + b_1)^2 + (a_2 + b_2)^2.
$$

开平方：
$$
\sqrt{c_1^2 + c_2^2} \le \sqrt{(a_1 + b_1)^2 + (a_2 + b_2)^2}.
$$

在 $\mathbb{R}^2$ 中，对向量 $(a_1, a_2)$ 和 $(b_1, b_2)$ 应用闵可夫斯基不等式（即欧几里得范数的三角不等式）：
$$
\sqrt{(a_1 + b_1)^2 + (a_2 + b_2)^2} \le \sqrt{a_1^2 + a_2^2} + \sqrt{b_1^2 + b_2^2}.
$$

将上述结果链式组合：
$$
\sqrt{c_1^2 + c_2^2} \le \sqrt{a_1^2 + a_2^2} + \sqrt{b_1^2 + b_2^2}.
$$

根据 $d$ 的定义，这等价于：
$$
d\big((x_1,x_2), (y_1,y_2)\big) \le d\big((x_1,x_2), (z_1,z_2)\big) + d\big((z_1,z_2), (y_1,y_2)\big).
$$

因此，$d$ 满足三角不等式。

> 如果$q\ge1$,$d\big((x_1,x_2),(y_1,y_2)\big) = (d_1^q(x_1,y_1) + d_2^q(x_2,y_2))^\frac{1}{q}$也是度量,令$q=\infty$此时$d=\rm{max}\{d_1(x_1,y_1),d_2(x_2,y_2)\}$.



### 拓扑空间

#### **度量拓扑的等价**

设 $d_1$ 和 $d_2$ 是集合 $X$ 上的两个度量。若对任意点 $x \in X$ 以及任意半径 $r > 0$，都存在正数 $\delta_1, \delta_2 > 0$，使得
$$
B_{d_1}(x, \delta_1) \subseteq B_{d_2}(x, r) \quad \text{且} \quad B_{d_2}(x, \delta_2) \subseteq B_{d_1}(x, r),
$$
则 $d_1$ 与 $d_2$ 诱导出相同的拓扑。

这是因为：
每个 $d_1$-开球在每一点处都包含某个 $d_2$-开球，故每个 $d_1$-开集也是 $d_2$-开集；
反之亦然。

由于拓扑由所有开集构成，上述双向包含关系意味着两个度量生成的拓扑完全一致（在度量诱导的拓扑中，我们对度量的具体大小不关心），从而它们是拓扑等价的。

#### 两种拓扑基定义的等价性

设 $X$ 为非空集合，$\mathcal{B} \subseteq \mathcal{P}(X)$。

**定义 A（公理化基）**

称 $\mathcal{B}$ 为一个**基**，若满足：  

1. $\forall x \in X,\ \exists B \in \mathcal{B},\ x \in B$；  
2. $\forall B_1, B_2 \in \mathcal{B},\ \forall x \in B_1 \cap B_2,\ \exists B_3 \in \mathcal{B},\ x \in B_3 \subseteq B_1 \cap B_2$。

**定义 B（拓扑意义下的基）**

设 $\tau$ 是 $X$ 上的一个拓扑。称开集族$\mathcal{B}$ 是 $\tau$ 的**基**，若
$$ \forall U \in \tau,\ \exists \mathcal{C} \subseteq \mathcal{B},\quad U = \bigcup_{B \in \mathcal{C}} B. $$

定义 A 与定义 B 等价，即：

- 若 $\mathcal{B}$ 满足定义 A，则存在唯一拓扑 $\tau$ 使得 $\mathcal{B}$ 是 $\tau$ 的基（按定义 B）；
- 反之，若 $\mathcal{B}$ 是某拓扑 $\tau$ 的基（按定义 B），则 $\mathcal{B}$ 满足定义 A。

 **证明:**

**(1) 定义 A ⇒ 定义 B**

假设 $\mathcal{B}$ 满足定义 A。
**定义** $$ \tau := \{ U \subseteq X\ |\  \forall x \in U,\ \exists B \in \mathcal{B} \text{ 使得 } x \in B \subseteq U \}. $$

**Step 1：$\tau$ 是拓扑**  

- $\varnothing \in \tau$（空真）；由 (A1)，$\forall x \in X$ 有 $x \in B \subseteq X$，故 $X \in \tau$。  
- 任意并：若 ${U_\alpha} \subseteq \tau$，对 $x \in \bigcup U_\alpha$，有 $x \in U_{\alpha_0}$，从而存在 $B \in \mathcal{B}$ 满足 $x \in B \subseteq U_{\alpha_0} \subseteq \bigcup U_\alpha$，故 $\bigcup U_\alpha \in \tau$。  
- 有限交：设 $U, V \in \tau$，$x \in U \cap V$。
	则 $\exists B_1, B_2 \in \mathcal{B}$ 使 $x \in B_1 \subseteq U$，$x \in B_2 \subseteq V$。
	由 (A2)，$\exists B_3 \in \mathcal{B}$ 使 $x \in B_3 \subseteq B_1 \cap B_2 \subseteq U \cap V$。
	故 $U \cap V \in \tau$。

因此 $\tau$ 是拓扑。

**Step 2：$\mathcal{B}$ 是 $\tau$ 的基（定义 B）**
取任意 $U \in \tau$。由 $\tau$ 的定义，对每个 $x \in U$，**存在** $B_x \in \mathcal{B}$ 使得
$$ x \in B_x \subseteq U. $$
令 $\mathcal{C} = \{ B_x \mid x \in U \} \subseteq \mathcal{B}$，则
$$ U = \bigcup_{x \in U} B_x = \bigcup_{B \in \mathcal{C}} B. $$
因此 $U$ 是 $\mathcal{B}$ 中元素的并，即 $\mathcal{B}$ 是 $\tau$ 的基。

**(2) 定义 B ⇒ 定义 A**

假设 $\mathcal{B}$ 是拓扑空间 $(X, \tau)$ 的基（按定义 B）。

- **(A1)**：因 $X \in \tau$，按照定义B，存在 $\mathcal{C} \subseteq \mathcal{B}$ 使得 $X = \bigcup_{B \in \mathcal{C}} B$。
	故对任意 $x \in X$，存在 $B \in \mathcal{C} \subseteq \mathcal{B}$ ，使得 $x \in B$。
- **(A2)**：设 $B_1, B_2 \in \mathcal{B}$，$x \in B_1 \cap B_2$。
	因 $\mathcal{B} \subseteq \tau$（基中元素必为开集），有 $B_1, B_2 \in \tau$，故 $B_1 \cap B_2 \in \tau$。
	由定义 B，存在 $\mathcal{D} \subseteq \mathcal{B}$ 使得 $B_1 \cap B_2 = \bigcup_{B \in \mathcal{D}} B$。
	所以 $\forall x \in B_1 \cap B_2$，存在 $B_3 \in \mathcal{D} \subseteq \mathcal{B}$ 使得 $x \in B_3 \subseteq B_1 \cap B_2$。

因此 $\mathcal{B}$ 满足定义 A。





设$f\in C[0,+\infty)\cap D^2(0,+\infty)$且$f''\le\frac{C}{x^2}$.则有
$$
\lim_{x\to 0^+} xf'(x)=0
$$
$\forall x \in (0,+\infty)$ 对于$\forall \eta \in (0,1)$,由Taylor中值定理我们知道存在$\theta \in (x-\eta x,x)$,使得:
$$
f(x-\eta x)=f(x)-\eta xf'(x)+(\eta x)^2\frac{f''(\theta)}{2}
$$

$$
f'(x)=\frac{f(x)-f(x-\eta x)}{\eta x}+(\eta x)\frac{f''(\theta)}{2}
$$

$$
\le f'(x)=\frac{f(x)-f(x-\eta x)}{\eta x}+(\eta x)(\frac{C}{2 {\theta}^2})
$$

因此：
$$
xf'(x) \le \frac{f(x)-f(x-\eta x)}{\eta}+(\eta x^2)(\frac{C}{2 {\theta}^2})
$$

$$
\le \frac{f(x)-f(x-\eta x)}{\eta}+\frac{C \eta}{2} \frac{x^2}{(x-\eta x)^2}
$$

$$
\le\frac{f(x)-f(x-\eta x)}{\eta}+\frac{C \eta}{2(1-\eta)^2}
$$



同理，展开$f(x+\eta x)$有$\vartheta \in (x,x+\eta x)$使得：
$$
f(x+\eta x)=f(x)+\eta xf'(x)+(\eta x)^2\frac{f''(\vartheta)}{2}
$$

$$
f'(x)=\frac{f(x+\eta x)-f(x)}{\eta x}-(\eta x)\frac{f''(\vartheta)}{2}
$$

$$
xf'(x) \ge f'(x)=\frac{f(x+\eta x)-f(x)}{\eta }-\frac{C\eta}{2}
$$

因此：
$$
\frac{f(x+\eta x)-f(x)}{\eta }-\frac{C\eta}{2}\le xf'(x) \le\frac{f(x)-f(x-\eta x)}{\eta}+\frac{C \eta}{2(1-\eta)^2}
$$
令$x\to 0^+$,得到：
$$
-\frac{C\eta}{2}\le \lim_{x\to0^+} xf'(x) \le \frac{C \eta}{2(1-\eta)^2}
$$
令$\eta \to 0^+$,得到：
$$
\lim_{x\to0^+} xf'(x) =0
$$


