-- 基础 tactic 和核心逻辑
import Mathlib.Tactic
import Mathlib.Logic.Basic

-- 基础代数结构（Ring, Group 等定义）
import Mathlib.Algebra.Ring.Defs

-- 实数基础（通常包含在默认 Mathlib 中）
import Mathlib.Data.Real.Basic

-- 常用 tactic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

-- 集合基础（非常重要，且通常稳定）
import Mathlib.Data.Set.Basic

-- 自然数、整数基础
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic

-- Aesop
import Aesop


example (x y z :ℝ) (h0 :x≤ y)(h1 :y ≤ z): x ≤ z := by
apply le_trans
apply h0
apply h1


example : |(0.1 : ℝ) + 0.2 - 0.3| =0 := by
norm_num
open Real
example (h : a ≤ b) : log (1 + exp a) ≤ log (1 + exp b) := by
apply log_le_log
apply add_pos
norm_num
apply exp_pos
apply add_le_add_left
apply exp_le_exp.mpr
apply h

example (h : a ≤ b) : log (1 + exp a) ≤ log (1 + exp b) := by
-- log is monotonic, so it suffices to show (1 + exp a) ≤ (1 + exp b)
apply log_le_log

-- First, show that 1 + exp a > 0 (needed for log to be defined/monotonic)
· apply add_pos
· norm_num -- 1 > 0
· apply exp_pos -- exp a > 0

-- Second, show 1 + exp a ≤ 1 + exp b
· apply add_le_add_left
-- exp is monotonic, so a ≤ b implies exp a ≤ exp b
apply exp_le_exp.mpr h

def cons (α :Type)(a:α )(as:List α ):List α :=
List.cons a as
#check cons

example : 0 ≤ a^2 := by
exact sq_nonneg a

example {a b c : ℝ} (h : a ≤ b) : c - exp b ≤ c - exp a := by
-- 第一步：证明 hcd : exp a ≤ exp b
have h_exp : exp a ≤ exp b := exp_le_exp.mpr h
-- 第二步：apply tsub_le_tsub，自动匹配 hab = le_refl c，hcd = h_exp
apply tsub_le_tsub (le_refl c) h_exp

-- 目标：c - b ≤ c - a（已知 a ≤ b）
example {a b c : ℝ} (h : a ≤ b) : c - b ≤ c - a := by
-- tsub_le_tsub 需要两个前提：a_tsub ≤ b_tsub 和 c_tsub ≤ d_tsub
-- 这里手动填写第一个前提（c ≤ c，由自反性成立），用 ?_ 表示第二个前提
refine tsub_le_tsub (le_refl c) ?_
-- 此时 Lean 只生成一个子目标：a ≤ b（即第二个前提）
exact h -- 用已知条件 h 完成证明

example {a b c : ℝ} (h : a ≤ b) : c - b ≤ c - a := by
-- 第一步：应用 tsub_le_tsub 定理，自动匹配目标结构
apply tsub_le_tsub
-- 此时生成两个子目标（对应 tsub_le_tsub 的两个前提）：
-- 子目标1：a_tsub ≤ b_tsub（由目标 c - b ≤ c - a 推断 a_tsub = c, b_tsub = c → 需证 c ≤ c）
-- 子目标2：c_tsub ≤ d_tsub（推断 c_tsub = a, d_tsub = b → 需证 a ≤ b）

-- 第二步：证明子目标1：c ≤ c（预序的自反性）
apply le_refl -- 或 exact le_refl c，le_refl 自动匹配 c ≤ c

-- 第三步：证明子目标2：a ≤ b（直接使用已知假设 h）
exact h

example {a b c : ℝ} (h : a ≤ b) : c - exp b ≤ c - exp a := by
-- 利用指数函数单调性得到 exp a ≤ exp b
have h_exp : exp a ≤ exp b := exp_le_exp.mpr h
-- linarith 自动处理减法不等式（结合 h_exp 推导）
linarith


example {a b c : ℝ} (h : a ≤ b) : c - exp b ≤ c - exp a := by
linarith [exp_le_exp.mpr h]

theorem le_of_sq_sub_nonneg (a b :ℝ ) : 2 * a * b ≤ a ^ 2 + b ^ 2 := by
have h : 0 ≤ a ^ 2 - 2 * a * b + b ^ 2
calc
a ^ 2 - 2 * a * b + b ^ 2 = (a - b) ^ 2 := by ring
_ ≥ 0 := by apply pow_two_nonneg

calc
2 * a * b = 2 * a * b + 0 := by ring
_ ≤ 2 * a * b + (a ^ 2 - 2 * a * b + b ^ 2) := add_le_add (le_refl _) h
_ = a ^ 2 + b ^ 2 := by ring


example (a b : ℝ) : |a * b| ≤ (a ^ 2 + b ^ 2) / 2 := by
-- Step 1: |a * b| = |a| * |b|
have h1 : |a * b| = |a| * |b| := by
rw [abs_mul]

-- Step 2: a² + b² = |a|² + |b|² （因为 x² = |x|² 对实数成立）
have h2 : a ^ 2 + b ^ 2 = |a| ^ 2 + |b| ^ 2 := by
norm_num -- abs_sq: |x| ^ 2 = x ^ 2
-- Step 3: 应用你已证的定理，但用 |a| 和 |b| 代替 a, b
have h3 : 2 * |a| * |b| ≤ |a| ^ 2 + |b| ^ 2 := by
apply le_of_sq_sub_nonneg
have h4 : |a| * |b| ≤ (|a| ^ 2 + |b| ^ 2) / 2 := by
linarith [h3]
-- Step 5: 用 h1 和 h2 替换回原表达式
calc
|a * b| = |a| * |b| := h1
_ ≤ (|a| ^ 2 + |b| ^ 2) / 2 := h4
_ = (a ^ 2 + b ^ 2) / 2 := by rw [← h2]

example (a b : ℝ) : |a * b| ≤ (a ^ 2 + b ^ 2) / 2 := by
calc
|a * b| = |a| * |b| := by rw [abs_mul]
_ ≤ (|a|^2 + |b|^2) / 2 := by
linarith [le_of_sq_sub_nonneg |a| |b|]
_ = (a^2 + b^2) / 2 := by norm_num
#check le_antisymm

example (a b : ℝ) : min a b = min b a := by
apply min_comm a b


example (a b : ℝ) : min a b = min b a := by
apply le_antisymm
· -- goal: min a b ≤ min b a
aesop
· -- goal: min b a ≤ min a b
aesop
example (a b : ℝ): min a b = min b a := by
apply le_antisymm
· simp
· simp
example (a b : ℝ): min a b = min b a := by
have h : ∀ x y : ℝ, min x y ≤ min y x := by
intros
apply le_min
. simp
. aesop
apply le_antisymm
apply h
apply h

variable (a b c : ℝ)
theorem aux : min a b + c ≤ min (a + c) (b + c) :=
le_min
(add_le_add (min_le_left a b) (le_refl c))
(add_le_add (min_le_right a b) (le_refl c))

theorem aux2 : min a b + c ≤ min (a + c) (b + c) := by
apply le_min
· apply add_le_add (min_le_left a b) (le_refl c)
· apply add_le_add (min_le_right a b) (le_refl c)

theorem aux3 : min a b + c ≤ min (a + c) (b + c) := by
have h1 : min a b + c ≤ a + c :=
add_le_add (min_le_left a b) (le_refl c)
have h2 : min a b + c ≤ b + c :=
add_le_add (min_le_right a b) (le_refl c)
exact le_min h1 h2

theorem aux4: min a b + c ≤ min (a + c) (b+c):= by
apply le_min
· apply add_le_add
· apply min_le_left a b
· apply le_refl c
· apply add_le_add
· apply min_le_right a b
· apply le_refl c

theorem aux5 : min a b + c ≤ min (a + c) (b + c) := by
aesop

example (a b : ℝ) : |a| - |b| ≤ |a - b| := by
have h : |a| = |(a - b) + b| := by rw [sub_add_cancel]
rw [h]
have h2 : |(a - b) + b| ≤ |a - b| + |b| := abs_add (a-b) b
linarith

variable (a b c d : ℝ)
example: |a| - |b|≤ |a -b|:=by
calc
|a| - |b| = |a - b + b| - |b| := by rw [sub_add_cancel]
_≤ |a-b|+|b|-|b| := by
apply sub_le_sub_right
apply abs_add
_≤ |a-b|:=by simp

example : |a| - |b| ≤ |a - b| := by
have h:= abs_add (a-b) b
rw [sub_add_cancel] at h
linarith
variable (w x y z : ℕ)
example (h0 : x ∣ y)(h1 : y ∣ z) : x ∣ z :=
dvd_trans h0 h1

example : x ∣ y*x*z:=by
apply dvd_mul_of_dvd_left
apply dvd_mul_left

example :x ∣ x^2 :=by
apply dvd_mul_left

example (h : x ∣ w) : x ∣ y * (x * z) + x ^ 2 + w ^ 2 := by
apply dvd_add
· apply dvd_add
· apply dvd_mul_of_dvd_right
apply dvd_mul_right
apply dvd_mul_left
rw [pow_two]
apply dvd_mul_of_dvd_right
exact h

variable (m n : ℕ)
#check (Nat.gcd_zero_right n : Nat.gcd n 0 = n)
#check Nat.lcm_zero_right n

example : Nat.gcd m n = Nat.gcd n m := by
apply Nat.dvd_antisymm
repeat
apply Nat.dvd_gcd
apply Nat.gcd_dvd_right
apply Nat.gcd_dvd_left

example {α : Type*} [SemilatticeInf α] (x y z : α) :
(x ⊓ y) ⊓ z = x ⊓ (y ⊓ z) := by
apply le_antisymm
. apply le_inf
. calc
(x ⊓ y) ⊓ z ≤ x ⊓ y := inf_le_left
_ ≤ x := inf_le_left
. apply le_inf
. calc
(x ⊓ y) ⊓ z ≤ x ⊓ y := inf_le_left
_ ≤ y := inf_le_right
. exact inf_le_right
. apply le_inf
. apply le_inf
. exact inf_le_left
. calc
x⊓(y⊓z)≤y⊓z:=inf_le_right
_ ≤ y:=inf_le_left
. calc
x⊓(y⊓z)≤y⊓z:=inf_le_right
_ ≤ z:=inf_le_right
