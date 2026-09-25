# 决策记录：仅限 MathML 路线时，"已有工具" vs "我们的 Typst 路线"

日期：2026-09-20
前提：只接受**语义 MathML** 作为公式载体（排除裁图 CLIPPING、排除 SVG、排除客户端 JS 渲染）。

---

## 结论

**已有路线更好，而且差距不小。** Typst 在这条链路上的正确位置不是主链路，而是**一个可插拔的 MathML 渲染后端 + 一个跨渲染 QA 校验器**。

---

## 一、为什么——四条判断依据

### 1. 分叉只发生在一个节点上

两条路线共用同一个上游（公式识别）和同一个下游（EPUB 打包）：

```
PDF 源 → 公式识别（VLM / OCR）→ LaTeX 字符串
                                  ├─ 已有路线：latex2mathml / Temml → MathML
                                  └─ Typst 路线：Typst 源 → typst --features html → MathML
                                        ↓（汇合）
                                  EPUB3：XHTML 内联 MathML
```

也就是说，"换成 Typst" 改变的**只是 MathML 由谁生成**，没有改善真正困难的那一步。

### 2. 真正的瓶颈两条路线完全相同，且 Typst 不改善它

STEM PDF 的难点是从页面里**把公式正确地读出来**（上次调研：无模型方案公式召回 0.565、保真度 0.677）。这一步 Typst 路线非但不改善，还多了一跳（LaTeX → Typst 源），属于净负。

### 3. Typst 唯一的质量优势是可移植的——不需要接纳整条路线

Typst 相对已有路线的实际增益只有一条：MathML 质量（编译器原生、且排版引擎保证与 PDF 视觉一致）。

但这一点**可以单独摘走**：

| 生成器 | 输入 | 许可 | 面向标准 | 说明 |
|---|---|---|---|---|
| **Temml** | LaTeX 字符串 | MIT | **MathML Core** | 从 KaTeX 分叉、专修 MathML bug，覆盖接近 MathJax，Node 无头可跑 |
| MathJax-node | LaTeX 字符串 | Apache-2.0 | MathML / SVG | 覆盖最全，最成熟，但重、启动慢 |
| Typst 内置 | Typst 数学语法 | Apache-2.0 | 贴合自身渲染 | `crates/typst-html/src/mathml.rs`，0.15 起原生 |
| latex2mathml | LaTeX 字符串 | MIT | MathML 3 风格 | 纯字符串转换、不排版；**Docling 的 HTML 导出走它** |

**要 LaTeX → MathML 的话，Temml 比 Typst 更对路**（直接吃 LaTeX、明确面向 MathML Core、MIT）；Typst 的优势场景是"源本来就是 Typst"。既然上游识别天然产出 LaTeX，不必绕到 Typst。

### 4. 风险面对比

| | 已有路线 | Typst 路线 |
|---|---|---|
| 今天能否端到端出 EPUB | ✅ pdf-craft 一条命令 | ❌ 需自建 PDF → Typst |
| MathML 生成器 | 可热插拔（4 个候选） | 绑定 Typst 编译器版本 |
| 官方成熟度标注 | 可用（pdf-craft v2.2.0 / Docling） | HTML 导出官网原文："Do not use this feature for production use cases" |
| 样式 | 自己控制 | **Typst 不输出 CSS**，只出语义标记 |
| 输出粒度 | 自己打包 | 只能出**单个 standalone HTML**，fragment 支持还在计划中 |
| 许可 | pdf-craft / Docling / MarkItDown = MIT | Typst = Apache-2.0，rheo = Apache-2.0/MIT |

---

## 二、两处会直接踩到的实测细节

1. **pdf-craft 自己承认 MathML 在 EPUB 里不可靠。** 其文档原文：MATHML 在 EPUB 2.0 不受支持，"if this rendering method is used, not all EPub readers can render the formula correctly"，并建议 **SVG** 才是全阅读器通用方案。也就是说它给了你 MathML 开关，但不为兼容性背书。

2. **pdf-craft 的 EPUB 行内公式默认保留为 LaTeX 代码。** `inline_latex=True`（EPUB 默认）说明是"可由兼容的 EPUB 阅读器渲染"——而事实上几乎没有 EPUB 阅读器带数学 JS 渲染器。**块级公式有 MathML、行内公式掉回裸 LaTeX 源码**，这在 STEM 书里是灾难级体验，必须显式处理。

---

## 三、EPUB 特有的 MathML 陷阱（与 HTML 不同）

- **HTML5 容错 ≠ XHTML 严格。** latex2mathml 在遇到畸形公式时会产出不合法结构；浏览器会糊过去，但 EPUB 的 XHTML 是严格 XML 解析，一个未转义的 `<` / `&` 直接让整本书打不开。**这就是"HTML 出得来、EPUB 打不开"的根因。**
- **命名空间必须正确**：内联 MathML 需要 `xmlns="http://www.w3.org/1998/Math/MathML"`。
- **浏览器实现的是 MathML Core**，latex2mathml 参照的是 MathML 3——存在系统性偏差。Temml / Typst 都明确面向 Core。
- **每个公式补 `alttext`**，否则低端阅读器上是空白。
- 阅读器支持矩阵（上次已记）：Calibre 完整、Apple Books 部分、**Kindle/KF8 不支持**、微信读书忽略。

---

## 四、推荐架构（取两边优点的交集）

```
PDF → [上游识别：pdf-craft / Docling]  → LaTeX
                                          ↓
      [MathML 生成：可插拔后端，默认 Temml，备选 Typst/mitex，兜底 latex2mathml]
                                          ↓
      [自建打包：xhtml + MathML 内联 + alttext 三层回退]
                                          ↓
                                     EPUB3
```

关键点：**MathML 生成器做成接口，不要绑定。** 这样 Typst 想用时随时能插进来，而不必让整条链路跟着 Typst 的版本与实验特性走。

Typst 保留的两种用法：
1. **MathML 渲染备选后端**（通过 mitex 直接吃 LaTeX 数学，`LaTeX → mitex → Typst → MathML`）。
2. **跨渲染 QA 校验器**：同一份公式同时喂给 TeX 引擎和 MathML 渲染器做视觉 diff，低置信度进人工队列——这正是之前定的 QA 闭环，而 Typst 在这里的价值是"能同时出 PDF 和 HTML"。

---

## 五、什么情况下结论会反转

- **重新引入 TeX 源（arXiv）**——Typst 路线立刻变成最优：无识别损耗，`tylax` 直接 AST 级转换，还能单源双出 PDF + EPUB。
- **要的是"长期可维护的书稿源"而不只是一次转换产物**——只有 Typst 同时具备类型化数学 AST + 可编译验证 + 双后端。
- **要对 MathML 做程序化加工**（批量注入 alttext / intent、按公式切分、按章节定制）——Typst 的 show rules 是天然控制点，而 latex2mathml 是黑盒。

---

## 附：本次核实到的原始事实

- Typst 0.15 原生 MathML：`typst.app/blog/2026/typst-0.15`；实现位于 `crates/typst-html/src/mathml.rs`
- Typst HTML 导出官方限制：`typst.app/docs/reference/html/`（单文件、无 CSS、勿用于生产）
- Docling 公式链路：`do_formula_enrichment`（CodeFormula 模型）→ LaTeX → HTML 导出经 latex2mathml 渲染
- pdf-craft 渲染模式与其自述的限制：`github.com/oomol-lab/pdf-craft`
- Temml：`github.com/ronkok/Temml`（0.13.5 / 2026-08）；latex2mathml：PyPI 3.81.0 / 2026-04，MIT
