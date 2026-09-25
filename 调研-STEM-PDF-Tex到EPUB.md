# 调研：PDF / TeX → Typst → EPUB（STEM 类）

> 结论速览：链路形状成立，但难度极度不均。GitHub 上每段都有现成项目，
> 唯一真正未解决的是「PDF → Typst」这一跳中的**公式还原**。
> 建议不要把三条输入路径合并成一个通用 pipeline，而是做成**路由 + 三条独立支线**。

---

## 1. 链路拆解与现有项目清单

### 1.1 TeX / LaTeX → Typst（难度：中）

| 项目 | 语言 | 能力 | 备注 |
|---|---|---|---|
| [scipenai/tylax](https://github.com/scipenai/tylax) | Rust | AST 级双向 LaTeX ↔ Typst；整篇文档（章节/列表/表格/booktabs/multicolumn/参考文献）；实验性 TikZ ↔ CeTZ | `cargo install tylax`；CLI `t2l`；有 WASM 在线 demo。**目前最接近需求的一个** |
| [qwinsi/tex2typst](https://github.com/qwinsi/tex2typst) | JS | 仅数学公式，双向 | Apache-2.0，工程稳定 |
| [ovo-Tim/latex2typst](https://github.com/ovo-Tim/latex2typst) | Rust | Markdown+LaTeX 数学、纯 LaTeX 文档；WASM | Apache-2.0，较新 |
| Pandoc | Haskell | `-f latex -t typst` | LaTeX reader 是尽力而为：自定义宏、复杂宏包容易掉；但 AST 稳定、生态最广 |
| [mitex](https://typst.app/universe/package/mitex) | Typst 包 | 在 Typst 中**原样排版 LaTeX 数学** | 关键逃生舱：把「公式转换失败」降级为「公式原样嵌入」 |

**主要坑**：自定义宏 / `\newcommand` 展开、TikZ / pgfplots、algorithm2e、lstlisting、
`\input` `\include` 扁平化、biblatex、交叉引用与编号语义（Typst 的 `@label` 与 LaTeX `\ref` 语义不同）。

### 1.2 PDF → Typst（难度：中高 ~ 高）

| 项目 | 状态 | 说明 |
|---|---|---|
| [pgarrett-scripps/rustypaper](https://github.com/pgarrett-scripps/rustypaper) | **可用，v0.2.1（2026-09）** | 结构感知地把**数字版**科学 PDF 转 Markdown / **Typst** / JSON。纯 CPU、无模型、单二进制、2.5–8.7 ms/页、3.2 MB。公式按字形几何重建（MaxTract 风格），带置信度，低置信度回退为裁剪图。MIT OR Apache-2.0。**明确对扫描件报 `Error::Scanned` 说不** |
| [jehaj/Convert-PDF-to-Typst](https://github.com/jehaj/Convert-PDF-to-Typst) | 早期草图 | pdftotext / Pix2Text / 本地 LLM 拼装，作者自己说 pdftotext 处理公式很糟 |
| [luoqiao6/TypstAgent](https://github.com/luqiao6/TypstAgent) | **空壳** | ReAct + Reflection 架构，只有接口定义，「具体实现待完善」。是想法仓库，别浪费时间 |
| [AnClark/PDF2Epub_2](https://github.com/AnClark/PDF2Epub_2) | 可用 | 中文**扫描版** PDF → EPUB + Typst，Tesseract OCR，标题层级与自然段重建（首行缩进/段末短行/行距启发式）。**纯散文，无公式**。架构值得参考 |

rustypaper 的自评指标（15 篇可评分 arXiv 论文）非常值得看，它给出了「无模型方案」的现实天花板：

- 正文 bigram 召回 **0.900**
- 公式召回 **0.565**
- 公式保真度 **0.677**
- 表格 62 / 90，参考文献 971 / 1174

> 即：**正文基本没问题，公式只有一半能找出来，找出来的也只有六成多是对的。**

**扫描件 / VLM 公式识别方案**（当 rustypaper 报 Scanned 时的下一档）：

| 方案 | 许可 | 公式 | 说明 |
|---|---|---|---|
| olmOCR (AI2) | Apache-2.0 | 强 | 7B VLM，olmOCR-Bench 82.4；「老扫描 + 数学」分项 82.3；需 GPU 或走托管 API（约 $0.07–0.10 / 百万输入 token） |
| PaddleOCR-VL | — | 很强 | OmniDocBench 91.93，当前最高档之一 |
| MinerU | **AGPL-3.0** | 强 | 中文与复杂版面最好；底层 PaddleOCR，生僻字受限 |
| dots.ocr | — | 强 | OmniDocBench 0.125 |
| Marker | **GPL-3.0** | 一般 | 速度快、适合批量 |
| Nougat | — | 一般 | 有**幻觉**问题，需校验 |
| Mathpix | 商业 | 最强 | OmniDocBench 0.191，公式领域天花板 |
| Gemini 2.5 Pro / Qwen2.5-VL-72B | — | 强 | 通用 VLM 直读页面 |

评测基准：OmniDocBench、olmOCR-Bench、dp-bench。**建议从第一天就自建回归集。**

### 1.3 Typst → EPUB（难度：低）

- [freecomputinglab/rheo](https://github.com/freecomputinglab/rheo) — **就是你要的那一环**。
  把目录下的 Typst 同时编译为 PDF / HTML / EPUB，Rust 实现、把 Typst 当库调用，
  有专门的 `rheo-epub` 组件；用 **Spines** 把多个文件合成多章节 EPUB。
  已锁定 Typst v0.15.0，版本 0.6.4（2026-09-19），Apache-2.0 / MIT 双许可，文档在 rheo.ohrg.org。
  另有 `rheo-author`（一个 Claude Code skill，用于撰写 Rheo 文档）。
- Typst 官方路线：`html` 导出仍**在实验特性后面**（`--features html`）；
  **0.15（2026-06-15）起公式可直接输出 MathML**，不再需要 `html.frame`；新增 `bundle` 目标支持多文件输出。
  Typst **目前不输出 CSS**，也**尚无原生 EPUB 导出**（仍在 roadmap）。
- 其他：`latexmlc --dest foo.epub foo.tex`（LaTeXML 直接出 EPUB）、tex4ebook、Calibre `ebook-convert`、Pandoc HTML→EPUB。

### 1.4 隐藏的深坑：EPUB 阅读器的数学渲染

EPUB 3 规范支持 MathML，但阅读器实现参差：

| 阅读器 | MathML | 说明 |
|---|---|---|
| Calibre E-book Viewer | 完整 | 参考实现 |
| Apple Books | 部分 | 某些版本忽略 `mrow` / `mtable` 排版节点 |
| Kindle / KF8 | 不支持 | 数学必须转图片 |
| 微信读书 | 忽略 | 但允许 MathJax |
| Android 各类阅读器 | 差 | 需注入 MathJax |

**业界通行做法是三层回退**：MathML 主体 + `altimg`（SVG/PNG 备用）+ `alttext`（纯文本描述）。
NCBI 那篇 NLM→ePub3 的经验分享直接说：因为回退链不可靠，他们最终干脆只发图片。

> 这意味着：**你 EPUB 的质量上限由阅读器决定，不由你的 pipeline 决定。**

---

## 2. 难度评估（分层）

| 层级 | 目标 | 现状 | 估量 |
|---|---|---|---|
| T0 | 单篇有 TeX 源的 arXiv 论文 → EPUB | tylax + rheo 直接串起来 | 1–2 周 |
| T1 | 批量 arXiv（约 90% 有 TeX 源） | 长尾在宏包与自定义命令；工程量主要在**失败分类 + 降级策略**，不在转换本身 | 2–3 个月到「能跑但需抽检」 |
| T2 | 数字版 STEM 书 PDF | 公式是天花板，rustypaper 的 0.565/0.677 就是当前无模型方案的现实水平。要提升必须上 VLM，而 VLM 会幻觉 | 3–6 个月到「可用但每章需校对」 |
| T3 | **扫描版 STEM 书** | 最难一档。密集公式的老扫描件全自动不现实 | 只能「人机协同 + 逐页校对」 |
| T4 | Typst → EPUB | rheo 基本免费；瓶颈在 Typst HTML 导出仍是实验特性 + 阅读器兼容 | 1–2 周集成，长期跟着 Typst 版本走 |

**真正的成本中心**：QA / 校对闭环 > 长尾宏包与版式 > 阅读器兼容矩阵。
不是「转换算法」。如果按「先写转换器」的思路做，大概率会在长尾上耗死。

---

## 3. 替代路线

| 路线 | 路径 | 优点 | 缺点 |
|---|---|---|---|
| **A. LaTeXML 直通** | LaTeX → LaTeXML → HTML5 + MathML → EPUB | 保真最高。arXiv 自己就用 LaTeXML；`ar5ivist` 提供 turnkey Docker；ar5iv 覆盖约 90% arXiv 的 HTML 版可直接对照；`latexmlc` 能直接出 EPUB | 中间层你不掌握，可编辑性差；EPUB 输出质量一般；需 epubcheck |
| **B. Typst 枢纽（你的方案）** | TeX/PDF → Typst → rheo → EPUB | 单一源、可编辑、同时产出漂亮 PDF，长线可维护 | 两跳损耗；Typst HTML 导出实验性；Typst 未 1.0，升级会 breaking |
| **C. Markdown 枢纽** | PDF/TeX → Pandoc AST/Markdown + LaTeX 数学 → (EPUB \| Typst) | 工程量最小，生态最广，两个后端各一跳 | Markdown 表达不了定理/交叉引用/编号语义；Pandoc 的 LaTeX reader 对重宏源会掉链子 |
| **D. 固定版式 / 图像化** | 公式区域保留高清 SVG/PNG，文本层 OCR | 100% 保真、成本极低 | 牺牲可重排与公式可搜索；阅读体验差 |
| **E. 绕过 EPUB** | Typst → HTML + MathML 网页阅读器 | 数学阅读体验最好，渲染你自己可控 | 不是 epub 生态；离线/阅读器支持另说 |

### 关键洞察
对**数学阅读体验**而言，响应式 HTML + MathML 往往比 EPUB 好——
这也正是 rheo 和 Free Computing Lab 那篇 *Document Infrastructure for Augmented Reading*
所描述的方向。**建议把 EPUB 当作降级导出，而不是唯一终点。**

---

## 4. 推荐架构

1. **输入路由层**：先判源类型（有 TeX 源 / 数字版 PDF / 扫描件），走三条独立支线。
   一个通用 pipeline 会同时继承三者的全部长尾。
2. **统一中间表示**：不要端到端 PDF → Typst。
   用结构化 IR 作契约（rustypaper 的 Document JSON / Pandoc AST / LaTeX AST），
   渲染器（Typst / HTML / EPUB）只是多个后端。
3. **LLM 只做语义补全，不做像素转文字**：对 VLM 输出施加结构约束 + 编译验证。
4. **杀器：编译 → 渲染 → diff 自动回归**。Typst 编译是毫秒级且可渲染 PNG，
   因此可以把输出页面与原始页面区域做结构化 diff / SSIM / 公式符号 diff。
   每页给置信度，低置信度进人工队列。
   **这是本项目相对「一次性 VLM 出结果」方案的核心差异化。**
5. **渐进式转换**：公式转不动就用 mitex 原样嵌入，不硬转；
   EPUB 端用 MathML + `altimg` + `alttext` 三层回退。
6. **从 TeX 路线起步**：它立刻给你可交付产物 + 完整 QA 基线，再往上加 PDF 路线。

---

## 5. 风险清单

- **许可传染**：MinerU 是 AGPL-3.0、Marker 是 GPL-3.0，服务化/商用需谨慎；
  olmOCR 为 Apache-2.0，rustypaper / Typst / rheo 为 MIT / Apache，较友好。
- **版权**：扫描或购买的书转制并分发有法律风险；arXiv 论文版权归作者，注意 CC 与非 CC 的区别。
- **Typst 未到 1.0**：HTML 导出是实验特性，升级可能 breaking；rheo 锁定了 Typst 0.15.0。
- **评测集**：公式识别的回归集必须从第一天建，否则无法判断迭代是否真的有效。

---

## 6. 参考链接

- https://github.com/scipenai/tylax
- https://github.com/qwinsi/tex2typst
- https://github.com/ovo-Tim/latex2typst
- https://github.com/pgarrett-scripps/rustypaper
- https://github.com/AnClark/PDF2Epub_2
- https://github.com/jehaj/Convert-PDF-to-Typst
- https://github.com/freecomputinglab/rheo ・ https://rheo.ohrg.org
- https://github.com/dginev/ar5ivist ・ https://ar5iv.labs.arxiv.org
- https://typst.app/docs/changelog/0.15.0
- https://typst.app/docs/reference/html/
