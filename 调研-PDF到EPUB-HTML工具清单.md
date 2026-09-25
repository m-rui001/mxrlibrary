# 调研:STEM 类 PDF → EPUB / HTML 工具清单

> 范围:只做 PDF 输入(不含 TeX/LaTeX 源)。
> 结论:**这一段已经有直接出 EPUB 的成品,不需要自己造。** `pdf-craft` 最接近需求。

---

## 0. 三条技术路线

| 路线 | 做法 | 保真 | 可重排 | 代表 |
|---|---|---|---|---|
| **重排型** | OCR/解析 → 结构化文档 → EPUB | 公式易失真 | ✅ 小屏友好 | Marker、Docling、pandoc |
| **混合型** ⭐ | 正文重排 + 公式区域裁剪为图 | 均衡 | ✅ | **pdf-craft(默认行为)** |
| **版面保真型** | 保留原页几何,输出定位 HTML/图片 | 100% | ❌ | pdf2htmlEX |

STEM 书的现实最优解是**混合型**:正文可重排保证小屏体验,公式保真保证不出错。

---

## 1. 直接出 EPUB 的

| 项目 | 状态 | 输入 | 公式 | 许可 | 备注 |
|---|---|---|---|---|---|
| **[oomol-lab/pdf-craft](https://github.com/oomol-lab/pdf-craft)** | ⭐4.6K,v2.2.0(2026-09),持续提交 | 扫描版书籍 PDF | ✅ LaTeX-OCR;EPUB 端可选 **MathML / SVG / 裁剪图** | MIT | **首选**。DeepSeek OCR 全程本地、无网络请求;自动生成 EPUB 目录、过滤页眉页脚、保留脚注(含脚注内图片)、表格。v1.0 起移除 LLM 文本矫正(需要则用 v0.2.8)。需 Poppler;公式/表格**抽取**需 CUDA,否则自动回退裁剪 |
| [porfanid/pdf2epub](https://github.com/porfanid/pdf2epub) | `pip install pdf2epub` | PDF | 宣称支持 | — | PDF→Markdown→EPUB,AI 版面检测 + AI 后处理(Anthropic)。有 GPU 加速与插件式 AI provider |
| [lgmoneda/pdf-to-epub](https://github.com/lgmoneda/pdf-to-epub) | 2026-03 起 | 学术 PDF / URL | 改善公式渲染 | — | Mistral OCR + Pandoc;**检测到 arXiv HTML 时自动走 HTML→Markdown→EPUB**;带 10 篇 arXiv 的 benchmark 套件 + 可选 LLM 质量判官。**这个反馈闭环值得直接抄** |
| [AnClark/PDF2Epub_2](https://github.com/AnClark/PDF2Epub_2) | 可用 | 中文扫描版 | ❌ 无公式 | — | Tesseract + 标题层级/自然段重建(缩进启发式)。散文向,无公式能力 |
| [venhow/pdf-craft](http://github.es/venhow/pdf-craft) | 镜像 | 同上 | 同上 | — | pdf-craft 的文档镜像,说明 CSS 较全 |

### pdf-craft 的公式/表格渲染配置(关键)

```python
from pdf_craft import PDFPageExtractor, ExtractedTableFormat, \
                      generate_epub_file, TableRender, LaTeXRender

extractor = PDFPageExtractor(
    device="cuda",
    extract_formula=True,                              # 开 LaTeX-OCR
    extract_table_format=ExtractedTableFormat.HTML,    # EPUB 场景必须是 HTML
)

generate_epub_file(
    from_dir_path=output_dir_path,
    epub_file_path="out.epub",
    table_render=TableRender.HTML,      # 或 CLIPPING(默认,截图)
    latex_render=LaTeXRender.MATHML,    # 或 SVG / CLIPPING(默认,截图)
)
```

- `TableRender`:`HTML` / `CLIPPING`(默认)
- `LaTeXRender`:**`MATHML`** / **`SVG`** / `CLIPPING`(默认截图)
- `SVG` 模式需本地安装 `latex --version` 可用,否则直接报错
- `OCRLevel.OncePerLayout` 可对同页多次 OCR 以抗模糊/漏字(慢但准)

---

## 2. PDF → Markdown / HTML(自己再打包 EPUB)

| 工具 | 星标 | 公式 | 输出 | 许可 | 定位 |
|---|---|---|---|---|---|
| **Docling**(IBM Research) | 20K+ | 中(弱于 Marker) | Markdown / **HTML** / JSON / DocTags | MIT | 最全面:布局分析、阅读顺序、合并单元格、图片内容分类、图表数据、**每个元素带页码+bbox 可溯源**、结构感知切块、官方 MCP server。有可选 GraniteDocling 258M VLM。CPU 默认可用 |
| **Granite-Docling 258M** | — | ✅ | 结构化 HTML | — | **258M 小模型,浏览器/Node 跑 ONNX(Transformers.js),无需 GPU 和 Python**;支持代码、公式、表格、图表、图注关联、bbox OCR |
| **Marker**(datalab-to) | 19K+ | ✅ 最强(Texify) | Markdown / JSON / **HTML** / chunks | GPL-3.0(code)+ Open RAIL-M(权重) | `--use_llm` 可叠加 LLM 提升精度;H100 约 25 页/秒(批量);峰值 5GB VRAM/worker |
| **MinerU**(OpenDataLab) | 30K+ | ✅ | Markdown / JSON | **AGPL-3.0** | 中文与复杂版面最好;硬件支持最广(含昇腾等) |
| **pdf2htmlEX** | — | 保留原样(**非语义 MathML**) | **版面保真 HTML** | GPLv3 | 文字层+精确定位,可搜索可复制;静态 HTML、无 JS 依赖;体积常比 PDF 还小。⚠️ 原仓库 coolwanglu 版**已停止维护**("New maintainers are wanted"),请用活跃 fork [pdf2htmlEX/pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX)(有 Debian 包 / AppImage / Docker,到 0.18.8.rc1) |
| **PyMuPDF4LLM** | 2K+ | ❌ | Markdown | — | 最轻,无模型、无 GPU |
| **MarkItDown**(Microsoft) | — | ❌ | Markdown | MIT | 零依赖最浅,能读 EPUB 但不能写 EPUB |
| **Mathpix Convert API** | 商业 | ✅ 最强 | LaTeX / Markdown / HTML | $0.025/页(4 万页以上 $0.01/页) | 公式领域天花板。封装示例:[ZealousEar/LaTeX-OCR-Document-Forger](https://github.com/ZealousEar/LaTeX-OCR-Document-Forger) |
| [ladislavsulc/pdf-to-html](https://github.com/ladislavsulc/pdf-to-html) | — | ❌ | 语义 HTML + schema.org JSON-LD | — | SEO 向,带 Gradio UI。不是阅读向 |

---

## 3. 架构层最值得抄:BabelDOC 的双向 IR

**[funstory-ai/BabelDOC](https://github.com/funstory-ai/BabelDOC)**(⭐8.4K+, 论文 [arXiv:2605.10845](https://arxiv.org/pdf/2605.10845))

它做的事:把**版面元数据与语义内容解耦成双向中间表示(IR)**,内容随便折腾(翻译、术语表约束、跨页上下文),完了再用自适应排版引擎**"回锚"到原版面**。内置 **公式占位(formula placeholdering)**——即"公式不参与文字处理,原样保留"。

- 这正是上次建议的架构,而且**已经工程化落地**。
- 上层封装:**PDFMathTranslate-next**(⭐26K+),CLI / WebUI / Windows EXE,CPU 可跑。
- ⚠️ 它读的是 **PDF 文字层** + DocLayout-YOLO 版面分析,**不做 OCR**。纯扫描件不适用,得先 OCR 成带文字层的 PDF。
- 它的能力对照表很有参考价值:公式处理 ✅、脚注翻译 ✅、视觉样式保留 ✅、图内文字翻译 ✅、中间表示 ✅、自动上下文术语 ✅。

---

## 4. 公式在 EPUB 里的三种渲染方式

| 模式 | 优点 | 缺点 |
|---|---|---|
| **MathML** | 文本可搜索、可访问(WCAG)、矢量不失真 | **Kindle/KF8 不支持**;各阅读器实现参差(Apple Books 部分忽略 `mrow`/`mtable`) |
| **SVG** | 阅读器支持最广(EPUB2 也能显示)、矢量无损 | 公式不可搜索、不可复制;需本地 LaTeX 渲染 |
| **裁剪原图(CLIPPING)** | 与原书 100% 一致、零额外依赖、零错误 | 不可搜索复制;高 DPI 下体积大、缩放模糊 |

**建议**:正文 MathML + `altimg`(SVG/PNG)+ `alttext` 三层回退;或直接 SVG(牺牲可搜索性换取"任何阅读器都不会坏")。用 `epubcheck` 验证。

---

## 5. 选型建议

1. **先跑 `pdf-craft` 做基线** —— 离需求最近,公式/表格/EPUB 打包/目录一条龙。
2. **正文质量要更高** → 用 **Marker** 或 **Docling** 出 Markdown/HTML,自己用 Pandoc 或 `ebooklib` 打包 EPUB。
3. **复杂版面 / 老扫描件兜底** → **pdf2htmlEX** 出保真 HTML,或公式裁剪图路线。
4. **想上"结构+版面分离"的正规架构** → 直接读 **BabelDOC** 的实现,别从零设计。
5. **评测** → 抄 lgmoneda/pdf-to-epub 的 benchmark 闭环(固定测试集 + 期望项 + LLM 判官);横向榜单看 **OmniDocBench** / **olmOCR-Bench** / **dp-bench**。参考 [dantetemplar/pdf-extraction-agenda](https://github.com/dantetemplar/pdf-extraction-agenda) 的汇总表。
6. **许可** → MinerU **AGPL-3.0**、Marker **GPL-3.0**、pdf2htmlEX **GPLv3** 有传染性;pdf-craft(MIT)、Docling(MIT)、MarkItDown(MIT)、PyMuPDF4LLM 较友好。
7. **版权** → 扫描/购买的书转制分发有法律风险,注意使用范围。

---

## 6. 链接汇总

- https://github.com/oomol-lab/pdf-craft ・ https://pdfcraft.ai ・ demo: https://pdf.oomol.com
- https://github.com/porfanid/pdf2epub
- https://github.com/lgmoneda/pdf-to-epub
- https://github.com/AnClark/PDF2Epub_2
- https://github.com/docling-project/docling ・ https://docling.ai
- https://github.com/datalab-to/marker
- https://github.com/opendatalab/MinerU
- https://github.com/pdf2htmlEX/pdf2htmlEX
- https://github.com/funstory-ai/BabelDOC ・ https://arxiv.org/pdf/2605.10845
- https://github.com/PDFMathTranslate-next/PDFMathTranslate-next
- https://github.com/opendatalab/DocLayout-YOLO
- https://github.com/pymupdf/pymupdf4llm
- https://github.com/microsoft/markitdown
- https://github.com/dantetemplar/pdf-extraction-agenda
- https://mathpix.com/convert
