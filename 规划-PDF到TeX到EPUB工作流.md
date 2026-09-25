# 规划：PDF → TeX → EPUB 工作流 + KaTeX Android 阅读器

> 路线调整：**放弃 MathML，改为输出 LaTeX 源码 + 自建 KaTeX 阅读器**。
> 这个调整是对的 —— 它把公式渲染从「依赖 EPUB 阅读器生态」变成「自己完全可控」，
> 一举绕开了此前调研中最难的那个坑。

---

## 一、为什么这个改动是对的

### 原来的死结

| 方案 | 问题 |
|---|---|
| MathML | Kindle 全系不支持、Apple Books 部分版本忽略 `mrow`/`mtable`、微信读书忽略 |
| 公式转图片 | 保真但不可重排、不可搜索、不可复制 |
| 依赖阅读器带数学 JS | 现成阅读器基本都不带 |

**根因：EPUB 的质量上限由阅读器决定，不由你的 pipeline 决定。**

### 新路线的破局点

把公式**以 LaTeX 源码内联在正文流里**，渲染交给自己的阅读器：

| | 结果 |
|---|---|
| 公式 | **可搜索、可复制、可重排** |
| 渲染质量 | 由 KaTeX 决定（基于 Knuth TeX 的排版），**不再取决于阅读器** |
| 降级 | 没装 KaTeX → 显示可读的 `\(P(x,y)\)` 源码，**不是空白或乱码** |
| 可逆性 | 源码原样保留，将来想转 MathML / SVG / PDF 都行 |

这比 MathML 路线**同时**拿到了更高保真和更高可移植性。

---

## 二、架构

```
                    ┌─────────────────────────────────────┐
   扫描版 PDF ──────►│                                     │
   （无文字层）      │   pdf-craft  +  远程 DeepSeek-OCR    │
                    │                                     │
   数字版 PDF ──────►│   （未来可换 Marker / MinerU 前端）    │
   （有文字层）      └──────────────┬──────────────────────┘
                                   │  .pcex 抽取归档 ← 契约层
                                   │  （公式已是 LaTeX 源码 + bbox + 图片 hash）
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       pcex2x --epub        pcex2x --tex          pcex2x --md
       EPUB3（公式=LaTeX）    全书 .tex               .md
              │
              ▼
       embed_katex.py  →  KaTeX 自包含 EPUB
              │
              ▼
       TeX Reader（Android，Readium + KaTeX）
```

**核心设计：`.pcex` 作契约层。**

它是 pdf-craft 的抽取归档，**不是渲染产物**：

- 公式以 **LaTeX 源码**保存（`\mathrm{d}x`、`Y_{1}(y)` 原样，不是 MathML、不是图片）
- 文本片段带 `bbox` + `page_index`，**可回溯原书版面做 QA**
- 图片按 sha256 独立存放

⇒ 同一份归档可无损再渲染成任意后端，**改参数不用重跑 OCR**。

---

## 三、关键实测数据（20 页样章：《常微分方程教程》第 2 章）

### OCR

| 指标 | 值 |
|---|---|
| 耗时 | **292 秒 / 20 页**（≈14.6 秒/页）→ 全书 300 页约 **70–75 分钟** |
| Token | 18,280 输入 + 21,201 输出 |
| 成本 | 硅基流动 `deepseek-ai/DeepSeek-OCR`，**免费额度** |

### 抽取质量

| 元素 | 数量 |
|---|---|
| 章节 | 11 |
| 段落 | 215 |
| 块级公式 | 130 |
| 行内公式 | 380 |
| 带编号公式 | 51 |
| 插图 | 4 |

### 校验（L0 确定性层）

| 指标 | 值 |
|---|---|
| 公式总数 | **508** |
| **KaTeX 编译失败** | **0** ← LaTeX 语法全部合法 |
| 有问题的 | **8**（全部为真问题，详见第六节） |

### 产物

| 文件 | 大小 |
|---|---|
| `sample.epub`（纯净，公式=LaTeX） | 241 KB |
| `sample-katex.epub`（KaTeX 自包含） | **585 KB** |
| `sample.tex`（可 xelatex 直接编译） | 41 KB |
| `sample.md` | 41 KB |
| `sample.html`（浏览器预览） | 75 KB |

**XHTML 严格 XML 校验：12/12 合法。**

---

## 四、公式在 EPUB 里的表示

这是本项目的核心设计，落在 `pipeline/assets/reader.css`：

```html
<!-- 行内 -->
<span class="math-inline">\(P(x,y)\)</span>

<!-- 块级：编号被拆出来单独右对齐 -->
<div class="math-block">
  <span class="math-body">\[P(x,y)\mathrm{d}x + Q(x,y)\mathrm{d}y = 0\]</span>
  <span class="eqno">(2.15)</span>
</div>
```

三个要点：

1. **定界符包裹**。KaTeX 的 `renderMathInElement` 直接吃这套定界符，零适配成本。
2. **编号分离**。pdf-craft 把编号留在 LaTeX 尾部（`\quad (2.15)`），我们拆出来做成
   `<span class="eqno">`，用 CSS grid 三栏做到「公式居中 + 编号贴右」。
3. **降级友好**。普通阅读器看到的是 `\(P(x,y)\)` —— 略吵，但完全可读，
   比 MathML 源码或空白好得多。

### 踩过的两个坑

| 坑 | 症状 | 解法 |
|---|---|---|
| **XHTML 是严格 XML** | 注入的内联 `<script>` 里裸的 `<` 运算符（`i < nodes.length`）让**整本书打不开** | `//<![CDATA[ ... //]]>` 包裹 |
| **`<ol>` 嵌套手写状态机** | 层级回退时标签配错，`nav.xhtml` 非法 XML | 改用真正的递归 |

---

## 五、三级校验流水线（`pipeline/verify.py`）

### 为什么是三层而不是两层

你的原设计是「Jev 检查 → 可疑的转 LLM 修正」。方向完全正确 ——
**先廉价筛选、只把可疑片段送贵模型**，这是唯一能规模化的形状。

但 Jev 有三条官方自曝的边界，决定了它**不能独自承担第一层**：

| Jev 的限制 | 后果 |
|---|---|
| **纯文本**（看不到图） | 无法判断「公式和原图长得像不像」 |
| **不可靠计数** | 无法判断「括号是否配对」「符号个数对不对」 |
| **不生成文本** | 只能判定，不能修正 |

⇒ 第一层必须拆成「确定性检查」+「Jev 语义判定」两块。

### 分层

| 层 | 实现 | 职责 | 成本 |
|---|---|---|---|
| **L0** | 纯本地 | 花括号配对、`\left\right`、`\begin\end`、空分组；**KaTeX 无头编译**；编号连续性、正文引用存在性 | **0** |
| **L1** | **Jev**（TypeSafe System One） | 「这条公式可疑吗」的语义判定 + 上下文矛盾检测 | **~$0.007/书** |
| **L2** | LLM / **VLM** | 带**原图裁剪** + 上下文重识别并修正 | ~$0.5/书 |

**L0 里最强的一项其实是 KaTeX 无头编译** —— 它是真正的 parser，有一张完整的宏表，
比任何手写白名单都可靠。（我们一开始手写了命令白名单，实测把 `\pm` `\mu` `\xi`
`\gg` `\S` 全误判成「残缺命令」，误报 23 条，最后整项删掉交给 KaTeX。）

### Jev 的正确用法

**端点**：`POST https://api.typesafe.ai/v1/systemone`
**认证**：`Authorization: Bearer $TYPESAFE_API_KEY`
**模型**：`jev-latest`（当前 `jev-1.13.0`）
**定价**：**$0.042 / 百万输入 token，输出免费**
**延迟**：70–500 ms
**限制**：64K token/请求 · **English-first** · 早期访问需排队

三个原语：

| 原语 | 返回 | 我们用它做什么 |
|---|---|---|
| `Noul` | 0–1 的「是」概率 | **`"这条 LaTeX 存在结构异常"`** → 用阈值路由到 L2 |
| `Choice` | 1/N 选项 + 每项概率 | 错误类型分类（编号错/符号错/切分错误/非公式） |
| `Score` | 有序档位上的位置 | 严重度分级，决定人工复核优先级 |

**关键技巧：一次调用打包整章的问题。**

Jev 一次调用可并行回答多个问题、且**加问题不加延迟**，所以不要一条公式一次调用：

```python
questions = {}
for f in batch:
    questions[f"ok_{f.fid}"]  = {"type": "noul", "instructions": JEV_CONSISTENT}
    questions[f"bad_{f.fid}"] = {"type": "noul", "instructions": JEV_SUSPICIOUS}
# 一章 200 条公式 → 1 次请求，全部并行返回
```

Instructions 用**英文**写（Jev 是 English-first），但这不影响它判断中文书里的数学符号。

### 成本估算（300 页书，约 3000 条公式）

| 层 | 调用量 | 成本 |
|---|---|---|
| L0 | 全量 | **￥0** |
| L1 Jev | 约 15 次批量请求（3000 × 350 tok ≈ 1M tok） | **≈ ￥0.3** |
| L2 VLM | 约 300 条可疑项 | **≈ ￥3–5** |
| **合计** | | **≈ ￥5 / 本** |

对比「全书丢给 VLM 重跑一遍」：约 **￥50–150**。省一个数量级。

---

## 六、L0 实测抓到的真问题（8 条）

这批问题**互相印证**，说明检查器是有效的：

### 1. 公式编号识别错（与「引用缺失」精确互补）

```
正文引用了 (2.11)、(2.13)、(2.19)、(2.31) —— 但公式编号里没有
编号跳号：2.10→2.12、2.12→2.14、2.18→2.20、2.30→2.32
编号重复：(2.20) 出现 2 次
```

两条独立检查指向同一批编号 ⇒ **确实错**。
`(2.19)` 缺失 + `(2.20)` 重复 ⇒ 有一条本该是 2.19 的被识别成了 2.20。

### 2. 公式被切分（新发现的缺陷类型）

```
c04-i097  (a\neq 0                     → 1 开 / 0 闭
c04-i114  [a - \epsilon ,a + \epsilon ]) → 1 开 / 2 闭
```

两条**加起来正好平衡** ⇒ 原始公式被 OCR **切成了两半**。

这不是「公式写错」，是「**切分错误**」，需要的是合并而不是修正 ——
属于 L2 里要单独处理的一类。

### 3. 非公式被误判为公式

```
\S 2.3 / \S 6.3 / \S 2.4.3   ← 这些是章节符号「§」，不是数学公式
```

会在 EPUB 里渲染成一个奇怪的公式，应该在生成端转成正文文本。

---

## 七、Android 阅读器（`android/TexReader`）

### 选型

**不 fork 完整的阅读器 App**，而是**基于 Readium kotlin-toolkit 自建精简 App**：

| 候选 | 结论 |
|---|---|
| Readium kotlin-toolkit | ✅ **选中**。EPUB 生态的工业级底座（解析/分页/目录/CFI 定位全有），Apache-2.0 |
| Anx Reader / Quill / Liseur | 均基于 Readium、但各自绑了 Firebase / RAG / 自建服务，剥起来比重写还费劲 |
| 自己从零写 WebView 阅读器 | EPUB 的坑太多（OPF 解析、CFI、分页），不值得 |

用 **Maven Central 上的 AAR 依赖**，不 clone 整个 kotlin-toolkit 源码编译
（后者首次构建要十几分钟）。

### 结构

```
app/src/main/java/org/texreader/
  MainActivity.kt    书架：SAF 选文件 → 复制到私有目录
  ReaderActivity.kt  打开出版物、安装 EpubNavigatorFragment、监听翻页
  KatexBridge.kt     核心：确保 KaTeX 渲染（带重试）
```

### 关键决策：KaTeX 放 EPUB 里，不靠阅读器注入

`servedAssets` / `TransformingFetcher` / `registerJavascriptInterface`
这些 API 都随 Readium 版本变动（3.x 就改过一轮），而
**「EPUB 引用自己的资源」是 EPUB 规范本身，永远成立**。

内嵌之后：

- 阅读器侧只剩「触发渲染」一件小事，且**跨版本稳定**；
- 同一个 EPUB 在**任何允许 JS 的 WebView 型阅读器**里都能正确显示；
- 代价仅 **+344 KB**（只打包 woff2 字体，砍掉 woff/ttf 的 2.6 MB 冗余）。

阅读器仍保留一层**兜底**：页面加载后用 `evaluateJavascript` 主动检查
`window.__katexDone`，没渲染就补渲染。因为 `evaluateJavascript` 是原生侧发起的，
不受「文档内脚本被禁用」影响。

---

## 八、环境与成本

### 当前机器状态

| 项 | 状态 |
|---|---|
| Python 3.13.5（系统） | ✅ pdf-craft 2.3.1 + epub-generator 0.1.7 已装 |
| Poppler 26.09.0 | ✅ `E:\EPUB\tools\poppler\Library\bin` |
| Node 22.22.2 | ✅ 用于 KaTeX 无头编译校验 |
| OCR | ✅ 硅基流动 `deepseek-ai/DeepSeek-OCR`（免费） |
| GPU | ❌ Intel Arc 核显，**无 CUDA** → pdf-craft 本地 OCR 不可用，只能走远程 |
| **JDK 17** | ❌ **未安装** |
| **Android SDK 35** | ❌ **未安装** |
| **Gradle** | ❌ 未安装 |

### Android 侧还需装什么

| 项 | 体积 | 备注 |
|---|---|---|
| JDK 17 (Temurin) | ~200 MB | |
| Android cmdline-tools + Platform 35 + Build-Tools 35 | ~3–5 GB | 不必装 Android Studio |
| Gradle 8.13 + 依赖缓存 | ~2–3 GB | 首次构建从 Maven Central 拉 Readium |
| **合计** | **~8–10 GB** | **C 盘只剩 19 GB（89% 满），建议装到 E 盘（126 GB 空闲）** |

---

## 九、交付清单

### 流水线 `E:\EPUB\pipeline\`

| 文件 | 作用 |
|---|---|
| `pcexlib.py` | `.pcex` → IR 解析（含公式编号拆分、噪声过滤） |
| `pcex2x.py` | IR → EPUB3 / LaTeX / Markdown / HTML 预览 |
| `embed_katex.py` | 任意 EPUB → KaTeX 自包含版（**对 Marker/MinerU 的产出同样适用**） |
| `html_preview.py` | 生成带 KaTeX 的浏览器预览页 + 失败公式清单 |
| `verify.py` | **三级校验**（L0 确定性 / L1 Jev / L2 LLM-VLM） |
| `assets/reader.css` | EPUB 样式（grid 三栏公式 + 编号） |
| `assets/katex/` | KaTeX 0.18.7 dist（3.1 MB） |

### Android `E:\EPUB\android\TexReader\`

完整可编译项目骨架（Gradle 配置 + 3 个 Kotlin 文件 + 资源 + README）。

### 产物 `E:\EPUB\out\tex\`

纯净 EPUB、KaTeX 自包含 EPUB、TeX 源码、Markdown、HTML 预览、校验报告。

---

## 十、下一步

### 立即可做

1. **看 HTML 预览**，确认 KaTeX 对这批 OCR 出来的 LaTeX 的渲染质量。
2. **跑全书 300 页**（约 70–75 分钟，一次 OCR，之后可反复重渲染）。

### 需要你决定的

3. **装 Android 环境**（~8–10 GB，建议装 E 盘），才能把阅读器编出来跑。
4. **拿 Jev 的 Key**：<https://console.typesafe.ai/settings/keys>（**早期访问需排队**），
   或走 Vercel AI Gateway。拿到后 L1 立刻可用。

### 待补的工作

| 项 | 说明 |
|---|---|
| 公式切分合并 | L0 已能**检出**（相邻公式括号互补），还缺**自动合并** |
| 非公式转文本 | `\S 2.3` 这类应在生成端转成正文，而不是渲染成公式 |
| 编号自动纠错 | 用「正文引用集合」反推正确编号（现在只能报警） |
| epubcheck 门禁 | 本机无 Java，没跑成；建议加进 CI |
| 目录质量 | 现有 heading 识别把 `§2.2` 错列为 `§2.1` 子项、`§2.4.1` 认成 `92.4.1` |
| 数字版 PDF 前端 | 现在统一走 pdf-craft；数字版可接 Marker/MinerU 提高正文质量 |
| 跨页段落合并 | 一个 `<text role="body">` 多 fragment 的拼接策略还比较粗糙 |

### 已知风险

| 风险 | 缓解 |
|---|---|
| Jev 是**早期访问 + waitlist**，定价厂商自测未复现 | L0 不依赖它，独立可用；L1 只是加速器 |
| 阅读器**未编译验证过**（本机无 JDK/SDK） | 已标注两处需按实际 API 调整的点，均在 `ReaderActivity` 一处 |
| 扫描件版权 | 转制分发有法律风险，自用需注意 |
