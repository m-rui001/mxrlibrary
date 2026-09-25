# mathbook PDF 到数学 EPUB 工作流程技术报告

**报告日期：2026-09-25**  
**范围：** E:\mathbook 数学书 PDF 生产线，涵盖目录体检、批量 OCR、断点续跑、pcex 中间格式、EPUB 渲染、公式校验与 GLM-OCR 修正。  
**依据：** 当前代码、运行账簿、批次日志和精修记录。历史数字会随书库及渲染版本变化，以下均注明数据口径。

## 1. 总览

产线把 PDF 正文、公式和图片抽取为结构化 pcex，再由项目渲染器生成 EPUB。公式以 LaTeX 源码保存，阅读器用 KaTeX 排版。OCR 抽取和 EPUB 渲染分层，因此修改公式或版式时，通常只需重渲染，不必再次发送整本 PDF 给 OCR 服务。

主流程：

PDF 体检 → 书单筛选 → 每书独立 worker → pdf-craft 逐页 OCR → 失败页定向重抽 → pcex 归档 → pcex2x 生成 EPUB → EPUB 结构检查和 KaTeX 门禁 → 缺陷定位与 corrections 修正 → 重渲染 → 复测。

关键实现：

- [batch.py](../pipeline/batch.py)：主调度、子进程、manifest、OCR 重抽和 EPUB 生成。
- [ocr_resume.py](../pipeline/ocr_resume.py)：坏页与非法资产清理、整页兜底页统计。
- [pcexlib.py](../pipeline/pcexlib.py)：pcex 解析、IR 结构、公式 ID 和 corrections 安全应用。
- [pcex2x.py](../pipeline/pcex2x.py)：pcex 到 EPUB、TeX、Markdown、HTML 的渲染。
- [check_epub.py](../pipeline/check_epub.py) 与 [check_math.py](../pipeline/check_math.py)：EPUB 结构与公式编译门禁。
- [autopilot.py](../work/autopilot.py)：按队列分批启动 batch.py 并管理运行预算。
- [glmocr_refine.py](../work/glmocr_refine.py)：公式缺陷页重识别和 corrections 三阶段工具。

## 2. 目录与数据职责

| 位置 | 内容与职责 |
|---|---|
| E:\mathbook | PDF 来源库，含根目录散放文件和现有学科目录。|
| E:\EPUB\pipeline | 批处理、pcex 解析与渲染、质量检查脚本。|
| E:\EPUB\out\mathbook | 交付 EPUB，输出目录随来源相对目录组织。|
| E:\EPUB\work\survey.json | PDF 页数、大小、扫描/数字版分类等目录体检结果。|
| E:\EPUB\work\mathbook\manifest.json | OCR 主账簿：每本书状态、页数、EPUB/pcex 路径、用时及公式统计。|
| E:\EPUB\work\mathbook\render_manifest.json | render-only 独立账簿，避免纯渲染任务和 OCR 批次互相覆盖状态。|
| E:\EPUB\work\mathbook\books\<safe-stem> | 每书持久工作区，含 analysing、逐页 XML、图片资产、pcex、渲染目录和 run.log。|
| E:\EPUB\work\mathbook\corrections | 逐书 LaTeX 覆写叠加层，不修改原始 pcex。|
| E:\EPUB\work\glmocr_refine | GLM-OCR 响应缓存、页图、调用账本和阶段日志。|

### 路径依赖

主 manifest 的键是相对于 E:\mathbook 的路径，survey 使用相同的 rel。batch.py 又以该相对路径净化生成工作目录名：斜杠转换为双下划线，非法文件名字符替换，长度最多 110 个字符。因此移动或改名 PDF 会影响 manifest 与 survey 键、页 XML 和缓存目录、pcex 路径、EPUB 输出位置，以及失败公式映射中保存的路径。

当前 E:\mathbook 有 446 个实际 PDF：171 个在根目录，其余分布在 10 个已有目录。survey 中有 447 条记录，其中一条指向当前不存在的 PDF。用户要求先不移动或改名这些书；本报告据此保留现有路径。

## 3. 目录体检与队列构建

### 3.1 Survey

pipeline/survey.py 递归扫描 PDF，采集页数、文件大小、是否加密、PDF 元数据、页面文本量和图像面积等数据，并估算 OCR 时间。书籍类型分为 digital、mixed、scan 或 error。

结果写入 work/survey.json。队列主要使用 rel、pages、kind。survey 是扫描时的快照，来源库增删后必须重新扫描或修复路径记录；当前记录与磁盘少一条的差异需要后续清理。

### 3.2 batch.py 选书

batch.py 根据来源目录建立任务，可由 --only 书单、页数上下限、类型和 --limit 过滤，再按页数排序，默认小书优先。超大书可以用 --chunk-pages 切页段；每段有独立 manifest 项和 EPUB，降低单次中断的损失，但会拆成多个产物。

主线通常以整本书为任务单位。OCR 页内顺序执行，整体吞吐主要通过多书并发实现。

### 3.3 autopilot

work/autopilot.py 从 survey 读取候选项并对照主 manifest：

- 状态为 done 且 EPUB 文件存在的书跳过。
- failed 默认不进入健康队列，避免反复选中后又被 batch.py 跳过而空转。
- 健康队列清空后才转入显式失败重试。
- 每轮书单记录在 work/autopilot_slice.txt，状态在 autopilot_state.json，日志在 autopilot.log。
- 使用锁和心跳防止重复启动。项目记忆明确指出 Windows 进程枚举和日志时间都不能单独作为存活判据；长任务用 work/diag_alive.py 检查锁心跳及 books 工作区实际写盘。

autopilot 默认 jobs=6、每轮 slice=8、本次预算 3 小时。slice 应至少为 jobs 的两倍，保持线程池有待运行书。当前代码按本轮最大书页数估算单书超时：约每页 22 秒加 40 分钟余量，且至少 60 分钟；不能让超时随预算尾段缩小。

## 4. 单书 OCR、重试与恢复

### 4.1 OCR 配置与并发

batch.py 使用 pdf-craft 调用 SiliconFlow 的 DeepSeek-OCR。凭证从环境变量或本机 secret 文件读入，由父进程放入 worker 环境变量；密钥不出现在命令行、manifest 或日志中。

生产页图为 300 DPI，OCR 请求使用 grounding 提示，单请求超时默认 300 秒，ocr_size 默认 gundam。单书内逐页运行，因此并行度加在多本书上。每书作为独立子进程，调度器可单独对其计时并在超过单书上限时终止，某书的客户端死等不会永久占住线程槽。

代码默认 jobs=2，autopilot 默认 jobs=6，启动参数可以覆盖。历史压测显示 48 路获得高总吞吐，64 路因服务端排队延迟增长而回落；本次最后收尾批次使用 16 路。不要把历史峰值当成任何机器和日期都稳定的吞吐保证。

### 4.2 页级断点续跑

analysing 目录必须持久保留。pdf-craft 根据 page_N.xml 和 page_N.failed 判断页面是否已完成：

- XML 存在且无 failed 标记：该页跳过。
- OCR 或页面渲染失败：写入 failed 标记，下一轮重做该页。
- 每轮重新生成完整 pcex；中间归档用 round 临时文件，成功后用原子替换覆盖正式文件。

默认允许单页 OCR 出错后继续整本转换。失败页会退化成 300 DPI 整页图，EPUB 因而可以生成，但该页失去可搜索文本和公式。batch.py 随后根据失败事件重抽这些页面。停止条件为失败页清零、重抽轮数耗尽，或连续多轮无改善。当前默认最多 4 轮、连续 2 轮无改善就停。所有目标页都没有可用 OCR 时会报 NoUsableOCRPagesError。

Job 的 first/last 使用 0 起始页序，而 pdf-craft page_indexes 使用 1 起始。batch.py 在提交时加 1，防止整体错一页或漏掉末页。

### 4.3 开始新一轮前的清理

续跑假设磁盘上的页文件完整，但 pdf-craft 的跳过条件只检查文件存在，不解析 XML。超时可能留下半截 page_N.xml；这类文件会一直被跳过，随后整书在解析章节时失败。ocr_resume.scrub_truncated_pages 会在每轮开始前删除这些坏页。

图片资产同样需要清理。pdf-craft 会遍历 extraction/assets；除 64 位小写十六进制文件名加 .png 外，任何成员都可能使整本书打包报 invalid asset member。孤儿 .png.temp 由 scrub_dirty_assets 在 OCR 开始前删除。若从外部清理正在运行的目录，必须用年龄阈值保护尚在写入的临时文件。

### 4.4 运行环境补丁

batch.py worker 启动时加载若干补丁：

- OCR 内容保存 XML 前清洗非法控制字符，例如会破坏 XML 的 0x08。
- Windows 文件改名对瞬时权限拒绝有限重试，必要时复制回退。
- 单书输出日志写失败只记警告，不允许牵连同批已完成的其他书。

长任务使用系统 Python；项目记录指出只有系统 Python 安装了 pdf-craft。换 Python 解释器可能造成整批 ModuleNotFoundError。

## 5. pcex 与 EPUB 渲染

### 5.1 pcex：抽取和渲染之间的稳定层

pcex 是 ZIP 形式的结构化抽取归档，当前格式版本为 3。主要内容：

- manifest.json：文档元数据。
- pages.xml：页序和页面尺寸。
- toc.xml：目录节点、页码、顺序和层级。
- chapters/*.xml：标题、正文、块级公式、行内公式和图片结构。
- assets：图片和公式裁剪图，按内容哈希存储。

正文和公式带页码、顺序或 bbox，可用于定位和抽查；公式原始记录为 LaTeX。pcexlib 将章节转换成 Chapter、Heading、Para、MathBlock、Figure、TextRun 和 MathRun 等 IR。pcex 不等于最终 EPUB，是 OCR 输出与渲染端之间的契约。只要 pcex 已存在，就能不重 OCR 地调整排版和公式修正。

### 5.2 结构归一化

pcexlib 负责公式文本清理、公式编号拆分、邻接块级公式缝合、图片占位符处理和噪声过滤。章节编号从 1 起；块级公式 ID 为 cNN-mNNN，行内公式 ID 为 cNN-iNNN。审核器和渲染器共用 iter_math，避免两处各自编号后漂移。

公式 corrections 可写为 fid 到修正值的映射。带 original 的条目会在应用时核对当前 LaTeX，比较时忽略空格差异。未找到 fid 或原文已变化时跳过并报警，避免位置 ID 漂移后悄悄改到另一条公式。

### 5.3 EPUB 生成

batch.py 调用 pcex2x.py 生成 EPUB，并传入书名、语言和 trace。XHTML 中公式仍带阅读器识别的 LaTeX 定界符；trace 会附上 data-src-page，供问题回溯。渲染器从 pcex 提取图像资产并打入 EPUB。

渲染时可以用 KaTeX 预编译公式。块级公式若编译失败，可以退回 pcex 中的原裁剪图；行内公式保留文本回退。corrections 先应用，再做失败公式降级，因此修好的公式不会被原图覆盖。

纯渲染路径写入独立 render_manifest.json；OCR 主账簿仍记录实际抽取状态。这样不会因为渲染任务与 OCR 批次同时读写整份 JSON 而丢记录。

## 6. EPUB 与内容正确性检查

### 6.1 结构检查

check_epub.py 检查 XHTML 是否可解析、OCR 图片占位符是否漏入正文、行内/块级公式定界符计数，以及 LaTeX begin/end 环境是否闭合。它关注格式和明显残留，不负责判断句子或公式的语义。

### 6.2 KaTeX 门禁

check_math.py 从最终 EPUB 的 XHTML 中提取 math-inline 与 math-body，解码 HTML entity，去除公式定界符后调用与阅读器相同的 KaTeX build。行内公式和 display 公式按各自模式编译。

KaTeX 失败能揭示不存在的命令、括号或环境不配对等语法问题。**KaTeX 编译通过不等于内容正确**：错变量、错符号、漏掉指数或把公式读成另一条合法公式，都可能照常编译。语义正确性要单独审核；必要时用文本上下文发现异常，并以页码/bbox 对应的原图裁剪确认高风险公式。

### 6.3 多层校验

verify.py 还提供实验性层次：

- L0：括号、环境、KaTeX 等确定性检查，不依赖模型。
- L1：Jev 对公式是否可疑做分数筛选，只用于形成候选，不负责修正。
- L2：LLM/VLM 结合原图裁剪和上下文重识别。

不是每本书都会自动跑完全部层。汇报时应分别标注“EPUB 已生成”“结构检查通过”“公式可编译”和“内容语义已审核”。

## 7. GLM-OCR 公式精修

### 7.1 三阶段

work/glmocr_refine.py 将精修分为 fetch、patch、apply：

1. fetch：从缺陷账单定位页，以 Poppler 渲染 300 DPI 页图并调用 GLM-OCR；原始响应写入 work/glmocr_refine/cache，后续调整匹配逻辑可复用缓存。
2. patch：将 GLM 公式文本和 bbox 与 pcex IR 公式匹配，生成候选 LaTeX，交给 KaTeX 预校验，然后写入逐书 corrections JSON。
3. apply：对受影响书只运行 pcex2x，重渲染 EPUB；不重跑 OCR，也不修改 pcex。

### 7.2 匹配和保护

块级主策略是 GLM 公式框中心落入 pcex 公式 bbox，容许多个 GLM 框对应一个 pcex 多行公式；备选策略用 IoU。候选会以多种换行拼接方式生成变体，由 KaTeX 选择第一个通过者。行内只处理强缺陷，并要求段落文本框和公式序列可对齐。

模型偶发把反斜杠加在非 ASCII 字符前。LaTeX 控制序列应由 ASCII 构成；这类误转义会剥除反斜杠，并在 strict:false 下让 KaTeX 渲染 Unicode。失败通道缺 bbox 时会从 pcex 参考对象回退取 bbox。

### 7.3 现有精修数据快照

- v1 和 v1.5：累计抓取 1,214 页，约 9.0M tokens；新增 664 条修正、覆盖 137 本，并完成当时受影响 EPUB 的重渲染。
- v2 失败门禁通道的一次快照：1,892 条 KaTeX 失败，涉及 208 本，其中 1,755 行内、137 块级；1,892 条均映射回 pcex fid。
- GLM 抓取累计账本：19,888,277 tokens、3,498 次调用、2,668 次成功、825 次 429、0 次最终请求失败。最后一轮 441 个页面任务中，434 个 OCR 成功、7 个页面渲染失败。预算硬上限 30M tokens，为 50M 套餐的 60%。
- v2 patch 报告新增 460 条 corrections，28 个候选被 KaTeX 拒绝而安全跳过。历史按新增量计算为 1,124 条修正。
- apply3.log 记录重渲染完成 217 本。后续 gate2.log 扫描约 235 本 EPUB，记录 2,120 条 KaTeX 失败；最新失败收集文件扩大到 259 本、2,673 条。书目集合和渲染时间点不同，不能据两者总数直接计算修复率。
- 当前 corrections 目录有 217 个逐书 JSON，共 2,991 条 fid 记录。这个目录累计数含前期记录，统计口径不同于每一阶段报告的新增修正数。
- 最新失败收集文件时间为 2026-09-25 11:53：2,673 条，分布在 259 个 EPUB（2,606 行内、67 块级），映射器全部定位到 fid。此 EPUB 集合大于 9 月 24 日的 208 本快照；总失败数不能直接用于判断修复前后升降。应使用同一批 EPUB 重新采集，做固定集合对照。

## 8. 常见故障与处置

| 现象 | 原因及影响 | 处置 |
|---|---|---|
| Failed to parse XML file | 超时留下半截页 XML，续跑把它当成完成页跳过 | 每轮调用 scrub_truncated_pages，坏页重新 OCR |
| invalid asset member | assets 中残留 .png.temp 等非法成员，整本 pcex 打包失败 | OCR 前清理孤儿资产；外部清理需加年龄保护 |
| 一页 OCR 错误导致整书失败 | pdf-craft 默认遇首错就抛异常 | 生产模式容许整页图降级，然后重抽失败页 |
| XML 控制字符 | OCR 输出中的非法控制字节破坏 XML 解析 | 保存前清洗非法字符 |
| Windows WinError 5 改名失败 | 瞬时文件锁或实时防护阻止 rename | 有限退避重试，必要时复制回退 |
| 两个任务覆盖 manifest | 同时读改写整份 JSON，后写者覆盖新状态 | render-only 使用独立账簿；主账簿使用锁和合并写 |
| 大书被批量超时 | 单书超时未按当前并发下的页速度估算 | 依据本轮最大书页数估算，并留导出和磁盘余量 |
| 移动 PDF 后找不到旧断点 | rel 同时决定账簿键和 books 工作目录名 | 移动前必须迁移工作目录，并同步 survey、manifest、EPUB/pcex 和映射文件 |

错误要按来源区分：OCR 服务失败、进程超时、临时 Windows 权限错误、缓存损坏和 PDF 本身结构缺陷处置不同。ode.pdf 的 NullObject 是已记录的源 PDF 结构异常，不应与普通 OCR 页错误混为一类。

## 9. 当前状态快照

截至 2026-09-25 17:51 的主 manifest：

| 指标 | 当前记录 |
|---|---:|
| survey 记录 | 447 条，合计 183,046 页 |
| 磁盘上实际 PDF | 446 本 |
| survey 路径存在 | 446/447 |
| 主 manifest 条目 | 326 本 |
| done | 323 本、102,597 页 |
| failed | 2 本，251 页和 341 页 |
| pending | 1 本：纯数学教程，491 页 |
| done 记录对应的 EPUB / pcex 文件 | 各 323 个 |
| books 工作目录 | 326 个 |
| 最近一轮收尾批次 | 16 项中 14 成功、2 失败；用时 122.2 分钟 |

两本失败书分别是代数学引论第 1 卷基础代数和第 2 卷线性代数；最新失败为 NoUsableOCRPagesError，前 1–16 页全部无可用 OCR 页。纯数学教程保持 pending，没有被最新批次完成。到 17:51 manifest 和 batch.log 同时更新，autopilot.lock 不存在；本报告不启动新书任务。

实际 PDF 为 446 本，而主账簿 326 条；其余 120 本没有主 manifest 记录。根据此前的停止要求，它们保持未启动，不应将其笼统汇报成“正在排队”。

## 10. 日常运行核对清单

1. 检查 E:\mathbook 与 survey 的路径、数量是否一致；来源库变化后重新 survey。
2. 从 manifest 读取 done/failed/pending；判断长任务使用 diag_alive，不凭旧日志里的失败行判断当前状态。
3. 对每个新批次保存明确书单，用 --only 限定范围；核对 jobs、最大页数、单书超时、OCR 请求超时和重抽参数。
4. 保留 books/analysing 作为断点目录；清理截断 XML 和孤儿资产时不要触碰运行中的临时文件。
5. 对照 manifest、EPUB 和 pcex；运行 check_epub.py 和 check_math.py。
6. 将 KaTeX 失败映射回 fid，优先修复结构错误及高影响公式；内容语义另行审核。
7. 修正写入 corrections，带上 original；只重渲染受影响 EPUB。
8. 用同一份 EPUB 集合比较修复前后失败率，并抽查文字和公式内容。不能把编译通过写成已证明语义正确。

命令形态示例，书单需由获准的任务生成：

    python pipeline\survey.py --root E:\mathbook --out E:\EPUB\work\survey.json
    python pipeline\batch.py --root E:\mathbook --out E:\EPUB\out\mathbook --work E:\EPUB\work\mathbook --only E:\EPUB\work\batch.txt --jobs 16 --timeout-min 300 --ocr-timeout 300 --ocr-size gundam
    python pipeline\check_epub.py E:\EPUB\out\mathbook\某本书.epub
    python pipeline\check_math.py E:\EPUB\out\mathbook\某本书.epub
    python work\glmocr_refine.py fetch --from-failures
    python work\glmocr_refine.py patch --from-failures
    python work\glmocr_refine.py apply

## 11. 统计解释与报告边界

out\mathbook\PROGRESS.md 是旧快照，不能替代当前 manifest。survey 有一条失效路径，需在下次完整目录扫描前重新生成或清理。不同日期 KaTeX 总失败数覆盖的 EPUB 集合不同，v2 效果应在固定集合上对照；最新 2,673 条是采集结果，不是语义错误总数。

本报告只整理技术流程和现状。按照用户“先别动了”的要求，没有移动、重命名或重新分类 E:\mathbook 中的源文件。


## 附录：相关 Python 源文件全文

以下为报告整理时工作区中的源码快照。代码块中的内容按原文件全文附入；密钥仅由程序运行时读取，本附录未包含密钥值。

### 1. pipeline/batch.py

```python
#!/usr/bin/env python3
"""batch.py — 把一整个目录的 PDF 批量转成 EPUB（公式为 LaTeX，交给 KaTeX 渲染）。

为什么是「外部子进程池」而不是线程池
------------------------------------
pdf-craft 的 `ExtractionOptions` **没有并发参数**，一次 `extract_pdf` 就是一条串行
流水线，页与页之间顺序 OCR。所以吞吐只能靠「同时跑多本书」来堆——这是唯一的杠杆，
也是把 31.8 天的串行工期压下来的唯一办法。

用子进程而不是线程，是为了两个别处换不来的东西：

1. **单本超时可真杀**。一本 1962 页的书要 8 小时，线程杀不掉，子进程能 kill。
2. **一本烂书不能拖垮全局**。OCR 客户端偶发死等；线程池会被一个卡住的线程占死
   一个槽位且无法回收。

代价是每本书多付一次解释器 + import 的开销（数秒），相对于书本身的几分钟到几小时
可以忽略。

断点续跑
--------
状态写在 `manifest.json`：每本书一开工就落一次「running」，收工改写「done/failed」。
重跑时凡是 `done` 且产物仍在的直接跳过。所以这个脚本可以随时 Ctrl-C，第二天接着跑。

对超大书建议 `--chunk-pages 400`：把一本书切成若干页段，每段是独立的 manifest 条目、
独立的 EPUB。牺牲「一本书一个文件」，换来「中断只损失一段」——对 8 小时级别的书，
这个交换非常划算。切段产物后续可以再用别的工具合并。

密钥
----
只从环境变量 `SF_API_KEY` 或 `C:\\Users\\hp\\.workbuddy\\secrets\\siliconflow.key`
读取，**绝不写进 manifest、日志或任何输出文件**。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ocr_resume import fallback_stats, scrub_dirty_assets, scrub_truncated_pages

PY = sys.executable
HERE = Path(__file__).resolve().parent
KEY_FILE = Path(r"C:\Users\hp\.workbuddy\secrets\siliconflow.key")
KEY_DIR = KEY_FILE.parent
KEY_GLOB = "siliconflow*.key"
POPPLER = r"E:\EPUB\tools\poppler\Library\bin"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-OCR"

ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESULT_TAG = "@@RESULT@@"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def api_keys() -> list[str]:
    """全部可用的 SF key（`siliconflow.key` / `siliconflow2.key` / …），去重保序。

    为什么需要多把 —— 2026-09-23 实测的数字链：

      * 生产里 pdf-craft **逐页串行**，而单页要 **12~22 秒**（300 DPI 页图 +
        `<|grounding|>` 提示词 ⇒ completion ≈ 1400 tok）。autopilot 日志里每本
        书都印着真实单价：11.69 / 12.68 / 12.93 / 13.73 / 15.17 s/p。
      * 于是「单进程产能 = 60/延迟 ≈ 4 页/分」，**唯一**的提速杠杆是并发进程数。
      * `_bench_real.py` 用生产同款载荷实测：1 路 13.8s → 12 路（单 key）29.4 页/分
        → **24 路（两把 key 各 12）47.9 页/分，24/24 成功、零 429**。
        全程无一次限流 ⇒ 瓶颈纯是「每请求固定 20 秒」，加并发就线性加吞吐。

    所以并发要摊到多把 key 上，避免单 key 的速率闸门。
    """
    ks: list[str] = []
    k = (os.environ.get("SF_API_KEY") or "").strip()
    if k:
        ks.append(k)
    for p in sorted(KEY_DIR.glob(KEY_GLOB)):
        try:
            v = p.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if v and v not in ks:
            ks.append(v)
    return ks


def api_key() -> str:
    """单 key 取值（仅用于「到底有没有凭证」的启动自检）。"""
    ks = api_keys()
    return ks[0] if ks else ""


def safe_stem(name: str, limit: int = 110) -> str:
    """Windows 文件名净化：保留中文，替换非法字符，压缩空白，限长。"""
    s = ILLEGAL.sub("_", name)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > limit:
        s = s[:limit].rstrip(" .")
    return s or "book"


def has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def load_manifest(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_manifest(path: Path, data: dict) -> None:
    """先写临时文件再替换：中途断电不会留下半个 JSON 把进度全毁掉。

    并发安全：与 promote_reocr / promote_mineru 同时改主 manifest 时，靠跨进程锁
    + 「重读盘 → update 合并 → 原子替换」避免互相抹条目（见 manifest_lock.py）。
    """
    try:
        from manifest_lock import merge_write
    except Exception:  # noqa: BLE001
        merge_write = None
    if merge_write is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return
    merge_write(path, data)


class Log:
    """自己写日志文件，不走 shell 重定向。

    PowerShell 的 `*>> file` 会把子进程的 UTF-8 按控制台代码页重新编码，
    中文一律变成乱码，还可能混进 NUL 字节让日志变成「二进制不可读」。
    让 Python 自己以 UTF-8 落盘，读日志的一方就再也不用跟编码打架。
    """

    def __init__(self, path: str | None) -> None:
        self._f = open(path, "a", encoding="utf-8") if path else None
        if self._f:
            self._f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self._f.flush()

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self._f:
            self._f.write(msg + "\n")
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


# ---------------------------------------------------------------------------
# 单本处理（在 worker 子进程里跑）
# ---------------------------------------------------------------------------


def inspect_epub(epub: Path) -> dict:
    """打开型门禁：每个 XHTML 必须是良构 XML，并数一下公式定界符。

    一个畸形的公式会让整本书在严格 EPUB 阅读器里**打不开**——浏览器却毫无反应，
    所以这一步不能省。定界符计数则是给「KaTeX 真的有事可做」提供证据。
    """
    import xml.etree.ElementTree as ET
    import zipfile

    out = {"xhtml": 0, "malformed": 0, "first_error": "", "inline_math": 0, "display_math": 0}
    with zipfile.ZipFile(epub) as z:
        for name in z.namelist():
            if not name.endswith(".xhtml"):
                continue
            out["xhtml"] += 1
            raw = z.read(name).decode("utf-8", "replace")
            try:
                ET.fromstring(raw)
            except Exception as e:  # noqa: BLE001
                out["malformed"] += 1
                if not out["first_error"]:
                    out["first_error"] = f"{name}: {e}"[:200]
            out["inline_math"] += raw.count("\\(")
            out["display_math"] += raw.count("\\[")
    return out


def resolve_pdf(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def page_count(path: Path) -> int:
    """体检没覆盖到这本书时的兜底：直接问 PDF 要页数。

    这件事不能含糊——页数为 0 会让 `page_indexes` 退化成「只抽第 1 页」，
    产出一本 1 页的「书」，而且 manifest 会把它记成成功。
    """
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # type: ignore
    try:
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001
        return 1


@dataclass
class Job:
    """一本书的一段页（不切段时就是整本）。"""

    rel: str                 # 相对 --root 的路径，manifest 的键
    pdf: Path
    first: int = 0           # 0-based，含
    last: int = 0            # 0-based，含；整本时为 pages-1
    pages: int = 0

    @property
    def chunked(self) -> bool:
        return not (self.first == 0 and self.last == self.pages - 1)


def extract_until_clean(
    *,
    craft,
    pdf: Path,
    pcex: Path,
    analysing: Path,
    ocr_dir: Path,
    max_rounds: int,
    stall_limit: int,
    make_opts,
) -> tuple[list[dict], list[int], list[str]]:
    """反复 `extract_pdf` 直到**没有整页兜底页**，返回 (每轮丢页数, 最后仍是丢页的页号, 说明)。

    为什么可以反复抽而不重付全本的钱
    ------------------------------
    pdf-craft 的 OCR 阶段自带页级断点续跑（`pdf/ocr.py:130`）：

        if page_N.xml 存在 and page_N.failed 不存在: → SKIP

    所以第二轮只重抽第一轮失败的那些页，成功的页一页都不重跑；而每一轮结束时
    `extract_pdf` 都会重新生成一份**完整**的 .pcex（chapters 阶段读 `ocr/` 里全部
    `page_*.xml`）。于是循环是严格收敛的：成功页单调增、丢页单调减、空集即收敛。

    终止条件（三个，缺一个都可能变成烧钱机器）
    ------------------------------------------
    1. 丢页归零 —— 正常出口；
    2. `max_rounds` 用完 —— 硬上限；
    3. 连续 `stall_limit` 轮**一页都没救回来** —— 说明这些页是硬骨头（页面本身
       渲染不出来、或内容 OCR 不出来），再抽也一样，停手并让 manifest 记下残页。

    前提：`analysing` 必须是磁盘上的固定目录。pdf-craft 默认用 TemporaryDirectory，
    跑完即删 —— 没有 `page_N.xml` 就没有续跑，每轮都要从头抽全本，这个循环就没有意义。

    为什么每轮写到**另一个文件名**
    ----------------------------
    pdf-craft 的 `PDFCraftExtraction.export`（`document/package.py:187`）会拒绝覆盖
    已存在的 .pcex，直接抛 `FileExistsError: PDFCraftExtraction already exists`。
    所以第二次 `extract_pdf` 不能写同一个路径 —— 每轮写 `X.roundN.tmp.pcex`，
    最后才 `os.replace` 到正式路径。

    这顺便修掉一个**原来就存在**的坑：`--no-resume` 强制重做一本已经抽过的书时，
    正式路径上正躺着上次的 .pcex，于是老代码必然在 `export` 这一步炸掉、这本书
    永远没法重抽。用 os.replace 收尾之后 `--no-resume` 才真的可用。
    """
    history: list[dict] = []
    notes: list[str] = []
    failed: list[int] = []
    prev: set[int] | None = None
    stalled = 0
    last_tmp: Path | None = None

    for rnd in range(1, max(1, max_rounds) + 1):
        # 先清坏页：被超时截断的半截 XML 会被续跑判据「永久跳过」，还会让 chapters
        # 阶段抛 Failed to parse XML file，整本从此必败。
        scrubbed = scrub_truncated_pages(ocr_dir)
        if scrubbed:
            notes.append(f"[reocr] 清掉被截断的 page xml（将重新 OCR）：{scrubbed[:12]}")

        # 同一个道理的第二类残渣：`assets/` 里的 `*.png.temp`（`AssetHub.clip()` 被硬杀时
        # 停在「写完 temp、还没 rename」之间）。pdf-craft 打包时遍历 assets，**任何**
        # 非「64 位 hex 的 .png」成员都让**整本书**抛 `invalid asset member` —— 不是跳过
        # 那张图。而 `analysing/` 是持久续跑目录，所以不在这里清掉，这本书会**永久**
        # 每轮选中、每轮失败，白占一个并发槽（2026-09-23 硬杀 24 worker 后实测 3 本中招）。
        dirty = scrub_dirty_assets(analysing)
        if dirty:
            notes.append(
                f"[reocr] 清掉 assets/ 非法残渣（不清则打包阶段整本 ValueError）：{dirty[:8]}"
            )

        tmp = pcex.with_suffix(f".round{rnd}.tmp.pcex")
        tmp.unlink(missing_ok=True)
        failed.clear()
        craft.extract_pdf(pdf, str(tmp), make_opts(failed), analysing_path=str(analysing))
        last_tmp = tmp
        history.append({"round": rnd, "degraded": len(failed)})

        if not failed:
            break

        cur = set(failed)
        stalled = stalled + 1 if (prev is not None and not (prev - cur)) else 0
        prev = cur
        if stalled >= max(1, stall_limit):
            notes.append(f"[reocr] 连续 {stalled} 轮无改善，停手；残页 {sorted(cur)[:12]}")
            break

        # 把这一轮仍失败的页写成标记，下一轮它们才会被重抽。
        # （pdf-craft 自己也会写；这里补一遍是为了「失败原因不是 OCRError」的情况，
        #  比如整页渲染失败——那时它不会写标记，我们得替它写。）
        for idx in cur:
            (ocr_dir / f"page_{idx}.failed").write_text("degraded", encoding="utf-8")

    # 收尾：最后一轮（也就是丢页最少的那一轮）成为正式产物。
    # `os.replace` 是原子的，且在 Windows 上也能覆盖已存在的文件——不用它就得先
    # 删掉旧 .pcex 再搬，中间那一刻断电就没有任何产物了。
    if last_tmp is not None and last_tmp.is_file():
        os.replace(last_tmp, pcex)
        for junk in pcex.parent.glob(f"{pcex.stem}.round*.tmp.pcex"):
            junk.unlink(missing_ok=True)

    return history, sorted(set(failed)), notes


def _patch_asset_hub() -> None:
    """让 pdf-craft 的 `AssetHub.clip()` 不再因为「重命名被拒」而毁掉整本书。

    原实现（`pdf_craft/common/asset.py:18`）是**先查后改**：

        if target_path.exists(): temp_path.unlink(); return image_hash
        temp_path.rename(target_path)

    在 Windows 上 `Path.rename` → `MoveFileExW(src, dst, 0)`，只要下面任一条成立，
    它就以 **ERROR_ACCESS_DENIED(5)** 失败 —— 注意**不是**重名错误 183：

      - 目标已存在、且此刻正被另一个句柄打开；
      - 目标已存在且带只读属性；
      - 源文件刚刚写完，正被实时防护 / 搜索索引器扫描（批量产出 PNG 时最常见）。

    后果是**一张图的裁切失败 = 整本书作废**，前面已经付过钱的 OCR 全部白烧。
    实测一本 312p 的书会在 `?s/p`（零页完成）就死掉；一次 8 本的书单里 5 本这么没的：
    Fundamentals of ODE / 流形上的分析 / 高观点下的初等数学1 / Riemannian Geometry /
    Visual Group Theory。

    修法：重命名失败后**回头再看一次目标在不在** —— 在，就是「同一张图已经建好了」，
    那是成功而不是失败（`clip()` 的语义本来就是按内容去重地存图并返回 hash）；
    真的不在才退避重试，重试用尽再把异常抛出去（保持原行为，不吞真错）。
    """
    try:
        from pdf_craft.common.asset import AssetHub
    except Exception:  # noqa: BLE001
        return

    import uuid as _uuid

    def clip(self, image, det):  # noqa: ANN001, ANN201
        cropped_image = image.crop(det)
        self._asset_path.mkdir(parents=True, exist_ok=True)
        temp_path = self._asset_path / f"{_uuid.uuid4().hex}.png.temp"
        try:
            cropped_image.save(temp_path, format="PNG")
            image_hash = self._calculate_file_hash(temp_path)
            target_path = self._asset_path / f"{image_hash}.png"
            delay = 0.02
            for attempt in range(9):
                if target_path.exists():
                    return image_hash
                try:
                    temp_path.rename(target_path)
                    return image_hash
                except OSError:
                    # 复查：目标已存在 ⇒ 别人（或别的进程、或上一轮）已经建好同一张图
                    if target_path.exists():
                        return image_hash
                    if attempt == 8:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 0.5)
            return image_hash
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    AssetHub.clip = clip


def _patch_save_xml() -> None:
    """OCR 模型偶发在输出里夹 XML 非法控制字节（实测 0x08，2026-09-23 华章69概率与计算
    因此整本书 extract failed：'Failed to parse XML file: ...page_345.xml'）。
    pdf-craft 的 save_xml 原样落盘 ⇒ 下次 extract 阶段 ElementTree 拒收。
    在写盘前对 element 就地消毒（text/tail/属性中的非法控制字节删除，内容无损）。
    各调用方都是 from-import 拿函数对象，必须逐模块重绑才生效。
    batch.py 每轮新起子进程自动生效，无需重启产线。"""
    import importlib
    import xml.etree.ElementTree as ET

    from pdf_craft.common import xml as _xmlmod

    illegal = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    _orig = _xmlmod.save_xml

    def save_xml(element, file_path):  # noqa: ANN001
        s = ET.tostring(element, encoding="unicode")
        if illegal.search(s):
            for el in element.iter():
                if el.text and illegal.search(el.text):
                    el.text = illegal.sub("", el.text)
                if el.tail and illegal.search(el.tail):
                    el.tail = illegal.sub("", el.tail)
                for k, v in el.attrib.items():
                    if illegal.search(v):
                        el.attrib[k] = illegal.sub("", v)
        _orig(element, file_path)

    _xmlmod.save_xml = save_xml
    for name in ("pdf.ocr", "document.package", "extractor.chapter.generation",
                 "extractor.toc.analysing", "pdf.furniture"):
        try:
            m = importlib.import_module(f"pdf_craft.{name}")
            if hasattr(m, "save_xml"):
                m.save_xml = save_xml
        except Exception:  # noqa: BLE001
            pass


def _patch_file_ops() -> None:
    """把「写完就改名」这类操作变成**不会因为 Windows 拒绝访问而毁掉整本书**。

    为什么不是一个一个补：pdf-craft 里有**三处**同样的「先查后改 + temp 改名」：

      * `common/asset.py`  `AssetHub.clip`  裁切图（temp 名是随机 uuid）
      * `common/xml.py`    `save_xml`       `ocr/page_N.xml`（temp 名是 `page_N.xml.tmp`）
      * `llm/runtime.py`   LLM 缓存落地（`*.txt`）

    三处都是 `if 目标.exists(): 删临时并返回  else: rename`。Windows 上
    `MoveFileExW` 只要**源刚写完、正被实时防护/索引器扫描**就以
    `ERROR_ACCESS_DENIED(5)` 失败 —— 实测**连目标不存在时也一样**（查过失败现场：
    `analysing/ocr/page_82.xml` 根本没生成、目录里 0 个只读文件、无锁）。
    异常一路上抛 ⇒ **一张图 / 一个 page.xml 失败 = 整本书作废**。

    我第一版只补了 `AssetHub`（`_patch_asset_hub`），结果 `Riemannian Geometry`
    死在**没补的** `save_xml` 上 —— 逐个补必然漏，所以这里**在 os 层全局兜住**。
    （`pathlib.Path.rename/replace` 内部就是调 `os.rename/os.replace`，一并换掉更保险。）

    兜底顺序：
      1. 重试 10 次，退避 0.05s → 2s；
      2. 仍失败 → **直接读源写目标**（create+write 不需要 rename 那套替换权限，
         对「源被扫描」这类最有效），成功后删临时；
      3. 写也失败、但目标已存在且**大小一致** → 视为成功（同图/同页，内容本就相同）；
      4. 都不行才抛原异常（保持原行为，不吞真错）。

    每次走到第 2/3 步都往 `work/_rename_retry.log` 记一行 —— 否则**无法事后确认补丁
    到底有没有生效**（第一版就没留证据，白信了"已修好"）。
    """
    log_path = pathlib.Path(__file__).resolve().parents[1] / "work" / "_rename_retry.log"
    _log_lock = threading.Lock()

    def _note(kind: str, src, dst, how: str) -> None:
        try:
            with _log_lock:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%m-%d %H:%M:%S')} [{kind}] {how} "
                            f"{pathlib.Path(src).name} -> {pathlib.Path(dst).name}\n")
        except Exception as exc:  # noqa: BLE001
            # 写不进日志文件时**必须**换个痕留下：2026-09-23 的失败现场里，
            # 连 _rename_retry.log 本身都被拒写（同一个瞬时锁），
            # 于是事后完全无法判断补丁究竟跑没跑 —— 白白怀疑了半天补丁。
            try:
                sys.stderr.write(
                    f"[rename-retry] [{kind}] {how} "
                    f"{pathlib.Path(src).name} -> {pathlib.Path(dst).name} "
                    f"(日志文件也写不进：{type(exc).__name__} {exc})\n"
                )
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass

    def _guarded(orig, kind: str):
        def wrapper(src, dst, *a, **kw):
            delay = 0.05
            last: OSError | None = None
            for attempt in range(10):
                try:
                    return orig(src, dst, *a, **kw)
                except OSError as e:
                    last = e
                    if False:  # ★必须跑满 10 次（见上方文档）。旧代码是 `attempt >= 2` 才 break，
                               # 即 3 次共 0.15s 就放弃 —— 而实时防护扫完一个刚写的文件要 1~3s，
                               # 于是「重试」形同虚设。2026-09-23 因此白丢 14 本书（含 OCR 已付的费）。          # 连续 3 次才认为不是瞬时抖动
                        break
                    time.sleep(delay)
                    delay = min(delay * 2, 2.0)
            # ---- 兜底 2：直接读源写目标 ----
            try:
                data = pathlib.Path(src).read_bytes()
                with open(dst, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.unlink(src)
                except OSError:
                    pass
                _note(kind, src, dst, f"copy-fallback({type(last).__name__})")
                return None
            except OSError as e2:
                # ---- 兜底 3：目标已在且大小一致 ----
                try:
                    if (pathlib.Path(dst).exists()
                            and pathlib.Path(dst).stat().st_size
                            == pathlib.Path(src).stat().st_size):
                        _note(kind, src, dst, "dst-equal-size")
                        return None
                except OSError:
                    pass
                _note(kind, src, dst, f"GIVE-UP({type(e2).__name__})")
                raise last
        return wrapper

    os.rename = _guarded(os.rename, "os.rename")
    os.replace = _guarded(os.replace, "os.replace")
    pathlib.Path.rename = lambda self, target, *a, **kw: os.rename(self, target)
    pathlib.Path.replace = lambda self, target, *a, **kw: os.replace(self, target)


def run_single(a: argparse.Namespace) -> int:
    """worker：处理一本书（或一段），把结果 JSON 打到 stdout。"""
    try:
        from pdf_craft import (
            DeepSeekOCRVendorConfig,
            DefaultPDFHandler,
            ExtractionOptions,
            OCREventKind,
            PDFCraft,
            PDFOptions,
        )
    except ModuleNotFoundError as exc:
        # 这条报错几乎只有一个成因：启动 worker 的 `sys.executable` 里没装 pdf-craft。
        # 本机有两个 Python，**只有系统那个装了**（managed 那个没装），而裸 `python`
        # 在 PATH 上先解析到 managed —— 于是「每一本」都在这里炸掉。可惜 batch.py 的
        # 父进程只用 worker stderr 的**末 400 字符**当 error 字段，主线看到的是一段
        # `Traceback ... line 786` 垃圾，极易误判成"这些书有问题"（我本人就误判过一次，
        # 白白去查了半天的 analysing 目录权限）。
        # 这里直接把它变成一行可行动的 JSON。
        print(json.dumps({
            "status": "failed",
            "stage": "env",
            "error": f"{type(exc).__name__}: {exc}；sys.executable={sys.executable}"
                     r"；请用 C:\Users\hp\AppData\Local\Programs\Python\Python313\python.exe 启动",
        }, ensure_ascii=False))
        return 1
    _patch_asset_hub()
    _patch_file_ops()
    _patch_save_xml()

    key = api_key()
    if not key:
        print(json.dumps({"status": "failed", "error": "no SF_API_KEY"}))
        return 1

    job = Job(
        rel=a.run_single,
        pdf=(Path(a.root) / a.run_single).resolve(),
        first=a.chunk_first,
        last=a.chunk_last,
        pages=a.chunk_pages_total,
    )
    stem = safe_stem(Path(job.rel).stem)
    if job.chunked:
        stem = f"{stem}__p{job.first + 1}-{job.last + 1}"

    workdir = Path(a.work) / "books" / safe_stem(job.rel.replace("\\", "/").replace("/", "__"))
    if job.chunked:
        workdir = workdir / f"p{job.first + 1}-{job.last + 1}"
    workdir.mkdir(parents=True, exist_ok=True)
    pcex = workdir / f"{stem}.pcex"
    log = workdir / "run.log"

    craft = PDFCraft(
        pdf=PDFOptions(
            ocr=DeepSeekOCRVendorConfig(
                base_url=BASE_URL, api_key=key, model=MODEL,
                timeout_seconds=a.ocr_timeout,
            ),
            pdf_handler=DefaultPDFHandler(poppler_path=POPPLER),
            # 不设 models_cache_path：pdf-craft 校验时会拒绝
            # 「ocr 与 models_cache_path 同时出现」——本地模型缓存只属于本地 OCR。
        )
    )

    # 单页 OCR 失败不该毁掉整本书。
    #
    # pdf-craft 默认（ignore_ocr_errors=False）让第一个 OCRError 直接冒泡：
    # 实测「代数学\线性代数五讲.pdf」第 9 页 layout 失败，这本 100 页的书就整本作废，
    # 前面 8 页的 OCR 全白烧。打开之后坏页降级成「整页图片」继续跑，只有**所有页**
    # 都失败才抛 NoUsableOCRPagesError。
    #
    # 但降级是**整页丢失**，而且完全静默：那一页的文字、公式、可搜索性、朗读全没了，
    # 只剩一张 300 DPI 的整页图（约 0.47 MB，所以成品体积会暴涨）。实测全库
    # 2744/8854 页 = 31.0% 是这么丢的，占全部图片字节的 74%，而产物看上去毫无异常。
    #
    # 所以光「数出来写进 manifest」不够 —— 必须**当场重抽直到不丢**。
    # 这一步之所以可行，靠的是 pdf-craft 自带的页级断点续跑：`ocr/page_N.xml` 还在的
    # 页一律 SKIP，丢页则由 `page_N.failed` 标记出来重抽。于是重抽一轮只付丢页的钱，
    # 而每轮都会重新生成一份**完整**的 .pcex。
    #
    # 两个前提，缺一个这个循环就是白转：
    #   1. `analysing_path` 必须指到磁盘上的固定目录。pdf-craft 默认用
    #      TemporaryDirectory，跑完即删 —— 没有 `page_N.xml` 就没有续跑，
    #      每轮都得从头抽一遍全本。
    #   2. 开工前必须清掉被超时截断的半截 XML（`scrub_truncated_pages`）。
    #      续跑判据只看文件在不在，半截文件会被**永远跳过**，而 chapters 阶段读
    #      `ocr/` 里全部 `page_*.xml` 会直接抛 `Failed to parse XML file`，
    #      整本重抽从此必败。详见 ocr_resume.py。
    analysing = workdir / "analysing"
    ocr_dir = analysing / "ocr"

    def make_opts(failed: list[int]) -> "ExtractionOptions":
        def on_ocr_event(ev) -> None:
            if ev.kind is OCREventKind.FAILED:
                failed.append(ev.page_index)

        return ExtractionOptions(
            # `page_indexes` 是 **1-based**：pdf-craft 的 `ref.page_index = i + 1`
            # （见 `pdf/page_ref.py` 的 `PageRefContext.__iter__`），而 Job 的
            # first/last 是 0-based 闭区间。直接写 `range(job.first, job.last + 1)`
            # 会整体错开一页 —— **最后一页永远抽不到**。全库实测 46/56 本的 pcex
            # 都比原 PDF 少 1 页（差值恒为 1），就是这一行造成的。
            page_indexes={i + 1 for i in range(job.first, job.last + 1)},
            ocr_size=a.ocr_size,
            includes_footnotes=a.footnotes,
            includes_cover=a.cover,
            ignore_ocr_errors=not a.strict_ocr,
            ignore_pdf_errors=not a.strict_ocr,
            on_ocr_event=on_ocr_event,
        )

    history: list[dict] = []
    notes: list[str] = []
    degraded: list[int] = []

    t0 = time.time()
    if a.render_only and pcex.is_file():
        # .pcex 就是流水线的契约层：渲染层改了什么，从这里重跑一次就够，
        # 不必为一个换行符再付几分钟的 OCR。
        extract_s = 0.0
    else:
        try:
            history, degraded, notes = extract_until_clean(
                craft=craft, pdf=job.pdf, pcex=pcex, analysing=analysing,
                ocr_dir=ocr_dir, max_rounds=a.reocr_rounds,
                stall_limit=a.reocr_stall, make_opts=make_opts,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:400]
            log.write_text("\n".join(notes) + f"\n[extract failed] {msg}\n", encoding="utf-8")
            print(json.dumps({"status": "failed", "stage": "extract", "error": msg}, ensure_ascii=False))
            return 1
        extract_s = time.time() - t0

    extract_notes = notes
    reocr_rounds = len(history)
    degraded_first = history[0]["degraded"] if history else 0

    # 渲染：走我们自己的 pcex2x，产出带 LaTeX 定界符的 EPUB（不是 pdf-craft 的 MathML）
    outdir = Path(a.out) / Path(job.rel).parent
    outdir.mkdir(parents=True, exist_ok=True)
    epub = outdir / f"{stem}.epub"
    lang = "zh" if has_cjk(stem) else "en"

    t1 = time.time()
    # 编译失败的块级公式降级为原图：必须给 pcex2x 传 --katex，否则它会静默跳过
    # （找不到 katex.js / node 就直接 return None），块级失败会变成红字 LaTeX 进 EPUB。
    # 用随仓库打包的 katex；目录不存在时也不强行传参，避免无意义的路径错误。
    katex_dir = HERE / "assets" / "katex"
    render_cmd = [
        PY, str(HERE / "pcex2x.py"), str(pcex),
        "--outdir", str(workdir / "render"),
        "--formats", "epub",
        "--stem", stem,
        "--language", lang,
        "--trace",
    ]
    if katex_dir.is_dir():
        render_cmd += ["--katex", str(katex_dir)]
    # 逐书修正叠加层：<work>/corrections/<书 stem>.json 存在就自动套上。
    #
    # 修正记录的是「这本书的 OCR 在这几处读错了」，所以它属于书、不属于批次。
    # 靠人记得手动传 --corrections 是靠不住的：重渲染时漏一次，前面看图逐条
    # 核对的成果就静默退回原样，而产物本身看起来毫无异常。放进约定目录就漏不掉。
    base_stem = safe_stem(Path(job.rel).stem)
    corrections = Path(a.work) / "corrections" / f"{base_stem}.json"
    if corrections.is_file():
        render_cmd += ["--corrections", str(corrections)]
    r = subprocess.run(
        render_cmd,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    render_s = time.time() - t1
    log.write_text(
        ("\n".join(extract_notes) + "\n" if extract_notes else "")
        + (r.stdout or "") + "\n" + (r.stderr or ""),
        encoding="utf-8",
    )
    if r.returncode != 0:
        print(json.dumps({"status": "failed", "stage": "render",
                          "error": (r.stderr or r.stdout or "")[:400]}, ensure_ascii=False))
        return 1

    built = workdir / "render" / f"{stem}.epub"
    if not built.is_file():
        print(json.dumps({"status": "failed", "stage": "render", "error": "no epub produced"}))
        return 1
    shutil.copy2(built, epub)

    info = inspect_epub(epub)

    # 取证：不读自己刚写的记账，直接量成品里有多少张「整页图」。
    #
    # 记账（FAILED 事件）只在**这一次进程**里存在，而整页图是**永久**留在成品里的。
    # 两者对不上就说明有我们没数到的丢页 —— 那正是当初「31% 的页丢了却没人发现」
    # 的成因。所以每次都要交叉验一次，把两个数一起写进 manifest，别让它们悄悄发散。
    fb: dict = {}
    try:
        fb = fallback_stats(pcex)
    except Exception:  # noqa: BLE001
        pass

    result = {
        "status": "done",
        "epub": str(epub),
        "epub_bytes": epub.stat().st_size,
        "pcex": str(pcex),
        "pcex_bytes": pcex.stat().st_size if pcex.is_file() else 0,
        "extract_seconds": round(extract_s, 1),
        "render_seconds": round(render_s, 1),
        "corrections": corrections.name if corrections.is_file() else "",
        "seconds_per_page": round(extract_s / max(job.last - job.first + 1, 1), 2),
        "pages": job.last - job.first + 1,
        "chunk": f"{job.first + 1}-{job.last + 1}" if job.chunked else "all",
        "language": lang,
        "degraded_pages": len(degraded),
        "degraded_indexes": sorted(degraded)[:30],
        # 重抽的经过：第一轮丢多少、抽了几轮、每轮剩多少。丢页归零才算干净。
        "degraded_first_round": degraded_first,
        "reocr_rounds": reocr_rounds,
        "reocr_history": history,
        # 独立取证数（可能比事件数少 1：封面本身可以是一张整页图）
        "fallback_pages": fb.get("fallback_pages"),
        "fallback_total_pages": fb.get("pages"),
        **info,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# 队列构建
# ---------------------------------------------------------------------------


def build_queue(a: argparse.Namespace) -> list[Job]:
    root = Path(a.root)
    survey = {}
    if a.survey and Path(a.survey).is_file():
        for b in json.loads(Path(a.survey).read_text(encoding="utf-8")):
            survey[b["rel"]] = b

    books: list[Job] = []
    for p in resolve_pdf(root):
        rel = str(p.relative_to(root))
        info = survey.get(rel, {})
        pages = int(info.get("pages") or 0)
        kind = info.get("kind") or "unknown"

        if a.kind != "all" and kind != a.kind:
            continue
        if a.max_pages and pages and pages > a.max_pages:
            continue
        if a.min_pages and pages and pages < a.min_pages:
            continue
        if not pages:
            pages = page_count(p)

        if a.chunk_pages:
            lo = 0
            while lo < pages:
                hi = min(lo + a.chunk_pages, pages) - 1
                books.append(Job(rel=rel, pdf=p, first=lo, last=hi, pages=pages))
                lo = hi + 1
        else:
            books.append(Job(rel=rel, pdf=p, first=0, last=pages - 1, pages=pages))

    # 依赖「页数」的切段必须知道真实页数；把体检里缺页数的书过滤掉更安全
    if a.sort == "small":
        books.sort(key=lambda j: j.pages)
    elif a.sort == "big":
        books.sort(key=lambda j: -j.pages)

    if a.only:
        wanted = {line.strip() for line in Path(a.only).read_text(encoding="utf-8").splitlines() if line.strip()}
        books = [j for j in books if j.rel in wanted]
    if a.limit:
        books = books[: a.limit]
    return books


def key_of(job: Job, chunked: bool) -> str:
    return f"{job.rel}#p{job.first + 1}-{job.last + 1}" if chunked else job.rel


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="批量 PDF → EPUB（LaTeX 公式）")
    ap.add_argument("--root", default=r"E:\mathbook")
    ap.add_argument("--out", default=r"E:\EPUB\out\mathbook")
    ap.add_argument("--work", default=r"E:\EPUB\work\mathbook")
    ap.add_argument("--survey", default=r"E:\EPUB\work\survey.json")
    ap.add_argument("--jobs", type=int, default=2, help="同时跑几本书（唯一的提速杠杆）")
    ap.add_argument("--sort", default="small", choices=["small", "big", "name"])
    ap.add_argument("--kind", default="all", choices=["all", "digital", "mixed", "scan"])
    ap.add_argument("--min-pages", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--chunk-pages", type=int, default=0, help="整本切段长度，0=不切")
    ap.add_argument("--ocr-size", default="gundam", choices=["tiny", "small", "base", "large", "gundam"])
    ap.add_argument("--footnotes", action="store_true")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--only", default=None, help="只跑这个书单文件里的相对路径")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout-min", type=int, default=240, help="单本超时（分钟），超时即杀")
    ap.add_argument("--strict-ocr", action="store_true",
                    help="单页 OCR 失败即整本作废（pdf-craft 默认）。批处理里几乎总是想反着来")
    ap.add_argument("--ocr-timeout", type=int, default=300,
                    help="单次 OCR 请求超时（秒）。默认 180 对密集页面偏紧")
    ap.add_argument("--reocr-rounds", type=int, default=4,
                    help="抽到没有整页兜底为止，最多抽几轮（1=关掉重抽，退回老行为）")
    ap.add_argument("--reocr-stall", type=int, default=2,
                    help="连续几轮「丢页数没减少」就放弃（避免对着啃不动的页反复烧钱）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="只从已有 .pcex 重渲染 EPUB（不 OCR）——渲染层改了参数/代码后用这个")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--log", default=r"E:\EPUB\work\mathbook\batch.log",
                    help="自写 UTF-8 日志（不要用 shell 重定向，中文会变乱码）")
    # worker 模式
    ap.add_argument("--run-single", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-first", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-last", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-pages-total", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.run_single is not None:
        return run_single(a)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    if not api_key() and not a.dry_run:
        sys.exit("ERROR: 没有 OCR 凭证。设 SF_API_KEY，或放到 " + str(KEY_FILE))

    log = Log(a.log)

    work = Path(a.work)
    manifest_path = work / "manifest.json"
    manifest = load_manifest(manifest_path)
    # --render-only 写自己的账本，绝不碰主 manifest。
    #
    # 批次可以同时在跑并持续往主 manifest 追加新完成的记录，而 save_manifest
    # 是整份 dict 的读-改-写：两边各持一份内存副本时，后写的一方会静默抹掉
    # 对方刚写进去的条目 —— 产物在磁盘上、记录里却没有，下次续跑把这些书
    # 重做一遍，白烧几小时 OCR。重渲染推进的不是 OCR 进度，没必要挤进那本账。
    write_path = (work / "render_manifest.json") if a.render_only else manifest_path
    written = load_manifest(write_path) if a.render_only else manifest

    chunked = bool(a.chunk_pages)
    queue = build_queue(a)
    todo = []
    skipped_done = skipped_failed = 0
    if a.render_only:
        # 重渲染的队列来自 manifest（哪本书已经抽好了就重渲染哪本），不是来自目录扫描。
        #
        # --only / --limit 在这里同样要生效：渲染层的改动是全库性的，但「先拿一两本
        # 验证、再放开全量」是唯一稳妥的顺序，而这个分支一度把这两个开关整个忽略掉。
        wanted = None
        if a.only:
            wanted = {ln.strip() for ln in Path(a.only).read_text(encoding="utf-8").splitlines() if ln.strip()}
        for k, rec in manifest.items():
            if rec.get("status") != "done":
                continue
            pcex_path = rec.get("pcex") or ""
            if not pcex_path or not Path(pcex_path).is_file():
                continue
            rel = rec.get("rel") or k
            if wanted is not None and rel not in wanted and k not in wanted:
                continue
            pages = int(rec.get("pages") or 0)
            first, last = 0, max(pages - 1, 0)
            chunk = str(rec.get("chunk") or "all")
            if chunk != "all" and "-" in chunk:
                lo, hi = chunk.split("-")
                first, last = int(lo) - 1, int(hi) - 1
            todo.append((k, Job(rel=rel, pdf=Path(a.root) / rel, first=first, last=last, pages=pages)))
        if a.limit:
            todo = todo[: a.limit]
    else:
        for j in queue:
            k = key_of(j, chunked)
            rec = manifest.get(k)
            if rec and rec.get("status") == "done" and Path(rec.get("epub", "")).is_file() and not a.no_resume:
                skipped_done += 1
                continue
            if rec and rec.get("status") == "failed" and not a.retry_failed and not a.no_resume:
                skipped_failed += 1
                continue
            todo.append((k, j))

    total_pages = sum(j.last - j.first + 1 for _, j in todo)
    log(f"[i] 队列 {len(queue)} 项（已完成跳过 {skipped_done}，失败跳过 {skipped_failed}）")
    log(f"[i] 本次待跑 {len(todo)} 项 / {total_pages:,} 页")
    log(f"[i] 并行 {a.jobs} · ocr_size={a.ocr_size} · 切段={a.chunk_pages or '关'} · 输出 {a.out}")
    if a.dry_run:
        for k, j in todo[:40]:
            log(f"    {j.pages:5d}p  {k}")
        if len(todo) > 40:
            log(f"    … 其余 {len(todo) - 40} 项")
        log.close()
        return 0

    timeout = a.timeout_min * 60
    started = time.time()
    logs_dir = work / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def launch(item):
        k, j = item
        # ★ 把并发摊到多把 key 上：按书路径的稳定校验和取模分配。
        #
        # 不能分配完才发现只有一把 key —— 那样 24 路并发全压在 key1 上，
        # 正好撞回单 key 的闸门。也不要让同一本书每次落到不同 key（不可复现的
        # 排查噩梦），所以用 `sum(ord(c))` 这种**跨进程稳定**的校验和，
        # 不用 `hash()`（PYTHONHASHSEED 随机化会让同一本书每次换 key）。
        _keys = api_keys()
        my_key = _keys[sum(ord(c) for c in j.rel) % len(_keys)] if _keys else ""
        cmd = [
            PY, str(Path(__file__).resolve()), "--root", a.root, "--out", a.out, "--work", a.work,
            "--run-single", j.rel, "--chunk-first", str(j.first), "--chunk-last", str(j.last),
            "--chunk-pages-total", str(j.pages), "--ocr-size", a.ocr_size,
            "--ocr-timeout", str(a.ocr_timeout),
            # 重抽参数必须转发：worker 是另一个进程，不转发就只会用默认值，
            # 于是 `--reocr-rounds 1`（关掉重抽）在主进程上看起来"生效了"其实没有。
            "--reocr-rounds", str(a.reocr_rounds),
            "--reocr-stall", str(a.reocr_stall),
        ]
        if a.strict_ocr:
            cmd.append("--strict-ocr")
        if a.footnotes:
            cmd.append("--footnotes")
        if a.cover:
            cmd.append("--cover")
        if a.render_only:
            cmd.append("--render-only")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                    # worker 里的 `api_key()` 优先读 SF_API_KEY，
                                    # 所以分配 key 只需改这一处环境变量，不必在
                                    # 命令行传密钥（避免密钥出现在进程列表里）。
                                    "SF_API_KEY": my_key})
            # worker 的完整输出留档：失败时唯一能看到 pdf-craft 真实抱怨的地方。
            # 但留档失败**绝不允许**掀翻整批 —— 这里跑在 ThreadPool 里，异常会经
            # `fut.result()` 冒泡到 main，把同批其余（可能早就跑完了的）书一起判死。
            # 2026-09-23 12:39 就是 `write_text` 撞上瞬时 Errno 13 → 整批 rc=1。
            try:
                (logs_dir / (safe_stem(k) + ".out.txt")).write_text(
                    (r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""), encoding="utf-8"
                )
            except OSError as exc:
                log(f"[w] 单本输出留档失败（不影响本书判定）：{type(exc).__name__}: {exc}")
            payload = (r.stdout or "").strip().splitlines()
            rec = None
            for line in reversed(payload):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        rec = json.loads(line)
                        break
                    except Exception:  # noqa: BLE001
                        continue
            if rec is None:
                rec = {"status": "failed", "stage": "worker",
                       "error": ((r.stderr or r.stdout or "no output")[-400:])}
        except subprocess.TimeoutExpired:
            rec = {"status": "failed", "stage": "timeout", "error": f">{a.timeout_min} min"}
        rec["key"] = k
        rec["rel"] = j.rel
        rec["pages"] = j.last - j.first + 1
        rec["wall_seconds"] = round(time.time() - t0, 1)
        rec["sf_key"] = my_key[:10] + "…" + my_key[-4:] if my_key else "(无)"
        return rec

    done = failed = 0
    total = len(todo)
    # 按「完成顺序」而不是「提交顺序」消费结果。
    #
    # 第一版用的是 `pool.map`，它按提交顺序产出。两个后果，第二个很严重：
    #   1. 一本慢书会把后面所有已完成的书全部藏起来，日志看起来像卡死，而工作其实在推进
    #      （实测：一本 42 页的分类目录花了 10 分钟，期间另有两本早就跑完了却不出现在日志里）；
    #   2. manifest 只在结果被消费时才写，所以中途 Ctrl-C 会丢掉所有「已完成但尚未消费」
    #      的书——产物在磁盘上、记录里没有，重跑会把它们再做一遍，白烧几小时 OCR。
    # `as_completed` 一完成就落盘，这两个问题一起消失。
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {pool.submit(launch, item): item for item in todo}
        for finished, fut in enumerate(as_completed(futures), 1):
            # 用 dict 记住「future → 哪一本」，这样哪怕 launch 抛了意外异常，
            # 也能把失败如实记到它自己名下，而不是让整个 as_completed 循环炸掉、
            # 连累同批已经跑完的书（它们的产物在磁盘上，账里却没记录）。
            _k0, _j0 = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "key": _k0,
                    "rel": _j0.rel,
                    "pages": _j0.last - _j0.first + 1,
                    "status": "failed",
                    "stage": "launch",
                    "wall_seconds": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            written[rec["key"]] = rec
            # **只写这一本的记录**，不要把整份内存快照写回去。
            #
            # `merge_write` 的合并是 `disk.update(data)` —— 传整份 `written` 时，
            # 我们在 T0 载入、之后被别人更新过的**旧条目**会把磁盘上的新值盖掉。
            # 典型事故：autopilot 与一次手工重跑同时在跑，重跑内存里某本书还是
            # `failed`/缺失，重跑一落盘就把 autopilot 刚写好的 `done` 抹成旧的
            # → 产物在磁盘、账里没有，下次续跑重做、白烧额度。
            # 只提交本进程真正产出的那一条，语义上就不可能覆盖别人。
            save_manifest(write_path, {rec["key"]: rec})
            ok = rec.get("status") == "done"
            done += ok
            failed += not ok
            tag = "OK " if ok else "!! "
            log(
                f"[{finished}/{total}] {tag}{rec['pages']:5d}p "
                f"{rec.get('seconds_per_page', '?'):>5}s/p "
                f"{rec['key'][:80]}"
                + ("" if ok else f"  → {str(rec.get('error', ''))[:160]}")
            )

    el = time.time() - started
    log(f"\n[✓] {done} 成功 / {failed} 失败 · 用时 {el/60:.1f} 分钟")
    log(f"    清单 {write_path}")
    log.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### 2. pipeline/ocr_resume.py

```python
#!/usr/bin/env python3
"""ocr_resume.py — OCR 断点续跑的两个原语：**清坏页** 与 **数丢页**。

为什么单独一个模块
------------------
这两件事同时被两条链路需要，而且**必须用同一套判据**：

* `batch.py`（主流程）——抽完一轮就数一次丢页，没数干净就再抽，直到干净；
* `reocr_book.py` / `reocr_all.py`（事后补跑老书）——同样的循环，只是驱动方式不同。

判据写两遍就一定会漂移，所以放在这里。

两个概念，别混
--------------
1. **丢页（fallback page）**：pdf-craft 在单页 OCR 失败且 `ignore_ocr_errors=True` 时，
   用 `_create_fallback_page` 把**整页渲染图**当内容塞进结果（`pdf/ocr.py`）。
   这一页的文字/公式**全部丢失**，只剩一张 300 DPI 的图（约 0.47 MB）。
   成品看起来"有东西"，所以这个损失是**静默**的。它同时会写一个 `page_N.failed` 标记。
2. **坏页（truncated page）**：进程被超时杀掉时留下的**半截** `page_N.xml`。见下。

坏页为什么会把整本书**永久**卡死
--------------------------------
`pdf/ocr.py:130` 的跳过判据是

    if file_path.exists() and not failure_path.exists():   # page_N.xml / page_N.failed
        → SKIP

**只看文件在不在，不解析内容**。而 `save_xml` 不是原子写。于是半截 XML 此后永远被
跳过，且 chapters 阶段要读 `ocr/` 里全部 `page_*.xml`，直接抛
`ValueError: Failed to parse XML file: .../page_211.xml` —— 整本重抽从此必败，
外层调度还一遍遍重试同一件必败的事。实测 2026-09-21 中招过一本（Tensor Algebra）。

所以「续跑」的前提是：**每次开工前先把坏页清掉**。通用教训——断点续跑的判据只要是
「文件存在」，就必须假设文件是**内容**坏的。
"""
from __future__ import annotations

import shutil
import struct
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# pdf_craft/document/package.py:863 `_is_asset_hash` 用的字符集：
# 资产文件名必须是**恰好 64 位小写 hex**（md5 的十六进制形态）。
_HEX = frozenset("0123456789abcdef")


def _is_asset_member(name: str) -> bool:
    """复刻 pdf-craft 对 `assets/` 成员的合法性判据（`document/package.py:690`）。

    原文遍历 `paths.assets.iterdir()`，要求每一项都满足：
    是普通文件（非符号链接）＋ 以 `.png` 结尾 ＋ 去掉扩展名后 `_is_asset_hash`
    （`len == 64` 且全是小写 hex）。**任何一项不满足就 `raise ValueError`** ——
    注意那是**整本书作废**，不是跳过那张图。
    """
    return name.endswith(".png") and len(name) == 68 and all(c in _HEX for c in name[:-4])


def scrub_dirty_assets(analysing: Path, min_age_seconds: float = 0.0) -> list[str]:
    """清掉 `analysing/extraction/assets/` 里会让**整本作废**的非法成员，返回被删的名字。

    `min_age_seconds`（很重要，别省）
    --------------------------------
    `.temp` 有**两种**来源，处理方式相反：

    * **孤儿**：进程被硬杀时停在「写完 temp、还没 rename」之间 ⇒ 该删。
    * **在途**：worker 正在 clip（`AssetHub.clip()` 写完 temp、马上要 rename 成 .png）
      ⇒ **绝不能碰**，删了它下一次 `rename` 就是 `FileNotFoundError`，等于我亲手弄坏这本书。

    `batch.py` 里是在**每轮 OCR 开始之前**调用（此刻本进程还没有任何在写的 temp），
    所以默认 `min_age_seconds=0`（全清）是安全的。
    但**带外调用**（人工清理、看护脚本，worker 正跑着）必须传一个年龄护栏，
    只清「存在超过 N 秒」的 —— 孤儿一定有年龄，在途的一秒都不到。

    为什么必须清
    ------------
    `pdf-craft` 打包 `.pcex` 前会遍历 assets 目录，任何非「64 位 hex 的 .png」成员
    都直接抛 `ValueError: invalid asset member: …` ⇒ **整本书失败**。
    2026-09-23 实测 3 本中招，其中一个 `.temp` **已存在 1144 分钟**（≈19 小时）
    —— 也就是说这不是"我硬杀才有的"，**长期潜伏**的孤儿一直在让这些书每轮必败。

    为什么必须**持久化**在代码里而不是手工删
    ----------------------------------------
    `analysing/` 是磁盘上的**持久续跑目录**（这正是能续跑的原因），残渣不会自己消失；
    不清就是「每轮选中 → 每轮在打包阶段失败」的永久空转，白占一个并发槽。

    安全性：判据与 pdf-craft 完全一致（`_is_asset_member`），**只删它必然会拒绝的东西**。
    干跑验证：全库 151,869 个合法资产**零误判**，精确命中 3 个 `.png.temp`。
    """
    assets = analysing / "extraction" / "assets"
    if not assets.is_dir():
        return []
    now = time.time()
    removed: list[str] = []
    for path in sorted(assets.iterdir()):
        try:
            if path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif _is_asset_member(path.name):
                continue                      # 合法资产，留着
            else:
                if min_age_seconds > 0:
                    try:
                        if now - path.stat().st_mtime < min_age_seconds:
                            continue          # 太新 ⇒ 可能是"在途"，绝不能删
                    except OSError:
                        continue
                path.unlink(missing_ok=True)
        except OSError:
            continue
        removed.append(path.name)
    return removed


def scrub_truncated_pages(ocr_dir: Path) -> list[int]:
    """删掉 `ocr/` 里解析不了的 `page_N.xml`，返回被删的页号。

    删掉就够了：下次 OCR 时该页因「无 XML」被正常重抽。顺手补一个 `.failed`
    标记让语义自洽（`page_N.failed` 的含义是「这页要重抽」）。

    没坏页时开销是「按文件名 glob + 逐个 parse」，250 页约几十毫秒，随便调用。
    """
    if not ocr_dir.is_dir():
        return []
    broken: list[Path] = []
    for path in sorted(ocr_dir.glob("page_*.xml")):
        try:
            ET.parse(path)
        except ET.ParseError:
            broken.append(path)
    for path in broken:
        index = path.stem.removeprefix("page_")
        (ocr_dir / f"page_{index}.failed").write_text("TruncatedXML", encoding="utf-8")
        path.unlink()
    return [int(p.stem.removeprefix("page_")) for p in broken]


def failed_marker_pages(ocr_dir: Path) -> list[int]:
    """`page_*.failed` 标记对应的页号 —— 即当前已知的丢页。

    这是 `_create_fallback_page` 留下的**直接证据**，比数事件更稳（跨进程、跨轮次
    都还在）。OCR 成功时 pdf-craft 会把这个标记删掉（`pdf/ocr.py:208`），
    所以它只反映「现在还没搞好」的页。
    """
    if not ocr_dir.is_dir():
        return []
    out: list[int] = []
    for path in ocr_dir.glob("page_*.failed"):
        stem = path.stem.removeprefix("page_")
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


# ---------------------------------------------------------------------------
# 取证：直接量成品，不读任何记账
# ---------------------------------------------------------------------------


def _page_width_heights(z: zipfile.ZipFile) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """返回 (按页序的尺寸列表, 去重后的尺寸集合)。

    两个都要：列表长度是**页数**，集合用来判「这张图是不是整页」。
    一开始只返回去重集合，于是「总页数」报成了「有几种页面尺寸」（一本 5 页的书
    报成 1 页）—— 这种字段名和值对不上的错，在汇总表里几乎看不出来。
    """
    root = ET.fromstring(z.read("pages.xml"))
    sizes = [(int(p.get("width")), int(p.get("height"))) for p in root.iter("page")]
    return sizes, set(sizes)


def _png_height_width(z: zipfile.ZipFile, name: str) -> tuple[int, int]:
    """只读 PNG 头 24 字节拿 IHDR 宽高，不整张解压。"""
    return struct.unpack(">II", z.open(name).read(24)[16:24])


def fallback_stats(pcex: Path) -> dict:
    """量一本 .pcex 里有多少**整页兜底图**。返回 dict，键：

    ``pages`` 页数 · ``page_size`` 页面像素 · ``assets`` 图片张数 ·
    ``fallback_pages`` 整页兜底页数 · ``fallback_bytes``/``assets_bytes`` 原字节

    **取证，不读任何记账。** 判据来自 pcex 内部两个硬事实，**不需要阈值**：

    1. `pages.xml` 给出每页渲染像素尺寸（`render_dpi="300"`，实测 2550×3300）；
    2. `_create_fallback_page` 存的是**整页**渲染图，像素尺寸必然**等于**页面尺寸。

    裁剪出来的公式/插图永远小于整页（Tao：107 张 2550×3300，其余 287 张全在
    1100 px 宽以下，中间是断开的），所以直接精确相等即可。只读 PNG 头 24 字节，
    不解压，几百张图也就几十毫秒。

    注意 `includes_cover` 时封面本身可能就是一张整页图，会被算进来 —— 所以这个数
    适合**报告与告警**，不适合当重抽循环的判据（循环用 OCR 失败事件或 `.failed` 标记）。
    """
    with zipfile.ZipFile(pcex) as z:
        sizes, page_set = _page_width_heights(z)
        pngs = [i for i in z.infolist() if i.filename.lower().endswith(".png")]
        full = [i for i in pngs if _png_height_width(z, i.filename) in page_set]
        return {
            "pages": len(sizes),
            "page_size": sorted(page_set)[0] if page_set else None,
            "assets": len(pngs),
            "fallback_pages": len(full),
            "fallback_bytes": sum(i.file_size for i in full),
            "assets_bytes": sum(i.file_size for i in pngs),
        }
```

### 3. pipeline/manifest_lock.py

```python
r"""跨进程文件锁：串行化主 manifest 的「读-改-写」。

为什么需要
----------
`batch.py`（主转换）与 `promote_reocr.py` / `promote_mineru.py`（入库）会同时写
`work/mathbook/manifest.json`。整份 dict 的读-改-写若并发，后写的一方会**静默抹掉**
对方的条目 → 产物在磁盘、账里没有，下次续跑还会重做、白烧额度。用本锁把每次写串行化，
并配合「写前重新读盘 + update 合并」消除丢条目。

Windows 用 msvcrt.locking（字节范围独占锁）；其它平台退化到 fcntl.flock。
"""
from __future__ import annotations

import contextlib
import os
import time

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


@contextlib.contextmanager
def file_lock(lock_path, timeout: float = 300.0):
    lock_path = str(lock_path)
    d = os.path.dirname(lock_path)
    if d:
        os.makedirs(d, exist_ok=True)
    f = open(lock_path, "a+")
    start = time.time()
    locked = False
    try:
        while True:
            try:
                if msvcrt is not None:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"取 manifest 锁超时（{timeout}s）: {lock_path}")
                time.sleep(0.15)
        yield
    finally:
        if locked:
            try:
                if msvcrt is not None:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
        f.close()


def merge_write(path, data: dict) -> None:
    """在锁内「重读盘 → update 合并 → 原子替换」，供各写入口复用。

    **调用契约：`data` 只放「本次自己产出的键」**，不要把整份内存快照传进来。
    合并是 `disk.update(data)` —— 传整份快照时，载入之后被别人更新过的**旧条目**
    会把磁盘上的新值盖掉（实测形态：autopilot 与手工重跑并行，重跑内存里某本书
    还是 `failed`，一落盘就把 autopilot 刚写好的 `done` 抹回旧值）。
    锁只保证「写不撕裂」，**不保证「你的数据不比磁盘旧」**，后者要靠这个契约。
    """
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.parent / ".manifest.lock"):
        disk = {}
        if path.is_file():
            try:
                disk = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                disk = {}
        if isinstance(disk, dict) and isinstance(data, dict):
            disk.update(data)
        else:
            disk = data
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(disk, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
```

### 4. pipeline/survey.py

```python
#!/usr/bin/env python3
"""survey.py — 给一个 PDF 目录做体检，决定「值不值得 OCR、要多久」。

为什么需要它
------------
pdf-craft 的抽取成本与**页数**成正比，与文件大小无关。一个 10 GB 的目录里可能
既有 1200 页的扫描大部头，也有 8 页的讲义。不知道页数分布就无法承诺工期，也无
法挑出「适合先验证流水线」的书。

抽取是逐页渲染 + 逐页远端 OCR，实测约 15 s/页（SiliconFlow 免费档）。所以
    预估分钟 = 页数 × 15 / 60

判定「扫描件还是原生 PDF」
--------------------------
不是靠猜文件名，也不是只看体积，而是**真的抽一次文本**：

* 抽样若干页，用 PyMuPDF 取 `page.get_text()` 的字符数；
* 原生 PDF 一页通常有 800–3000 字符，扫描件接近 0（除非它自带一层很差的 OCR）；
* 再数一下页面里的图片面积占比，扫描件的图片几乎铺满整页。

据此分三档：`digital`（有可信文本层）、`scan`（无文本层，必须 OCR）、
`mixed`（有文本层但薄，大概率是陈旧的 OCR 层，仍需重做）。

注意：**有文本层不代表不需要 OCR**。原生 PDF 里的公式是一堆 Unicode 符号或
嵌入字体私用区码位，转换不出 LaTeX。这个脚本只回答成本问题，不回答质量问题。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import pymupdf
except ImportError:  # PyMuPDF 改名过一次，旧的 import 名仍可用
    import fitz as pymupdf  # type: ignore

# 每本书抽这么多页来估算文本密度（不读全书，1200 页的书也能秒回）
SAMPLE_PAGES = 40

# 一页的文本字符数落在哪个区间
DIGITAL_CHARS_PER_PAGE = 400
SCAN_CHARS_PER_PAGE = 80

# 实测：SiliconFlow 免费档 DeepSeek-OCR 约 15 s/页
SECONDS_PER_PAGE = 15.0


@dataclass
class Book:
    path: str
    rel: str
    bytes: int
    pages: int
    sampled: int
    chars_per_page: float
    image_area: float          # 抽样页里图片面积的平均占比 0..1
    kind: str                  # digital | scan | mixed | error
    error: str = ""
    title: str = ""
    author: str = ""
    producer: str = ""
    est_minutes: float = 0.0
    encrypted: bool = False


def classify(chars_per_page: float, image_area: float) -> str:
    if chars_per_page >= DIGITAL_CHARS_PER_PAGE:
        return "digital"
    if chars_per_page <= SCAN_CHARS_PER_PAGE:
        return "scan"
    return "mixed"


def probe(path: Path, root: Path) -> Book:
    b = Book(
        path=str(path), rel=str(path.relative_to(root)), bytes=path.stat().st_size,
        pages=0, sampled=0, chars_per_page=0.0, image_area=0.0, kind="error",
    )
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # noqa: BLE001
        b.error = f"{type(e).__name__}: {e}"[:200]
        return b

    try:
        b.encrypted = bool(doc.needs_pass)
        b.pages = doc.page_count
        meta = doc.metadata or {}
        b.title = (meta.get("title") or "").strip()[:200]
        b.author = (meta.get("author") or "").strip()[:200]
        b.producer = (meta.get("producer") or "").strip()[:120]

        if b.encrypted:
            b.kind = "error"
            b.error = "encrypted (needs password)"
            return b

        # 均匀抽样，避开封面/版权页这类必然没内容的页
        n = doc.page_count
        if n <= SAMPLE_PAGES:
            idxs = list(range(n))
        else:
            step = n / SAMPLE_PAGES
            idxs = [int(i * step) for i in range(SAMPLE_PAGES)]

        total_chars = 0
        total_area = 0.0
        for i in idxs:
            page = doc.load_page(i)
            total_chars += len(page.get_text("text") or "")
            # 图片面积占比：判断是不是「整页扫描图」
            try:
                parea = abs(page.rect.width * page.rect.height) or 1.0
                iarea = 0.0
                for info in page.get_image_info():
                    r = pymupdf.Rect(info["bbox"])
                    iarea += abs(r.width * r.height)
                total_area += min(iarea / parea, 1.0)
            except Exception:  # noqa: BLE001
                pass

        b.sampled = len(idxs)
        if idxs:
            b.chars_per_page = total_chars / len(idxs)
            b.image_area = total_area / len(idxs)
        b.kind = classify(b.chars_per_page, b.image_area)
        b.est_minutes = b.pages * SECONDS_PER_PAGE / 60.0
    finally:
        doc.close()
    return b


def human(n: float) -> str:
    return f"{n:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="PDF 目录体检：页数、扫描/原生、OCR 工期")
    ap.add_argument("--root", default=r"E:\mathbook")
    ap.add_argument("--out", default=r"E:\EPUB\work\survey.json")
    ap.add_argument("--summary", default=r"E:\EPUB\work\survey.md")
    ap.add_argument("--limit", type=int, default=0, help="只查前 N 本（调试用）")
    a = ap.parse_args()

    # Windows 控制台默认 GBK，写日志时遇到 ✓ 这类符号会直接抛 UnicodeEncodeError
    # 把整个进程打死（报告已经写完了，却看不到结果）。统一切到 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    root = Path(a.root)
    files = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    if a.limit:
        files = files[: a.limit]

    books: list[Book] = []
    for i, p in enumerate(files, 1):
        b = probe(p, root)
        books.append(b)
        print(f"[{i}/{len(files)}] {b.kind:8s} {b.pages:5d}p  {b.chars_per_page:7.0f}c/p  {b.rel[:70]}", flush=True)

    ok = [b for b in books if b.kind != "error"]
    by_kind: dict[str, list[Book]] = {}
    for b in books:
        by_kind.setdefault(b.kind, []).append(b)

    total_pages = sum(b.pages for b in ok)
    est_hours = total_pages * SECONDS_PER_PAGE / 3600

    lines = [
        "# PDF 目录体检报告",
        "",
        f"目录：`{root}`",
        "",
        "## 总览",
        "",
        f"- PDF 文件：**{len(books)}** 个（其中 {len(books) - len(ok)} 个无法打开）",
        f"- 总页数：**{human(total_pages)}** 页",
        f"- 总体积：{sum(b.bytes for b in books) / (1024 ** 3):.2f} GB",
        f"- 预估 OCR 工期（{SECONDS_PER_PAGE:.0f} s/页）：**{est_hours / 24:.1f} 天**",
        "",
        "## 按类型",
        "",
        "| 类型 | 数量 | 页数 | 预估工期 |",
        "|---|---|---|---|",
    ]
    for k in ("digital", "mixed", "scan", "error"):
        g = by_kind.get(k, [])
        if not g:
            continue
        pg = sum(b.pages for b in g)
        lines.append(f"| `{k}` | {len(g)} | {human(pg)} | {pg * SECONDS_PER_PAGE / 86400:.1f} 天 |")

    # 页数分布：给出「如果只做轻量书能省多少」
    ok.sort(key=lambda b: b.pages)
    for cap in (100, 200, 400, 600):
        g = [b for b in ok if b.pages <= cap]
        pg = sum(b.pages for b in g)
        lines.append("")
        lines.append(
            f"- 页数 ≤ {cap}：{len(g)} 本 / {human(pg)} 页 / {pg * SECONDS_PER_PAGE / 86400:.1f} 天"
        )

    lines += ["", "## 最小的 20 本（适合先验证）", "",
              "| 页数 | 类型 | 字符/页 | 文件 |", "|---|---|---|---|"]
    for b in ok[:20]:
        lines.append(f"| {b.pages} | {b.kind} | {b.chars_per_page:.0f} | `{b.rel}` |")

    lines += ["", "## 最大的 15 本", "", "| 页数 | 类型 | 文件 |", "|---|---|---|"]
    for b in ok[-15:][::-1]:
        lines.append(f"| {b.pages} | {b.kind} | `{b.rel}` |")

    errs = by_kind.get("error", [])
    if errs:
        lines += ["", "## 打不开的文件", "", "| 文件 | 原因 |", "|---|---|"]
        for b in errs:
            lines.append(f"| `{b.rel}` | {b.error} |")

    Path(a.out).write_text(
        json.dumps([asdict(b) for b in books], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(a.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[✓] {a.out}")
    print(f"[✓] {a.summary}")
    print(f"    总 {len(books)} 本 / {human(total_pages)} 页 / 预估 {est_hours/24:.1f} 天")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 5. pipeline/pcexlib.py

```python
"""pcexlib — 把 pdf-craft 的 .pcex 抽取归档解析成可渲染的中间表示（IR）。

设计要点
--------
.pcex 的价值在于它**不是渲染产物**，而是结构化的抽取归档：

* 公式以 **LaTeX 源码**保存（不是 MathML、不是图片）—— `\\mathrm{d}x`、`Y_{1}(y)` 原样保留；
* 文本片段带 `bbox` 与 `page_index`，可回溯到原书版面做 QA；
* 图片按 sha256 独立存放，同一资产可复用。

因此它可以从同一份归档无损地再渲染成任意后端（EPUB / TeX / Markdown / HTML），
**不需要重跑 OCR**。这就是整条流水线的契约层。

.pcex 结构（format_version 3）
------------------------------
    manifest.json       文档元数据
    pages.xml           页面尺寸表
    toc.xml             目录树（只有 id / page_index / order / level，无标题文本）
    chapters/*.xml      正文结构化流
    assets/<sha256>     图片、公式裁剪图（PNG）

chapters/*.xml 的元素
---------------------
    <chapter id level>
      <flow>
        <text role="heading" level="N">
          <fragment page_index source_order bbox>文字 <inline_expr kind="\\(">C</inline_expr> 文字</fragment>
        </text>
        <text role="body">…</text>
        <display-formula>
          <asset ref="formula" page_index bbox asset_hash>
            <content>LATEX</content>
            <caption>可选说明，内含 <inline_expr/></caption>
          </asset>
        </display-formula>
        <standalone-asset>
          <asset ref="image" page_index bbox asset_hash>
            <content>图2-1</content>          <!-- 图注 -->
          </asset>
        </standalone-asset>
      </flow>
    </chapter>

目录与正文的关联键
------------------
toc.xml 的 `<item page_index="1" order="0">` 与正文 heading 的
`<fragment page_index="1" source_order="0">` **精确对应**，因此可以可靠地把
目录层级接到标题文本上。
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from typing import Iterator, Optional, Union
from xml.etree import ElementTree as ET

# 公式文本的结构精修单独一个模块：它是纯字符串规则，与「块怎么组装」无关，
# 也正因如此它可以脱离 .pcex 单独跑门禁（work/check_texfix.py）。
from texfix import fix_math, has_content

__all__ = [
    "Pcex",
    "Chapter",
    "Heading",
    "Para",
    "MathBlock",
    "Figure",
    "TextRun",
    "MathRun",
    "TocItem",
    "split_eqno",
    "Block",
]

# ---------------------------------------------------------------------------
# 公式编号抽取
# ---------------------------------------------------------------------------

# pdf-craft 把公式编号作为 LaTeX 的一部分保留下来，形如：
#     P(x,y)\mathrm{d}x + Q(x,y)\mathrm{d}y = 0 \quad (2.15)
# 渲染时把它拆出来单独右对齐，比留在公式里好看得多。
# 只匹配「结尾的 (数字.数字…)」，避免误伤 f(x)=(1) 这类。
_EQNO_RE = re.compile(
    r"""\s*
    (?:\\quad|\\qquad|\\hspace\*?\{[^}]*\}|\\hfill)?   # 可选的水平间距命令
    \s*
    (                       # 捕获组：只保留 (2.15) 本身，不要前面的 \quad
      \(
        (?:\d+(?:\.\d+)+        # 2.15 / 2.4.1
          |[A-Z]\.\d+(?:\.\d+)* # A.1 / B.2.3
        )
      \)
    )
    \s*$""",
    re.VERBOSE,
)


def split_eqno(latex: str) -> tuple[str, Optional[str]]:
    """把公式末尾的编号拆出来。

    >>> split_eqno(r"a+b = c \\quad (2.15)")
    ('a+b = c', '(2.15)')
    >>> split_eqno(r"f(x)=x^2")
    ('f(x)=x^2', None)
    """
    m = _EQNO_RE.search(latex)
    if not m:
        return latex.strip(), None
    return latex[: m.start()].strip(), m.group(1)


# ---------------------------------------------------------------------------
# 行内元素
# ---------------------------------------------------------------------------


@dataclass
class TextRun:
    text: str


@dataclass
class MathRun:
    latex: str
    # kind 为 "\\(" 时是行内公式，"\\[" 时是行间公式（auto-render 会按 display 处理）
    kind: str = r"\("

    @property
    def display(self) -> bool:
        return self.kind == r"\["

    @property
    def delimiter(self) -> tuple[str, str]:
        return (r"\[", r"\]") if self.display else (r"\(", r"\)")


# ---------------------------------------------------------------------------
# 块级元素
# ---------------------------------------------------------------------------


@dataclass
class Heading:
    level: int
    runs: list[Union[TextRun, MathRun]] = field(default_factory=list)
    page: Optional[int] = None
    order: Optional[int] = None
    bbox: Optional[str] = None
    kind: str = "heading"

    @property
    def text(self) -> str:
        """纯文本标题（剥掉公式，用于生成目录）。"""
        return "".join(r.text for r in self.runs if isinstance(r, TextRun)).strip()


@dataclass
class Para:
    runs: list[Union[TextRun, MathRun]] = field(default_factory=list)
    page: Optional[int] = None
    order: Optional[int] = None
    bbox: Optional[str] = None
    kind: str = "para"

    @property
    def has_math(self) -> bool:
        return any(isinstance(r, MathRun) for r in self.runs)


@dataclass
class MathBlock:
    latex: str = ""
    eqno: Optional[str] = None
    caption: Optional[list[Union[TextRun, MathRun]]] = None
    page: Optional[int] = None
    order: Optional[int] = None
    bbox: Optional[str] = None
    asset_hash: Optional[str] = None
    kind: str = "math"
    # 渲染成图片而不是 LaTeX。由渲染端在 KaTeX 编译失败时置位——原图是 100%
    # 忠实的，而多行 array 阵列（三行 ∇ 之类）OCR 会逐格切碎，规则救不回来。
    # 用显式字段而不是「latex 为空」当信号：后者会让 tex/md 后端也丢掉内容。
    force_image: bool = False
    # 组内对齐："" 未判定 / "center" 居中 / "left" 靠左。
    # 一个多行 display 公式被 OCR 归成一块时，行与行的缩进关系不在文本里，
    # 只存在于原图（或各行的包围盒）——见 extent_align / image_line_extents。
    # 渲染端据它决定整块是否贴左，LaTeX 侧的对齐方式在合并时就写进表达式了。
    align: str = ""


@dataclass
class Figure:
    asset_hash: str = ""
    caption: Optional[list[Union[TextRun, MathRun]]] = None
    page: Optional[int] = None
    order: Optional[int] = None
    bbox: Optional[str] = None
    kind: str = "figure"


Block = Union[Heading, Para, MathBlock, Figure]


# ---------------------------------------------------------------------------
# 被版面切断的公式：缝回去
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\\(begin|end)\s*\{\s*([^}]*)\s*\}")
_END_HEAD_RE = re.compile(r"^\s*\\end\s*\{\s*([^}]*)\s*\}")
_LEAD_RELATION_RE = re.compile(
    r"^\s*(?:=|\\approx|\\equiv|\\le|\\leq|\\ge|\\geq|\\neq|\\sim|\\pm|\\mp"
    r"|\\to|\\rightarrow|\\Rightarrow|\\Leftrightarrow|\\propto)(?![a-zA-Z])"
)


def _open_environments(latex: str) -> list[str]:
    """返回这段 LaTeX 里**尚未闭合**的环境名（按嵌套顺序）。

    >>> _open_environments(r"\\begin{array}{cc} a & b")
    ['array']
    >>> _open_environments(r"\\begin{array}{cc}\\end{array}")
    []
    """
    stack: list[str] = []
    for m in _ENV_RE.finditer(latex):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif stack and stack[-1] == m.group(2):
            stack.pop()
    return stack


def _unbalanced_delims(latex: str) -> bool:
    """`\\left` 与 `\\right` 数量不等——多半是在这里被切断了。"""
    n_left = len(re.findall(r"\\left(?![a-zA-Z])", latex))
    n_right = len(re.findall(r"\\right(?![a-zA-Z])", latex))
    return n_left != n_right


# ---------------------------------------------------------------------------
# 定界符泄漏：OCR 把 \( \) \[ \] 当成正文写进了公式体
# ---------------------------------------------------------------------------

# `(?<!\\)` 用来放过 LaTeX 的行分隔：`\\` 与 `\\[2mm]` 里的方括号紧跟着反斜杠，
# 不是泄漏。除此之外出现在公式体里的定界符都必然是 OCR 的噪音——数学模式中
# 它们本来就只是「进入数学模式」的标记，不是内容。
_LEAK_DELIM_RE = re.compile(r"(?<!\\)\\[()\[\]]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATEX_NOISE_RE = re.compile(r"\\[a-zA-Z]+|\\.|[{}^_&$~\\]")


def strip_leaked_delims(latex: str) -> str:
    r"""剥掉混进公式体的 `\(` `\)` `\[` `\]`，返回清理后的正文。

    >>> strip_leaked_delims(r"\[ \mathsf{\Delta} x")
    ' \\mathsf{\\Delta} x'
    >>> strip_leaked_delims(r"P\] \[Q")
    'P Q'
    >>> strip_leaked_delims(r"a \\[2mm] b")   # 合法行分隔，不动
    'a \\\\[2mm] b'
    """
    return _LEAK_DELIM_RE.sub("", latex)


def prose_payload(latex: str) -> Optional[str]:
    r"""若这条「公式」其实是正文被 OCR 吞进来了，返回该正文；否则 None。

    与上一条的区别在于**内容**而不是定界符：`到\(`、`显然 \(u` 这类条目剥掉
    定界符也依旧不是公式，它们是行内公式的**左边界被认错**，把前面的中文一起
    卷了进来。留在数学字体里渲染只会显得怪，退回文本才是它本来的样子。

    判据：剥掉 LaTeX 命令与结构符之后，剩下的可见字符里 CJK 过半。
    纯英文的「公式混正文」（如 `Z_{i}called the free energy`）不在此列——
    那种情况剥掉定界符后仍含真实公式内容，退回文本反而丢东西。
    """
    stripped = strip_leaked_delims(latex).strip()
    if not stripped:
        return None
    cjk = len(_CJK_RE.findall(stripped))
    if not cjk:
        return None
    visible = _LATEX_NOISE_RE.sub("", stripped)
    visible = re.sub(r"\s+", "", visible)
    if cjk * 2 >= max(len(visible), 1):
        return stripped
    return None


def _heal_math_text(blocks: list[Block]) -> tuple[int, int, int]:
    r"""就地清理所有公式：剥定界符、把假公式退回正文、修结构错误。

    返回 (剥离了定界符的条数, 退回正文的条数, 结构修复过的条数)。三个数都要
    报出来——静默地把公式变成文本会悄悄改变公式总数，之后拿总数做基线对比
    就会对不上；静默地改公式内容则更隐蔽，它是唯一一类「不报错、不崩溃，
    但读者看到的和书上不一样」的改动。

    第三步（`texfix.fix_math`）修的是**一条公式自己的文本**：环境名配错、
    花括号不成对、`\\` 被折成 `\ `。它和这里另外两步的区别在于有没有外部
    证据——那两步靠的是块之间的关系（哪个块是续行、哪段是说明文字），这一步
    只靠公式自身的结构（`\begin` 与 `\end` 的配对、括号的计数）。
    """
    stripped_n = degraded_n = repaired_n = 0

    def repair(latex: str, inline: bool) -> str:
        nonlocal repaired_n
        fixed, applied = fix_math(latex, inline=inline)
        if applied and has_content(fixed):
            repaired_n += 1
            return fixed
        return latex

    def fix_run(runs: list) -> list:
        nonlocal stripped_n, degraded_n
        out: list = []
        for r in runs:
            if not isinstance(r, MathRun):
                out.append(r)
                continue
            before = r.latex
            prose = prose_payload(before)
            if prose is not None:
                out.append(TextRun(prose))
                degraded_n += 1
                continue
            after = strip_leaked_delims(before)
            if after != before:
                stripped_n += 1
            # 行内与行内式显示（`kind="\["` 但落在段落里的）走不同修法：显示式
            # 里 `\tag` 是允许的，删掉会平白丢掉公式编号。
            after = repair(after, not r.display)
            r.latex = after
            out.append(r)
        return out

    for b in blocks:
        if isinstance(b, MathBlock):
            before = b.latex
            after = strip_leaked_delims(before)
            if after != before:
                stripped_n += 1
            b.latex = repair(after, False)
            # 说明文字里的公式同样在成品里渲染（caption 的 MathRun 输出成
            # math-inline），漏掉它，那些公式就绕过了全部修复——实测有几条
            # 未定义命令正是这样活到成品里的。
            if b.caption:
                b.caption = fix_run(b.caption)
        elif isinstance(b, Figure):
            if b.caption:
                b.caption = fix_run(b.caption)
        elif isinstance(b, (Para, Heading)):
            b.runs = fix_run(b.runs)

    return stripped_n, degraded_n, repaired_n


def _is_continuation(prev: MathBlock, nxt: MathBlock) -> str:
    """判断下一块是不是上一块那条公式的续行。返回理由，空串表示不是。

    判据取「确凿的未完成信号」，宁可漏缝不可错缝——把两条独立公式粘成一条，
    比让一条公式分成两行显示的伤害大得多。

    1. `\\begin{env}` 没闭合，而下一块正好以它的 `\\end{env}` 开头；
    2. 上一块留下未配对的 `\\left(`/`\\right)`；
    3. 上一块以行分隔 `\\\\` 收尾（作者本来就写了换行）；
    4. **下一块以关系符开头**——一个独立的公式不会从 `=` 开始。

    第 4 条来自实测：Hamilton 那本书里公式 (3) 就是这样被切成
    `\\nabla \\cdot \\overleftrightarrow{A} = \\left( ...` 和 `= \\frac{\\partial ...}{...}` 两块，
    各自看起来都像完整公式，只有拼起来才是书上那一条。
    """
    pending = _open_environments(prev.latex)
    head = _END_HEAD_RE.match(nxt.latex)
    if pending and head and head.group(1) == pending[-1]:
        return "env"
    if _unbalanced_delims(prev.latex):
        return "delim"
    if prev.latex.rstrip().endswith("\\\\"):
        return "rowbreak"
    if _LEAD_RELATION_RE.match(nxt.latex):
        return "relation"
    return ""


def _stitch_split_formulas(blocks: list[Block]) -> int:
    """把被版面切成两块的公式接回去，返回缝合了几处。

    抽取是按版面切块的，一条跨行的大公式经常变成两个 `display-formula`。
    合并后用换行连接：对 LaTeX 而言换行就是空白，正是它被切开前的样子；
    补一个 `\\\\` 反而会凭空多出一行。
    """
    stitched = 0
    out: list[Block] = []
    for b in blocks:
        if out and isinstance(b, MathBlock) and isinstance(out[-1], MathBlock):
            prev = out[-1]
            if _is_continuation(prev, b):
                prev.latex = prev.latex + "\n" + b.latex
                prev.eqno = prev.eqno or b.eqno
                if prev.caption is None:
                    prev.caption = b.caption
                if prev.asset_hash is None:
                    prev.asset_hash = b.asset_hash
                stitched += 1
                continue
        out.append(b)
    blocks[:] = out
    return stitched


# ---------------------------------------------------------------------------
# 多行公式：把被 OCR 切碎的显示块合回一个整体
# ---------------------------------------------------------------------------


def is_blank_math(s: str) -> bool:
    r"""`{}`、`{{}}`、空白——没有任何内容的占位。

    >>> is_blank_math("{}"), is_blank_math("{{}}"), is_blank_math("{x}")
    (True, True, False)
    """
    return not re.sub(r"[{}\s]", "", s or "")


def join_math_lines(parts: list[str], align: str = "") -> str:
    r"""把同一显示块的若干行合回一个 LaTeX 表达式。

    默认用 `gathered`（每行居中堆叠）而**不是**换行拼接：对 LaTeX 而言换行只是
    空白，`a\nb` 渲染出来仍是一行 `ab`。书上是上下两行的公式，只有
    `\begin{gathered}` 才是它本来的样子。

    `align="left"` 时改用 `aligned` 并在**每行行首**加一个 `&`：这样第一列
    全部为空、内容列左对齐，各行从同一个 x 起排——书上的推导组就是这么排的。
    （KaTeX 会为它生成 `col-align-r` + `col-align-l` 两列，实测见
    `work/katex_align_struct.js`。）

    行内**本来就有** `&` 时不动它，直接沿用 `aligned`：那是 OCR 真的保留下了
    作者的对齐点，比我们按包围盒猜出来的准。

    >>> join_math_lines(["{a}", "{b}"])
    '\\begin{gathered}{a} \\\\ {b}\\end{gathered}'
    >>> join_math_lines(["{a}", "{b}"], align="left")
    '\\begin{aligned}& {a} \\\\ & {b}\\end{aligned}'
    >>> join_math_lines(["x &= 1", "y &= 2"], align="left")   # 已有的对齐点不动
    '\\begin{aligned}x &= 1 \\\\ y &= 2\\end{aligned}'
    >>> join_math_lines(["a"])
    'a'
    >>> join_math_lines(["a", "{}", "  "])
    'a'
    """
    lines = [strip_leaked_delims(p or "").strip() for p in parts]
    lines = [ln for ln in lines if ln and not is_blank_math(ln)]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    if any("&" in ln for ln in lines):
        env, body = "aligned", " \\\\ ".join(lines)
    elif align == "left":
        env, body = "aligned", " \\\\ ".join("& " + ln for ln in lines)
    else:
        env, body = "gathered", " \\\\ ".join(lines)
    return "\\begin{%s}%s\\end{%s}" % (env, body, env)


def _align_from_asset(block: MathBlock, read_asset=None) -> str:
    """读这一块自己的裁剪图来量组内对齐；图拿不到就返回空串（用中性默认）。"""
    if read_asset is None or not block.asset_hash:
        return ""
    try:
        png = read_asset(block.asset_hash)
    except Exception:  # noqa: BLE001 - 归档缺资产不该让整本书停下来
        return ""
    if not png:
        return ""
    return extent_align(image_line_extents(png))


def _rejoin_caption_math(blocks: list[Block], read_asset=None) -> int:
    r"""把 caption 里被当成说明文字的**行间公式**并回公式本体。

    实测《几何不等式》冷岗松 p14：书上一条两行的公式

        x + y + z + 2\sqrt{xy} + 2\sqrt{xz} + 2\sqrt{yz}
        < \frac{5}{4}(a + b + c + 2\sqrt{ab} + 2\sqrt{bc} + 2\sqrt{ac}).

    OCR 把第一行写成 content、第二行写成 caption 里的
    `<inline_expr kind="\[">` —— 而两者**共用同一个 asset_hash**，
    也就是说那张裁剪图里本来就有两行。不并回去，渲染端会把它们画成两个
    各自独立的公式，正文里就此多出一条书上没有的式子。

    只认 `kind="\["`。说明文字里的公式是行内的 `\(`，用 `\[` 包起来的，
    是 OCR 自己认定它「独占一行」；全库 16,472 个公式块里有 1,881 处如此，
    抽查过的每一条都是同一显示块的续行或并列行。

    对齐要**趁合成前**定：合成之后整块只剩一个包围盒，行边界就只存在于
    那张裁剪图里了——所以这里要用 `read_asset` 现场量一次。
    """
    rejoined = 0
    for b in blocks:
        if not isinstance(b, MathBlock) or not b.caption:
            continue
        extra = [r.latex for r in b.caption if isinstance(r, MathRun) and r.display]
        if not extra:
            continue
        align = _align_from_asset(b, read_asset)
        merged = join_math_lines([b.latex] + extra, align)
        if not merged:
            continue
        b.latex = merged
        if align:
            b.align = align
        rest = _clean_runs(
            [r for r in b.caption if not (isinstance(r, MathRun) and r.display)]
        )
        b.caption = rest or None
        rejoined += 1
    return rejoined


# 同一显示块被切成上下两块时，纵向间距不会超过前一块自身的高度的一半。
# 实测：逗号收尾的相邻块 ratio 中位 0.46，句号收尾的中位 1.16。
_GROUP_GAP_RATIO = 0.5

# 行尾是「这行还没说完」的标点。句号才是终结。
_OPEN_TAIL = ",;"

# 已经合成过的多行组，尾巴是一个环境结束符，真正那一行的标点在它前面。
_TAIL_ENV_RE = re.compile(r"\\end\s*\{\s*[A-Za-z]*\s*\}\s*$")


def _tail_open(latex: str) -> bool:
    r"""行尾是否明确表示「后面还有」——只认逗号和分号。

    句号（中英文皆然）表示一条公式说完了，后面那块是另一条。没有标点的
    一律**不合并**：书上本来就有一大批 display 公式不带标点，把它们当作
    「未完」会让相邻的两条独立公式粘成一条。

    看之前要先剥两样东西，否则标点会被挡住：合成过的组以 `\end{gathered}`
    收尾，而 OCR 常在行末补一个空的 `{}` 占位。实测就是这样让三行的组
    只合了两行——第二行末尾那点被 `\end{gathered}` 盖住了。

    >>> _tail_open("a+b,"), _tail_open("a+b."), _tail_open("a+b")
    (True, False, False)
    >>> _tail_open("{a} {}")          # 剥掉尾部空占位，露出的是 '}'，不是标点
    False
    >>> _tail_open(r"\begin{gathered}a, \\ b,\end{gathered}")
    True
    >>> _tail_open(r"\begin{gathered}a, \\ b.\end{gathered}")
    False
    """
    t = latex.rstrip()
    m = _TAIL_ENV_RE.search(t)
    if m:
        t = t[: m.start()].rstrip()
    while t.endswith("{}"):
        t = t[:-2].rstrip()
    if not t:
        return True
    return t[-1] in _OPEN_TAIL


def _parse_bbox(s: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if not s:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in s.split(","))
    except (ValueError, AttributeError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# 组内对齐：文本层里没有、只在原图里有的那点信息
# ---------------------------------------------------------------------------

def _bbox_extent(bbox: Optional[str]) -> Optional[tuple[int, int]]:
    """包围盒的水平范围（x1, x2）。

    裁剪图是**紧贴内容**裁出来的——实测每个公式块的 bbox 宽高与它那张图的
    像素尺寸逐块相等——所以 x1/x2 就是那一行文字的左右边界，可以直接拿来
    做几何比较，不必再打开图。
    """
    parsed = _parse_bbox(bbox)
    return (parsed[0], parsed[2]) if parsed else None


def extent_align(extents: list[tuple[int, int]]) -> str:
    r"""一组行的水平范围 → 组内是怎么对齐的。

    居中排版的特征是「各行中心几乎重合、而左边界参差」；左对齐正好相反，
    各行左边界贴合而中心散开。所以比**两个极差谁更小**，而不设绝对阈值：
    绝对阈值要随字号和栏宽变，而两个极差同量纲，直接比大小就与字号无关了。

    >>> extent_align([(88, 796), (22, 1168)])           # 左边界差 66、中心差 153
    'left'
    >>> extent_align([(19, 1052), (219, 889)])          # 左边界差 200、中心差 19
    'center'
    >>> extent_align([(866, 1259), (872, 1255), (895, 1213)])
    'center'
    >>> extent_align([(10, 100)])                       # 单行谈不上组内对齐
    ''
    """
    if len(extents) < 2:
        return ""
    lefts = [e[0] for e in extents]
    centers = [(e[0] + e[1]) // 2 for e in extents]
    if max(centers) - min(centers) < max(lefts) - min(lefts):
        return "center"
    return "left"


# 裁剪图是白底黑字、紧贴内容的，行与行之间至少隔着一个纯白行带。
_ALIGN_INK = 200        # 灰度低于它算墨迹
_ALIGN_MIN_BAND = 4     # 细于这么多像素的行带当噪点丢掉
_ALIGN_BLANK_RUN = 2    # 连续这么多空行才算行间分隔


def image_line_extents(png: bytes) -> list[tuple[int, int]]:
    r"""从一张公式裁剪图里量出各行的左右边界（自上而下，单位像素）。

    为什么非要从图里量：OCR 把一个多行 display 公式归成**一个** equation 块时
    只给一个包围盒，行与行之间是顶格、居中、还是对着关系符缩进，文本层里一个
    字都没有——但图上白纸黑字地画着。

    没有 PIL、或者图本身打不开时返回空表，调用方退回中性默认（居中）：
    排版差一点可以接受，因为一张坏图崩掉整本书不行。
    """
    try:
        import io as _io

        from PIL import Image
    except ImportError:  # pragma: no cover - 取决于运行环境
        return []
    try:
        with Image.open(_io.BytesIO(png)) as handle:
            image = handle.copy()
    except Exception:  # noqa: BLE001 - 损坏的资产不该拖垮整本书
        return []

    if image.mode in ("RGBA", "LA", "PA"):
        # 透明像素灰度化后是 0（黑）。不合到白底的话，整幅图都会被当成墨迹。
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        image = background
    mask = image.convert("L").point(lambda v: 1 if v < _ALIGN_INK else 0)
    width, height = mask.size
    if not width or not height:
        return []

    # 二值图每像素 1 字节，bytes.count 在 C 里跑，比逐像素循环快两个数量级。
    raw = mask.tobytes()
    counts = [raw[y * width:(y + 1) * width].count(1) for y in range(height)]

    bands: list[tuple[int, int]] = []
    start: Optional[int] = None
    blank = 0
    for y, n in enumerate(counts):
        if n:
            if start is None:
                start = y
            blank = 0
        elif start is not None:
            blank += 1
            if blank >= _ALIGN_BLANK_RUN:
                end = y - blank + 1
                if end - start >= _ALIGN_MIN_BAND:
                    bands.append((start, end))
                start = None
                blank = 0
    if start is not None and height - start >= _ALIGN_MIN_BAND:
        bands.append((start, height))

    extents: list[tuple[int, int]] = []
    for y1, y2 in bands:
        # getbbox 判的是「非零像素」——白底 255 也是非零，所以必须先二值化，
        # 让墨迹=1、背景=0，否则整幅图都会被算成有内容。
        box = mask.crop((0, y1, width, y2)).getbbox()
        if box:
            extents.append((box[0], box[2]))
    return extents


_MULTILINE_ENV_RE = re.compile(
    r"\\begin\s*\{\s*(?:gathered|aligned|alignedat|gather|split|array|matrix|cases)\b"
)


def _is_multiline(latex: str) -> bool:
    r"""这段 LaTeX 渲染出来是不是多行的——组内对齐只对多行有意义。

    >>> _is_multiline(r"\begin{gathered}a \\ b\end{gathered}")
    True
    >>> _is_multiline(r"a \\ b")
    True
    >>> _is_multiline("a+b")
    False
    """
    return bool(_MULTILINE_ENV_RE.search(latex)) or "\\\\" in latex


def _stacks_below(prev: MathBlock, nxt: MathBlock) -> bool:
    """两块同页、上下紧挨着——在书上是同一个显示块的两行。

    间距拿**两块里较矮的那个**当尺子，而不是前一块。前一块在合并过程中
    会变成整组的高度，拿它当尺子，阈值就随行数一起放大，一组公式会贪婪地
    把后面所有挨得近的块都吞进来；行高才是这里稳定的量。
    """
    if prev.page != nxt.page:
        return False
    a, b = _parse_bbox(prev.bbox), _parse_bbox(nxt.bbox)
    if not a or not b:
        return False
    gap = b[1] - a[3]
    if gap < 0:
        return True                     # 包围盒纵向重叠：只可能同属一块
    ruler = min(a[3] - a[1], b[3] - b[1])
    return gap <= _GROUP_GAP_RATIO * max(ruler, 1)


def _union_bbox(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """两块合并后的包围盒。留着旧的那一个，下一块的间距就会按错的原点量。"""
    pa, pb = _parse_bbox(a), _parse_bbox(b)
    if not pa:
        return b
    if not pb:
        return a
    return ",".join(
        str(v) for v in (min(pa[0], pb[0]), min(pa[1], pb[1]),
                         max(pa[2], pb[2]), max(pa[3], pb[3]))
    )


def _group_adjacent_math(blocks: list[Block]) -> int:
    r"""把上下紧邻、前一块又没写完的若干 display 块合成一个多行公式。

    书上的一组推导演算常被 OCR 逐行切成多个 `display-formula`：每行各自都是
    完整的式子，行尾用逗号，最后一行用句号。它们本来就是**一个**显示块——
    《几何不等式》p14 那三行

        2\sqrt{xy} \leqslant 2x + \frac{1}{2}y,
        2\sqrt{xz} \leqslant 2x + \frac{1}{2}z,
        2\sqrt{yz} \leqslant y + z.

    就是这样被拆成三条独立公式的。

    判据要求「逗号/分号收尾」与「版面紧邻」**同时**成立：逗号是数学写作里
    明确的行间标点，句号才是终结；紧邻与否由 bbox 量出，不靠猜内容。
    宁可漏合也不可错合——把两条真正独立的公式粘成一条，读者看到的是一条
    从未在书上出现过的式子。
    """
    out: list[Block] = []
    grouped = 0
    i = 0
    total = len(blocks)
    while i < total:
        head = blocks[i]
        if not isinstance(head, MathBlock):
            out.append(head)
            i += 1
            continue
        # **先收齐整组，再一次性合成。** 一边判断一边改写的话，组里第一块
        # 已经变成 `\begin{gathered}...\end{gathered}`，下一轮的「行尾标点」
        # 就落在 `\end{gathered}` 上而不是真正那一行上，而且再 join 一次
        # 会套出 `\begin{gathered}\begin{gathered}...` 这种娃中娃。
        run = [head]
        j = i + 1
        # 入参里**本来就是多行**的块一律不参与合并。它的若干行已经合成一个
        # 环境（caption 归位那一步做的），再包一层就是
        # `\begin{gathered}\begin{gathered}…` 这样的同级套娃——KaTeX 对它
        # 直接报错（全库 22 处、13 本书）。run 里各块的 latex 到循环结束才被
        # 改写，所以这里看到的一直是**原始**内容，头一块是多行也不影响
        # 「本组内各块都还只是一行」这个前提。
        while (
            j < total
            and isinstance(blocks[j], MathBlock)
            and not _is_multiline(run[-1].latex)
            and not _is_multiline(blocks[j].latex)
            and _tail_open(run[-1].latex)
            and not is_blank_math(run[-1].latex)
            and _stacks_below(run[-1], blocks[j])
        ):
            run.append(blocks[j])
            j += 1
        if len(run) > 1:
            # 逐块的包围盒就是各行自己的范围（裁剪图紧贴内容），据此判组内对齐。
            # **必须在这里判**：块一合并，每块各自那张只有一行高的图就作废了
            # （asset_hash 会被清掉），之后再想量行边界就没有数据了。
            align = extent_align(
                [e for e in (_bbox_extent(r.bbox) for r in run) if e]
            )
            head.latex = join_math_lines([r.latex for r in run], align)
            if align:
                head.align = align
            for extra in run[1:]:
                head.eqno = head.eqno or extra.eqno
                head.bbox = _union_bbox(head.bbox, extra.bbox)
                if head.caption is None:
                    head.caption = extra.caption
                # 两张裁剪图各只覆盖自己那一行，合并之后原图已不等于整块
                # 内容。留着它，KaTeX 一旦编译不过就会拿第一行的图冒充整块
                # ——那比显示红色源码更糟，是静默丢内容。
                if head.asset_hash != extra.asset_hash:
                    head.asset_hash = None
            grouped += len(run) - 1
        out.append(head)
        i = j
    blocks[:] = out
    return grouped


# `\mathstrut` 是个看不见的支撑，手写公式里几乎不会用；它成批出现在 OCR
# 输出里，意思是「这里有个东西，但我认不出来」。
_JUNK_STRUT_RE = re.compile(r"\\mathstrut")

# 少数几个可能只是作者真用了它，成批出现才是整块失败。
_JUNK_STRUT_MIN = 3


def _mark_unreadable_math(blocks: list[Block]) -> int:
    r"""整块认不出来的公式改用原图——原图是 100% 忠实的，垃圾 LaTeX 不是。

    《几何不等式》p14 那三行推导被识别成一串
    `\vec{\mathstrut}\vec{\mathstrut\{{}\mathstrut}\mathstrut\mathstrut\downarrow}`
    ——连「⑤ 的左边」那几个汉字都在里面。它编译得过（`\mathstrut` 是合法
    命令），所以 KaTeX 那道闸门拦不住它，读者只会看到一堆莫名其妙的箭头和
    符号。这种块唯一的出路是 `.pcex` 里那张裁剪图，图上内容是完整的。

    只在**有原图可退**时才标记：退无可退的时候，留着原文总比留一片空白强。
    """
    marked = 0
    for b in blocks:
        if not isinstance(b, MathBlock) or b.force_image or not b.asset_hash:
            continue
        if len(_JUNK_STRUT_RE.findall(b.latex)) >= _JUNK_STRUT_MIN:
            b.force_image = True
            marked += 1
    return marked


# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------


@dataclass
class TocItem:
    id: str
    page_index: Optional[int]
    order: Optional[int]
    level: int
    title: str = ""
    children: list["TocItem"] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.page_index, self.order)


@dataclass
class Chapter:
    id: str
    level: int
    blocks: list[Block] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 公式的规范编号（verify 与渲染端必须共用这一份）
# ---------------------------------------------------------------------------


@dataclass
class MathRef:
    """一条公式在文档里的规范坐标。"""

    fid: str          # c07-m012（块级） / c07-i034（行内）
    kind: str         # display | inline
    latex: str
    obj: object       # MathBlock 或 MathRun，可直接改写 .latex
    chapter: int      # 1 起
    block: int        # 块下标，用来取上下文


def iter_math(chapters) -> list[MathRef]:
    """给文档里每条公式编一个稳定 id。

    **这是 verify.py 与 pcex2x.py 之间唯一的 id 约定，必须只有这一份实现。**

    两边各写一遍迟早会漂移，而漂移的后果很阴险：修正文件仍然能加载、仍然会套用，
    只是套到了另一条公式上——不报错、不崩溃，静默改坏正文。共用一份实现是唯一
    能保证「报告里说的哪条」就是「渲染时改的哪条」的办法。

    编号规则（与历史报告保持一致）：
      * 章序号 1 起，按 `pcex.chapters()` 的顺序；
      * 块级公式 `m###` 在章内递增；
      * 行内公式 `i###` 在章内递增，Para 与 Heading 共用同一个计数器。
    """
    refs: list[MathRef] = []
    for ci, ch in enumerate(chapters, 1):
        mi = ii = 0
        for bi, b in enumerate(ch.blocks):
            if isinstance(b, MathBlock):
                mi += 1
                refs.append(MathRef(f"c{ci:02d}-m{mi:03d}", "display", b.latex, b, ci, bi))
            elif isinstance(b, (Para, Heading)):
                for r in b.runs:
                    if isinstance(r, MathRun):
                        ii += 1
                        refs.append(MathRef(f"c{ci:02d}-i{ii:03d}", "inline", r.latex, r, ci, bi))
    return refs


def apply_corrections(chapters, corrections: dict) -> tuple[int, list[str], list[str]]:
    """把 `{fid: 修正}` 套用到已解析的章节上。

    返回 (套用数, 未命中的 fid, 原文对不上的 fid)。

    修正值有两种写法：
      * `"<修正后的 LaTeX>"` —— 只按 fid 套用；
      * `{"latex": ..., "original": ...}` —— **先核对原文再套用**，推荐这种。

    为什么必须支持核对原文：fid 是**位置**编号，IR 层一改（比如缝合规则变了）编号就会
    整体错位。实测踩过：旧报告里 4 条修正在新编号下只对上 2 条，另 2 条被静默丢弃。
    如果只按位置硬套，那 2 条就会**悄悄改到别的公式身上**——不报错、不崩溃，
    正文却被改坏，这是最难发现的一类事故。带上 `original` 就能把这种事变成一次响亮的跳过。

    未命中和原文不符都一定要报出来，不能吞。
    """
    refs = iter_math(chapters)
    by_fid = {r.fid: r for r in refs}
    applied = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for fid, spec in corrections.items():
        r = by_fid.get(fid)
        if r is None:
            missing.append(fid)
            continue
        if isinstance(spec, str):
            new, original = spec, None
        else:
            new = spec.get("latex")
            original = spec.get("original")
        if not new:
            missing.append(fid)
            continue
        if original is not None and _norm_latex(original) != _norm_latex(r.obj.latex):
            mismatched.append(fid)
            continue
        r.obj.latex = new
        applied += 1
    return applied, missing, mismatched


def _norm_latex(s: str) -> str:
    """比较原文时忽略空白差异——OCR 的换行/空格不稳定，但内容一致就是同一条。"""
    return " ".join((s or "").split())


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _display_formula_latex(content: Optional[ET.Element]) -> str:
    r"""取 `<content>` 里的 LaTeX。两种形态都要认：

    - **老管线**：LaTeX 直接写成文本节点（`<content>LATEX</content>`）。
    - **打过 pdf-craft `_normalize_equation` 补丁之后**：一个显示块里的多条续行
      合成一个公式，整个包在 `\[..\]` 里（`<inline_expr kind="\[">`）。

    为什么非要包起来：`<content>` 的文本要过 `_parse_block_content`，而那里只有
    定界表达式会被保护。裸文本会过 markdown，markdown 把 `\\` 折成一个 `\`——
    行分隔符就此消失、两行挤成一行（实测 4 个反斜杠进、3 个出）。
    """
    if content is None:
        return ""
    runs = _parse_runs(content)
    if not runs:
        return (content.text or "").strip()
    return " ".join(
        r.latex if isinstance(r, MathRun) else r.text for r in runs
    ).strip()


def _parse_runs(node: ET.Element, with_tail_of: Optional[ET.Element] = None):
    """把 <fragment>/<caption>/<content> 里的文本与 <inline_expr> 混排解析成 run 列表。

    ElementTree 把混排内容放在 ``node.text`` 与各子元素的 ``tail`` 上，需要手工遍历。
    """
    runs: list[Union[TextRun, MathRun]] = []

    def add_text(s: Optional[str]) -> None:
        if s:
            runs.append(TextRun(s))

    add_text(node.text)
    for child in node:
        if child.tag == "inline_expr":
            latex = (child.text or "").strip()
            if latex and not is_noise_formula(latex):
                runs.append(MathRun(latex, kind=child.get("kind") or r"\("))
        else:  # 未知标签：当纯文本处理，不丢内容
            add_text("".join(child.itertext()))
        add_text(child.tail)

    # 同一节点的首尾空白交给外层拼接逻辑处理
    return runs


def _merge_runs(
    a: list[Union[TextRun, MathRun]], b: list[Union[TextRun, MathRun]]
) -> list[Union[TextRun, MathRun]]:
    """合并两段 run；若两段之间是「字母/数字 ↔ 字母/数字」则补一个空格。

    中文不需要空格，英文断行需要。这样既不会把英文单词粘死，也不会在中文里塞空格。
    """
    if not a:
        return list(b)
    if not b:
        return list(a)
    merged = list(a)
    last, first = merged[-1], b[0]
    if isinstance(last, TextRun) and isinstance(first, TextRun):
        if last.text and first.text and re.search(r"[A-Za-z0-9]$", last.text) and re.match(
            r"[A-Za-z0-9]", first.text
        ):
            merged.append(TextRun(" "))
    merged.extend(b)
    return merged


def _clean_runs(runs: list[Union[TextRun, MathRun]]) -> list[Union[TextRun, MathRun]]:
    """合并相邻文本、去掉空 run、压缩连续空白（保留单个空格）。"""
    out: list[Union[TextRun, MathRun]] = []
    for r in runs:
        if isinstance(r, TextRun):
            t = re.sub(r"\s+", " ", r.text)
            if not t:
                continue
            if out and isinstance(out[-1], TextRun):
                out[-1] = TextRun(out[-1].text + t)
            else:
                out.append(TextRun(t))
        else:
            out.append(r)
    if out and isinstance(out[0], TextRun):
        out[0] = TextRun(out[0].text.lstrip())
    if out and isinstance(out[-1], TextRun):
        out[-1] = TextRun(out[-1].text.rstrip())
    return [r for r in out if not (isinstance(r, TextRun) and not r.text)]


# 有些「公式」其实只是把定界符本身识别了出来（例如孤立的一个 \[），
# 渲染出来是个空盒子，还会污染下游校验。这类条目应直接丢弃。
_NOISE_FORMULA_RE = re.compile(r"^(?:\\[\[\]\(\)]|\s)+$")


def is_noise_formula(latex: str) -> bool:
    """这条「公式」是否只是定界符噪声（内容全是 \\( \\[ \\) \\] 与空白）。"""
    s = (latex or "").strip()
    if not s:
        return True
    return bool(_NOISE_FORMULA_RE.match(s))


class Pcex:
    """只读的 .pcex 归档。"""

    def __init__(self, path: str):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self._names = set(self._zip.namelist())

    # -- 元数据 ------------------------------------------------------------
    @property
    def manifest(self) -> dict:
        return json.loads(self._zip.read("manifest.json").decode("utf-8"))

    @property
    def document_meta(self) -> dict:
        return self.manifest.get("document", {})

    @property
    def format_version(self) -> int:
        return int(self.manifest.get("format_version", 0))

    @property
    def producer(self) -> dict:
        return self.manifest.get("producer", {})

    # -- 页面 --------------------------------------------------------------
    @property
    def pages(self) -> dict[int, tuple[int, int]]:
        """页码 → (宽, 高)，单位是 OCR 像素。"""
        root = ET.fromstring(self._zip.read("pages.xml"))
        out = {}
        for p in root.findall("page"):
            out[int(p.get("index"))] = (int(p.get("width")), int(p.get("height")))
        return out

    # -- 目录 --------------------------------------------------------------
    def toc(self) -> list[TocItem]:
        """解析目录树。注意 toc.xml 不含标题文本，需用 _attach_toc_titles 补。"""
        raw = self._zip.read("toc.xml").decode("utf-8")
        root = ET.fromstring(raw)
        index: dict[tuple, TocItem] = {}

        def build(node: ET.Element) -> list[TocItem]:
            items = []
            for el in node.findall("item"):
                it = TocItem(
                    id=el.get("id", ""),
                    page_index=int(el.get("page_index")) if el.get("page_index") else None,
                    order=int(el.get("order")) if el.get("order") else None,
                    level=int(el.get("level", 0)),
                )
                index[it.key] = it
                it.children = build(el)
                items.append(it)
            return items

        top = build(root)
        self._toc_index = index
        return top

    # -- 正文 --------------------------------------------------------------
    def chapters(self, attach_toc_titles: bool = True) -> list[Chapter]:
        names = sorted(
            (n for n in self._names if n.startswith("chapters/") and n.endswith(".xml")),
            key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1))
            if re.search(r"(\d+)", n.split("/")[-1])
            else 0,
        )
        chapters: list[Chapter] = []
        headings_by_key: dict[tuple, Heading] = {}
        stitched = 0
        stripped = 0
        degraded = 0
        repaired = 0
        rejoined = 0
        grouped = 0
        unreadable = 0
        left_aligned = 0

        for name in names:
            root = ET.fromstring(self._zip.read(name).decode("utf-8"))
            ch = Chapter(id=root.get("id", ""), level=int(root.get("level", 0)))
            flow = root.find("flow")
            if flow is None:
                chapters.append(ch)
                continue
            for el in flow:
                blk = self._parse_block(el)
                if blk is None:
                    continue
                if isinstance(blk, Heading):
                    headings_by_key[(blk.page, blk.order)] = blk
                ch.blocks.append(blk)
            # 顺序要紧，各步各修一类毛病，后面的步骤依赖前面的结果：
            #   1. 公式文本本身（_heal_math_text）：剥定界符泄漏、把假公式
            #      退回正文、结构精修（环境配对、花括号、被折掉的行分隔）。
            #      其中的结构精修是唯一一步会增删 `{}` 的，它按公式自身的
            #      证据判定，所以必须**先于**下面几步——caption 归位要数括号、
            #      缝合要看环境有没有闭合，拿一份括号不平衡的文本去判，
            #      判出来的组是错的。剥定界符同样是为了这个：留着它，「未闭合」
            #      就是噪音，缝合会错缝；
            #   2. caption 归位——本块的第二行先回到 latex 里，否则下一步
            #      会把一条被拆开的公式当成两条分别看待；
            #   3. 缝合被版面切断的公式（结构信号：环境/定界符/行分隔/关系符）；
            #   4. 合成多行组（标点 + 版面几何）。
            # 2 和 4 都要在**合成之前**把组内对齐定下来：合成之后整块只剩一个
            # 包围盒，行边界要么在裁剪图里（2 读图）、要么在各块自己的包围盒里
            # （4 直接比），错过那一步就再也拿不到了。
            d_stripped, d_degraded, d_repaired = _heal_math_text(ch.blocks)
            stripped += d_stripped
            degraded += d_degraded
            repaired += d_repaired
            rejoined += _rejoin_caption_math(ch.blocks, self.read_asset)
            stitched += _stitch_split_formulas(ch.blocks)
            grouped += _group_adjacent_math(ch.blocks)
            # 放在合成之后：块 129 那种垃圾是 caption 里的行，只有归位之后
            # 才看得到；也放在分组之后，那时 asset_hash 的取舍已经落定。
            unreadable += _mark_unreadable_math(ch.blocks)
            left_aligned += sum(
                1
                for b in ch.blocks
                if isinstance(b, MathBlock) and b.align == "left"
            )
            chapters.append(ch)

        self._headings_by_key = headings_by_key
        self.stitched_environments = stitched
        self.stripped_delims = stripped
        self.degraded_to_text = degraded
        self.math_repaired = repaired
        self.caption_math_rejoined = rejoined
        self.multiline_grouped = grouped
        self.marked_unreadable = unreadable
        self.left_aligned_blocks = left_aligned
        if attach_toc_titles:
            try:
                self._attach_toc_titles(self.toc())
            except Exception:
                pass
        return chapters

    def _parse_block(self, el: ET.Element) -> Optional[Block]:
        tag = el.tag

        if tag == "text":
            role = el.get("role", "body")
            runs: list[Union[TextRun, MathRun]] = []
            first = None
            for frag in el.findall("fragment"):
                frag_runs = _parse_runs(frag)
                if first is None and frag_runs:
                    first = frag
                runs = _merge_runs(runs, frag_runs)
            runs = _clean_runs(runs)
            if not runs:
                return None
            page = int(first.get("page_index")) if first is not None and first.get("page_index") else None
            order = (
                int(first.get("source_order"))
                if first is not None and first.get("source_order")
                else None
            )
            bbox = first.get("bbox") if first is not None else None
            if role == "heading":
                return Heading(
                    level=int(el.get("level", 0)),
                    runs=runs,
                    page=page,
                    order=order,
                    bbox=bbox,
                )
            return Para(runs=runs, page=page, order=order, bbox=bbox)

        if tag == "display-formula":
            asset = el.find("asset")
            if asset is None:
                return None
            content = asset.find("content")
            latex = _display_formula_latex(content)
            if is_noise_formula(latex):
                return None
            latex, eqno = split_eqno(latex)
            cap_el = asset.find("caption")
            caption = _clean_runs(_parse_runs(cap_el)) if cap_el is not None else None
            return MathBlock(
                latex=latex,
                eqno=eqno,
                caption=caption,
                page=int(asset.get("page_index")) if asset.get("page_index") else None,
                order=int(asset.get("source_order")) if asset.get("source_order") else None,
                bbox=asset.get("bbox"),
                asset_hash=asset.get("asset_hash"),
            )

        if tag == "standalone-asset":
            asset = el.find("asset")
            if asset is None:
                return None
            content = asset.find("content")
            cap = _clean_runs(_parse_runs(content)) if content is not None else None
            return Figure(
                asset_hash=asset.get("asset_hash", ""),
                caption=cap,
                page=int(asset.get("page_index")) if asset.get("page_index") else None,
                order=int(asset.get("source_order")) if asset.get("source_order") else None,
                bbox=asset.get("bbox"),
            )

        return None

    def _attach_toc_titles(self, items: list[TocItem]) -> None:
        for it in items:
            h = self._headings_by_key.get(it.key)
            if h is not None:
                it.title = h.text
            self._attach_toc_titles(it.children)

    # -- 资产 --------------------------------------------------------------
    def read_asset(self, asset_hash: str) -> Optional[bytes]:
        # 归档里的资产名带扩展名（assets/<sha256>.png），同时容忍不带扩展名的写法
        for name in (f"assets/{asset_hash}", f"assets/{asset_hash}.png"):
            if name in self._names:
                return self._zip.read(name)
        return None

    @property
    def asset_names(self) -> list[str]:
        return sorted(n for n in self._names if n.startswith("assets/"))

    # -- 统计 --------------------------------------------------------------
    def stats(self) -> dict:
        chapters = self.chapters(attach_toc_titles=False)
        counters = {"heading": 0, "para": 0, "math": 0, "figure": 0}
        inline_math = 0
        eqno = 0
        for ch in chapters:
            for b in ch.blocks:
                if isinstance(b, Heading):
                    counters["heading"] += 1
                    inline_math += sum(1 for r in b.runs if isinstance(r, MathRun))
                elif isinstance(b, Para):
                    counters["para"] += 1
                    inline_math += sum(1 for r in b.runs if isinstance(r, MathRun))
                elif isinstance(b, MathBlock):
                    counters["math"] += 1
                    if b.eqno:
                        eqno += 1
                    if b.caption:
                        inline_math += sum(1 for r in b.caption if isinstance(r, MathRun))
                elif isinstance(b, Figure):
                    counters["figure"] += 1
                    if b.caption:
                        inline_math += sum(1 for r in b.caption if isinstance(r, MathRun))
        return {
            "chapters": len(chapters),
            "blocks": counters,
            "inline_math_runs": inline_math,
            "numbered_equations": eqno,
            "assets": len(self.asset_names),
        }

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "Pcex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    import sys

    p = Pcex(sys.argv[1])
    print("producer :", p.producer)
    print("fmt ver  :", p.format_version)
    print("meta     :", p.document_meta)
    print("pages    :", len(p.pages))
    print("stats    :", json.dumps(p.stats(), ensure_ascii=False))
```

### 6. pipeline/pcex2x.py

```python
#!/usr/bin/env python3
"""pcex2x — 把 .pcex 抽取归档再渲染成 EPUB3 / LaTeX / Markdown / HTML 预览。

整条流水线的契约层是 .pcex：公式已经是 LaTeX 源码、文本带 bbox、图片按 hash 独立。
本工具只做「IR → 目标格式」，不重跑 OCR。

公式在 EPUB 里的表示（本项目的核心设计）
----------------------------------------
    <span class="math-inline">\\(P(x,y)\\)</span>
    <div class="math-block">
      <span class="math-body">\\[ P(x,y)\\mathrm{d}x + Q(x,y)\\mathrm{d}y = 0 \\]</span>
      <span class="eqno">(2.15)</span>
    </div>

即**公式以 LaTeX 源码内联在正文流里**，用标准定界符包裹。这样：

1. EPUB 本身是完全合法的 EPUB3 / XHTML，任何阅读器都能打开；
2. 普通阅读器降级显示为可读的 `\\(P(x,y)\\)` 源码，而不是空白或乱码；
3. 我们的 KaTeX 阅读器只需对该元素调用 renderMathInElement 即可渲染成印刷级公式；
4. 公式**保持可逆**——源码原样保留，未来可再转 MathML、SVG 或 PDF。

用法
----
    python pcex2x.py book.pcex --outdir out --formats epub,tex,md \\
        --title "常微分方程教程（第三版）" --author 丁同仁 --author 李承治
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 本机直接跑（GBK/cp936 终端）时，stdout 默认编码打不出 ✓ ★ 这类符号，会在成功末尾
# 抛 UnicodeEncodeError 让脚本以非零退出——batch.py 会误判「转换失败」。
# run.py 会把 stdout 换成 StringIO（没有 .buffer 属性），那一路已经能正常编码，必须跳过，
# 否则调 reconfigure 会 AttributeError。只在「真·文件/管道」上重包成 utf-8。
try:
    if getattr(sys.stdout, "buffer", None) is not None and sys.stdout.encoding:
        if sys.stdout.encoding.replace("-", "").lower() not in ("utf8", "utf_8"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import texfix as tx  # noqa: E402
from pcexlib import (  # noqa: E402
    Chapter,
    Figure,
    Heading,
    MathBlock,
    MathRun,
    Para,
    Pcex,
    TextRun,
    apply_corrections,
    iter_math,
)

Run = object  # Union[TextRun, MathRun]

MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# ---------------------------------------------------------------------------
# 「公式的內容其实是一张图」
# ---------------------------------------------------------------------------

# DeepSeek-OCR 遇到复杂到它不愿转写的公式时，不返回 LaTeX，而是返回一个指向裁剪
# 图的引用，在正文里长成 `![]('img_url')`。pdf-craft 把它当普通公式内容存了下来。
# 不特判的后果是：渲染端把这串字面量当 LaTeX 输出，读者读到的是一行垃圾文本——
# 而那张裁剪图就躺在归档的 assets/ 里，白白浪费。
_IMG_PLACEHOLDER_RE = re.compile(r"!\[\]\s*\(\s*['\"]?img_url['\"]?\s*\)")

# 渲染过程中累计丢弃/还原了多少条，最后打印出来——静默吞掉内容是最坏的做法
PLACEHOLDER_STATS = {
    "display_restored": 0,
    "display_lost": 0,
    "inline_dropped": 0,
    "display_katex_fallback": 0,
}


def is_image_placeholder(latex: str) -> bool:
    return bool(_IMG_PLACEHOLDER_RE.search(latex))


def image_src(asset_hash: str) -> str:
    return f"../images/{asset_hash}.png"


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def xesc(s: str) -> str:
    """XHTML 文本节点转义。LaTeX 里的 \\ { } 都是合法字符，不用动；
    只有 & < > 会破坏 XML 解析，必须转义。"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# OCR（pdf-craft）会把 markdown 加粗原样留在正文里，`**定义 2**` 渲染成 XHTML
# 就是个字面星号——阅读器里读者直接看到 `**`。实测产出库 116 本 / 15015 处
# （最多一本 1627 处），是系统性残留，不是个别现象。
#
# 判据刻意保守：**成对**、内容**不含 * < >**、长度 ≤60、**两端非空白**。
# 数学正文里几乎不可能出现这种形状——`a**b` 这类乘法不会「无空格成对地
# 包住一段文字」。宁可漏（漏了只是仍旧显示星号），不可误伤正文。
_MD_BOLD = re.compile(r"\*\*(?!\s)([^*<>\n]{1,60}?)(?<!\s)\*\*")


def md_bold(s: str) -> str:
    """把 OCR 残留的 markdown 加粗转成真正的 <strong>。

    ★ 调用前必须已经过 xesc（本函数只在已转义的文本上做替换，产物里
      只会多出 <strong> 标签本身，不会引入未转义的 & < >）。
    """
    return _MD_BOLD.sub(r"<strong>\1</strong>", s)


def aesc(s: str) -> str:
    """XHTML 属性值转义 —— 比文本节点多一条：引号必须转义。

    属性是 `alt="..."` 包起来的，所以正文里任何一个英文双引号都会提前闭合
    属性，剩下的文字被当成属性名解析，整章立刻 not well-formed。
    classifications2000 的 ch017 就是这么坏的：某页 OCR 正文里出现了
    `"special"`，整章 66 KB 直接解析失败。文本节点里 `"` 本来合法，
    所以 xesc() 不能顺手改，得单独有这个函数。"""
    return xesc(s).replace('"', "&quot;")


# alt 是给无障碍朗读和碎裂图片占位用的，一句话就够。pdf-craft 有时把整页
# OCR 正文塞进 Figure.caption，几千字的 alt 没人看得见，却会让朗读器念上半天；
# 真正的内容交给 <figcaption>。截断也更安全 —— 短字符串更难带出坏字符。
ALT_MAX = 200


def alt_text(caption: str, fallback: str) -> str:
    s = " ".join((caption or "").split())
    if not s:
        return fallback
    return s if len(s) <= ALT_MAX else s[:ALT_MAX].rstrip() + "…"


def tesc(s: str) -> str:
    """LaTeX 文本转义（用于 .tex 输出的正文）。"""
    out = []
    for ch in s:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def mesc(s: str) -> str:
    """Markdown 转义：只处理会被解析成标记的字符。"""
    return re.sub(r"([\\`*_\[\]])", r"\\\1", s)


def stable_uuid(seed: str) -> str:
    """由内容派生稳定 UUID —— 同一本书重复生成不会被认为是新书。"""
    return str(uuid.UUID(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]))


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# run 级渲染
# ---------------------------------------------------------------------------


def runs_to_xhtml(runs, trace: bool = False) -> str:
    out = []
    for r in runs:
        if isinstance(r, TextRun):
            out.append(md_bold(xesc(r.text)))
        elif isinstance(r, MathRun):
            if is_image_placeholder(r.latex):
                # 行内公式没有 asset_hash，还原不了图；只能丢掉那个标记本身。
                # 留着它等于在句子中间插一句 `![]('img_url')`，比没有更糟。
                PLACEHOLDER_STATS["inline_dropped"] += 1
                continue
            left, right = r.delimiter
            body = xesc(f"{left}{r.latex}{right}")
            cls = "math-inline" + (" math-display" if r.display else "")
            out.append(f'<span class="{cls}">{body}</span>')
    return "".join(out)


def runs_to_tex(runs) -> str:
    out = []
    for r in runs:
        if isinstance(r, TextRun):
            out.append(tesc(r.text))
        elif isinstance(r, MathRun):
            if is_image_placeholder(r.latex):
                continue
            if r.display:
                out.append("\n\\[\n%s\n\\]\n" % r.latex)
            else:
                out.append("$%s$" % r.latex)
    return "".join(out)


def runs_to_md(runs) -> str:
    out = []
    for r in runs:
        if isinstance(r, TextRun):
            out.append(mesc(r.text))
        elif isinstance(r, MathRun):
            if is_image_placeholder(r.latex):
                continue
            out.append("$$%s$$" % r.latex if r.display else "$%s$" % r.latex)
    return "".join(out)


def runs_plain(runs) -> str:
    return "".join(r.text for r in runs if isinstance(r, TextRun)).strip()


# ---------------------------------------------------------------------------
# 块级渲染：XHTML
# ---------------------------------------------------------------------------


def heading_tag(level: int) -> str:
    return f"h{min(max(level, 0) + 1, 6)}"


def block_to_xhtml(b, trace: bool = False) -> str:
    tattr = ""
    if trace and getattr(b, "page", None):
        tattr = f' data-src-page="{b.page}"'

    if isinstance(b, Heading):
        return f"<{heading_tag(b.level)}{tattr}>{runs_to_xhtml(b.runs)}</{heading_tag(b.level)}>"

    if isinstance(b, Para):
        return f"<p{tattr}>{runs_to_xhtml(b.runs)}</p>"

    if isinstance(b, MathBlock):
        # 两种「用图代替 LaTeX」的情形：
        #   1. 内容本来就是图——OCR 只给了个 `![]('img_url')` 引用标记；
        #   2. KaTeX 编译不过——多行 array 阵列（三行 ∇ 公式那种）会被 OCR 逐格
        #      切碎，规则救不回来，而 .pcex 里存着裁剪图，原图 100% 忠实。
        # 交付成品里因此不再出现红色报错源码。
        if is_image_placeholder(b.latex) or b.force_image:
            if not b.asset_hash:
                PLACEHOLDER_STATS["display_lost"] += 1
                return ""
            if b.force_image:
                PLACEHOLDER_STATS["display_katex_fallback"] += 1
            else:
                PLACEHOLDER_STATS["display_restored"] += 1
            cap = runs_plain(b.caption) if b.caption else ""
            alt = aesc(alt_text(cap, "公式"))
            out = [f'<figure class="fig"{tattr}>']
            out.append(f'<img src="{image_src(b.asset_hash)}" alt="{alt}" />')
            if cap:
                out.append(f"<figcaption>{xesc(cap)}</figcaption>")
            out.append("</figure>")
            return "".join(out)

        # 书上的推导组是通栏左对齐的，KaTeX 的 display 却默认整块居中，
        # 所以「靠左」这件事要同时落在两处：LaTeX 里用 aligned（组内对齐，
        # 见 pcexlib.join_math_lines）、这里再用一个类把整块按到版心左边。
        klass = "math-block align-left" if b.align == "left" else "math-block"
        parts = [f'<div class="{klass}"{tattr}>']
        parts.append(f'<span class="math-body">{xesc(r"\[" + b.latex + r"\]")}</span>')
        if b.eqno:
            parts.append(f'<span class="eqno">{xesc(b.eqno)}</span>')
        parts.append("</div>")
        if b.caption:
            parts.append(f'<p class="math-caption">{runs_to_xhtml(b.caption)}</p>')
        return "".join(parts)

    if isinstance(b, Figure):
        cap = runs_plain(b.caption) if b.caption else ""
        alt = aesc(alt_text(cap, "插图"))
        src = f"../images/{b.asset_hash}.png"
        out = ['<figure class="fig">']
        out.append(f'<img src="{src}" alt="{alt}" />')
        if cap:
            out.append(f"<figcaption>{md_bold(xesc(cap))}</figcaption>")
        out.append("</figure>")
        return "".join(out)

    return ""


def chapter_to_xhtml(ch: Chapter, title: str, lang: str, trace: bool = False) -> str:
    body = "\n".join(block_to_xhtml(b, trace) for b in ch.blocks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">
<head>
  <meta charset="utf-8" />
  <title>{xesc(title)}</title>
  <link rel="stylesheet" type="text/css" href="../styles/reader.css" />
</head>
<body>
<section epub:type="chapter" class="chapter" id="ch{ch.id}">
{body}
</section>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 目录（nav.xhtml）
# ---------------------------------------------------------------------------


def build_nav(
    chapters: list[Chapter],
    toc_items: list | None,
    title: str,
    lang: str,
    max_level: int = 1,
) -> str:
    """生成 EPUB3 导航文档。

    优先用 toc.xml 的层级（它有 pdf-craft 判定出的父子关系），标题文本通过
    `(page_index, order)` 与正文 heading 对齐。若 toc 匹配率太低则退回
    「按 heading level 平铺」——后者顺序与文字一定正确，只是层级更扁。
    """
    items: list[tuple[int, str, str]] = []  # (level, text, href)

    if toc_items:
        flat: list = []

        def walk(nodes):
            for it in nodes:
                flat.append(it)
                walk(it.children)

        walk(toc_items)
        matched = [it for it in flat if it.title]
        if len(matched) >= max(3, int(0.5 * len(flat))):
            for it in flat:
                if not it.title or it.level > max_level:
                    continue
                href = None
                items.append((it.level, it.title, href or ""))

    if not items:
        for i, ch in enumerate(chapters, 1):
            for b in ch.blocks:
                if isinstance(b, Heading) and b.level <= max_level:
                    items.append((b.level, b.text, f"text/ch{i:03d}.xhtml"))

    # 去空、去重（同名相邻项）
    seen = set()
    cleaned = []
    for lv, txt, href in items:
        if not txt:
            continue
        key = (txt, href)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((lv, txt, href))

    if not cleaned:  # 兜底：直接列章节
        cleaned = [(0, f"第 {i} 节", f"text/ch{i:03d}.xhtml") for i in range(1, len(chapters) + 1)]

    # 用真正的递归建立嵌套 <ol>。
    # 手写状态机在「层级回退」时极易产生不闭合标签——实测会让 nav.xhtml 变成
    # 非法 XML，EPUB 直接打不开。递归是唯一稳的写法。
    norm: list[tuple[int, str, str]] = []
    prev = -1
    for lv, txt, href in cleaned:
        if lv > prev + 1:  # 归一化：不允许跳级（0 → 3 会破坏嵌套结构）
            lv = prev + 1
        lv = max(0, lv)
        norm.append((lv, txt, href))
        prev = lv

    def build(items, pos, level):
        html = ["<ol>"]
        while pos < len(items):
            lv, txt, href = items[pos]
            if lv < level:
                break
            label = (
                f'<a href="{aesc(href)}">{md_bold(xesc(txt))}</a>'
                if href
                else f"<span>{md_bold(xesc(txt))}</span>"
            )
            if pos + 1 < len(items) and items[pos + 1][0] > level:
                sub, pos = build(items, pos + 1, items[pos + 1][0])
                html.append(f"<li>{label}{sub}</li>")
            else:
                html.append(f"<li>{label}</li>")
                pos += 1
        html.append("</ol>")
        return "".join(html), pos

    toc_html = "  " + (build(norm, 0, norm[0][0])[0] if norm else "<ol></ol>")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">
<head>
  <meta charset="utf-8" />
  <title>目录</title>
  <link rel="stylesheet" type="text/css" href="styles/reader.css" />
</head>
<body>
<nav epub:type="toc" id="toc">
  <h1 class="toc-title">目录</h1>
{toc_html}
</nav>
<nav epub:type="landmarks" hidden="hidden">
  <ol>
    <li><a epub:type="toc" href="nav.xhtml">目录</a></li>
  </ol>
</nav>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# EPUB 打包
# ---------------------------------------------------------------------------

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def build_opf(
    title: str,
    authors: list[str],
    lang: str,
    book_uuid: str,
    n_chapters: int,
    image_hashes: list[str],
    extra_meta: dict | None = None,
    has_math: bool = True,
) -> str:
    meta = extra_meta or {}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="{lang}">',
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">',
        f'    <dc:identifier id="bookid">urn:uuid:{book_uuid}</dc:identifier>',
        f"    <dc:title>{xesc(title)}</dc:title>",
        f"    <dc:language>{lang}</dc:language>",
    ]
    for a in authors:
        lines.append(f"    <dc:creator>{xesc(a)}</dc:creator>")
    if meta.get("publisher"):
        lines.append(f'    <dc:publisher>{xesc(meta["publisher"])}</dc:publisher>')
    if meta.get("description"):
        lines.append(f'    <dc:description>{xesc(meta["description"])}</dc:description>')
    lines.append(f'    <meta property="dcterms:modified">{now_iso()}</meta>')

    # 声明公式能力：告知阅读器本书含数学内容（也便于我们自己的阅读器识别）
    if has_math:
        lines.append(
            '    <meta property="schema:accessibilityFeature">MathML</meta>'
        )
        lines.append(
            '    <meta property="schema:accessibilityFeature">structuralNavigation</meta>'
        )
    lines.append("  </metadata>")

    lines.append("  <manifest>")
    lines.append('    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')
    lines.append('    <item id="css" href="styles/reader.css" media-type="text/css"/>')
    for i in range(1, n_chapters + 1):
        lines.append(
            f'    <item id="ch{i:03d}" href="text/ch{i:03d}.xhtml" media-type="application/xhtml+xml"/>'
        )
    for h in image_hashes:
        lines.append(f'    <item id="img-{h[:12]}" href="images/{h}.png" media-type="image/png"/>')
    lines.append("  </manifest>")

    lines.append('  <spine page-progression-direction="ltr">')
    for i in range(1, n_chapters + 1):
        lines.append(f'    <itemref idref="ch{i:03d}"/>')
    lines.append("  </spine>")
    lines.append("</package>")
    return "\n".join(lines)


def write_epub(
    out_path: Path,
    xhtml_pages: list[tuple[str, str]],
    nav_xhtml: str,
    css: str,
    opf_xml: str,
    assets: dict[str, bytes],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        # mimetype 必须第一个条目且不压缩
        info = zipfile.ZipInfo("mimetype", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, "application/epub+zip")
        z.writestr("META-INF/container.xml", CONTAINER_XML)
        z.writestr("OEBPS/content.opf", opf_xml)
        z.writestr("OEBPS/nav.xhtml", nav_xhtml)
        z.writestr("OEBPS/styles/reader.css", css)
        for href, content in xhtml_pages:
            z.writestr(f"OEBPS/{href}", content)
        for name, data in assets.items():
            z.writestr(f"OEBPS/images/{name}", data)


# ---------------------------------------------------------------------------
# LaTeX / Markdown 输出
# ---------------------------------------------------------------------------

PREAMBLE = r"""% !TEX program = xelatex
% 本文件由 pcex2x 从 pdf-craft 的 .pcex 抽取归档自动生成。
% 公式为 OCR 识别结果，可能存在错误，请以原书为准。
% 编译：xelatex <file>.tex   （需要 ctex 宏包）
\documentclass[11pt,a4paper]{ctexart}

\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{float}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}

\graphicspath{{images/}}
\setlength{\parindent}{2em}
\linespread{1.25}

\title{@@TITLE@@}
\author{@@AUTHORS@@}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


def tex_heading(level: int, text: str) -> str:
    cmd = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
    return "\\%s*{%s}" % (cmd[min(max(level, 0), len(cmd) - 1)], tesc(text))


def build_tex(
    chapters: list[Chapter],
    title: str,
    authors: list[str],
    include_figures: bool = True,
) -> str:
    parts = [
        PREAMBLE.replace("@@TITLE@@", tesc(title)).replace(
            "@@AUTHORS@@", r" \and ".join(tesc(x) for x in authors)
        )
    ]
    for ch in chapters:
        parts.append("\n%% ---------- chapter %s ----------\n" % ch.id)
        for b in ch.blocks:
            if isinstance(b, Heading):
                parts.append(tex_heading(b.level, runs_plain(b.runs)) + "\n")
            elif isinstance(b, Para):
                parts.append(runs_to_tex(b.runs) + "\n")
            elif isinstance(b, MathBlock):
                if is_image_placeholder(b.latex):
                    if b.asset_hash:
                        parts.append(
                            "\\begin{figure}[H]\n  \\centering\n  \\includegraphics[width=0.8\\linewidth]{%s.png}\n\\end{figure}\n"
                            % b.asset_hash
                        )
                    continue
                latex = b.latex + (r" \quad " + b.eqno if b.eqno else "")
                parts.append("\\[\n%s\n\\]\n" % latex)
                if b.caption:
                    parts.append(runs_to_tex(b.caption) + "\n")
            elif isinstance(b, Figure) and include_figures:
                cap = runs_plain(b.caption) if b.caption else ""
                parts.append(
                    "\\begin{figure}[H]\n  \\centering\n  \\includegraphics[width=0.75\\linewidth]{%s}\n  \\caption{%s}\n\\end{figure}\n"
                    % (b.asset_hash + ".png", tesc(cap))
                )
    parts.append("\n\\end{document}\n")
    return "".join(parts)


def build_markdown(chapters: list[Chapter], title: str, authors: list[str]) -> str:
    parts = [f"# {title}\n"]
    if authors:
        parts.append("\n> " + "、".join(authors) + "\n")
    parts.append("\n---\n")
    for ch in chapters:
        for b in ch.blocks:
            if isinstance(b, Heading):
                parts.append("\n" + "#" * min(max(b.level, 0) + 2, 6) + " " + runs_plain(b.runs) + "\n")
            elif isinstance(b, Para):
                parts.append("\n" + runs_to_md(b.runs) + "\n")
            elif isinstance(b, MathBlock):
                if is_image_placeholder(b.latex):
                    if b.asset_hash:
                        cap = runs_plain(b.caption) if b.caption else ""
                        parts.append(f"\n![{cap}](images/{b.asset_hash}.png)\n")
                    continue
                latex = b.latex + (r" \quad " + b.eqno if b.eqno else "")
                parts.append("\n$$\n%s\n$$\n" % latex)
                if b.caption:
                    parts.append("\n" + runs_to_md(b.caption) + "\n")
            elif isinstance(b, Figure):
                cap = runs_plain(b.caption) if b.caption else ""
                parts.append(f"\n![{cap}](images/{b.asset_hash}.png)\n")
                if cap:
                    parts.append(f"*{cap}*\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

DEFAULT_CSS = Path(__file__).resolve().parent / "assets" / "reader.css"


def katex_failed_fids(chapters, katex_dir: str | None) -> set[str] | None:
    """跑一遍 KaTeX，返回**编译失败的块级公式** fid 集合；跑不了时返回 None。

    只用 KaTeX 判、不用启发式：实测被切碎的 array 阵列里，碎片有的括号平衡、
    有的不平衡，数括号既会漏也会误判。KaTeX 是唯一权威——它正是阅读器里真正
    要执行的那个解析器。

    注意调用时机必须在**套用修正之后**：一条公式被人工改好后能编译通过，
    就不该再被图顶替，否则白修。
    """
    try:
        import verify as V  # 同目录；verify.py 是可安全导入的（argparse 在 main 里）
    except Exception as e:  # noqa: BLE001
        print(f"    [i] 无法导入 verify 模块，跳过「失败公式转原图」：{e}")
        return None

    katex_js = V.find_katex_js(katex_dir)
    node = V.find_node()
    if not katex_js or not node:
        print("    [i] 未找到 KaTeX 或 node，跳过「失败公式转原图」")
        return None

    refs = iter_math(chapters)
    # 块级和行内都要查。块级失败的处置是「退回裁剪图」，行内失败的处置是
    # 「判断它到底是不是公式」——后者的结论同样要用这里的编译结果，
    # 因为「不是公式」这个判断只在 KaTeX 已经说它不成立之后才成立。
    formulas = [
        V.Formula(fid=r.fid, chapter=r.chapter, kind=r.kind, latex=r.latex)
        for r in refs
    ]
    ok, msg = V.l0_katex(formulas, katex_js, node)
    if not ok:
        print(f"    [i] KaTeX 校验未完成，跳过「失败公式转原图」：{msg}")
        return None
    return {
        f.fid
        for f in formulas
        if any(i.code == "katex-parse-error" for i in f.issues)
    }


def mark_failed_math(chapters, katex_dir: str | None) -> None:
    """把 KaTeX 编译不过的公式收尾：块级退回原图，行内退回文本。

    KaTeX 只跑一遍——两种失败的结论出自同一次编译，分开跑等于把最贵的一步
    （把全书的公式喂给 node）做两遍。
    """
    failed = katex_failed_fids(chapters, katex_dir)
    if failed is None:
        return
    _mark_failed_display(chapters, failed)
    _mark_failed_inline(chapters, failed)


def _mark_failed_display(chapters, failed: set[str]) -> None:
    """块级公式编译不过 → 用归档里那张裁剪图渲染。"""
    marked = skipped = 0
    for r in iter_math(chapters):
        if r.kind != "display" or r.fid not in failed:
            continue
        blk = r.obj
        if blk.asset_hash:
            blk.force_image = True
            marked += 1
        else:
            # 没有裁剪图就只能留着 LaTeX：读者端 KaTeX 会把它显示成红色源码，
            # 那是一个可见的质量信号，比悄悄丢掉好。
            skipped += 1
    if marked or skipped:
        print(f"    KaTeX 编译不过的块级公式：{marked} 条改用原图"
              + (f"，{skipped} 条无裁剪图保留 LaTeX" if skipped else ""))


def _mark_failed_inline(chapters, failed: set[str]) -> None:
    r"""行内公式编译不过 → 判它是不是公式；不是就退回文本。

    行内公式**没有退路**：它在 `.pcex` 里没有自己的 bbox，裁不出原图，所以
    编译不过就是正文正中一段红色报错。能做的只有一件事——问它到底是不是公式。
    OCR 会把散文、表格、页码吞进数学模式（`\mathrm{We~exercise~7.5.4.}}`、
    整栏的 `1111111111`），这些东西继续修 LaTeX 无从修起，退回文本才是它们
    本来的样子；能从里面捞回可读字的就留字，捞不回的丢掉——摆一段乱码在
    正文里和摆一段红色报错一样没用。

    是公式但修不好的（`\llangledown`、`A_\vec{x}`）**一律留着**。红色报错是
    个可见的质量信号，而「猜一个符号填进去」是静默改坏公式，比红字坏得多。
    """
    targets: dict[int, str] = {}
    for r in iter_math(chapters):
        if r.kind != "inline" or r.fid not in failed:
            continue
        if not tx.is_not_math(r.latex):
            continue
        targets[id(r.obj)] = tx.salvage_text(r.latex)
    if not targets:
        return

    salvaged = dropped = 0
    for ch in chapters:
        for b in ch.blocks:
            runs = getattr(b, "runs", None)
            if not runs or not any(id(x) in targets for x in runs):
                continue
            out = []
            for x in runs:
                if isinstance(x, MathRun) and id(x) in targets:
                    text = targets[id(x)]
                    if text:
                        out.append(TextRun(text))
                        salvaged += 1
                    else:
                        dropped += 1
                    continue
                out.append(x)
            b.runs = out
    print(f"    KaTeX 编译不过的行内公式：{salvaged} 条退回文本"
          + (f"，{dropped} 条纯乱码丢弃" if dropped else ""))



def main() -> int:
    ap = argparse.ArgumentParser(
        description="pcex → EPUB3 / LaTeX / Markdown / HTML 预览",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pcex", help="输入 .pcex 归档")
    ap.add_argument("--outdir", "-o", default="out", help="输出目录")
    ap.add_argument(
        "--formats",
        default="epub,tex,md",
        help="逗号分隔：epub,tex,md,html（html 需要 --katex 指向 KaTeX dist）",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--author", action="append", default=[])
    ap.add_argument("--language", default="zh")
    ap.add_argument("--publisher", default=None)
    ap.add_argument("--toc-max-level", type=int, default=1, help="目录收录到第几层")
    ap.add_argument("--use-toc", action="store_true", help="使用 toc.xml 的层级（默认从正文 heading 扁平生成）")
    ap.add_argument("--trace", action="store_true", help="在 XHTML 里写 data-src-page 便于溯源")
    ap.add_argument("--no-figures", action="store_true", help="TeX 输出不含插图")
    ap.add_argument("--katex", default=None, help="KaTeX dist 目录（用于 html 预览）")
    ap.add_argument(
        "--no-auto-image",
        action="store_true",
        help="不要用 KaTeX 预检失败公式并改用原图（默认会做，需要能跑 node）",
    )
    ap.add_argument("--stem", default=None, help="输出文件名主干，默认取输入名")
    ap.add_argument("--corrections", default=None,
                    help="公式修正叠加层（JSON：{fid: 修正后的 LaTeX}）。不改 .pcex，渲染时套用")
    a = ap.parse_args()

    src = Path(a.pcex)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = a.stem or src.stem
    formats = {f.strip() for f in a.formats.split(",") if f.strip()}

    with Pcex(str(src)) as p:
        meta = p.document_meta or {}
        title = a.title or meta.get("title") or stem
        authors = a.author or list(meta.get("authors") or [])
        chapters = p.chapters()

        # 人工/模型修正过的公式。保持 .pcex 原样不动——它是 OCR 的原始记录，
        # 修正以「叠加层」的形式在渲染时套用，这样才能随时重渲染、也能一眼看出
        # 哪些公式是被人改过的。
        if a.corrections:
            cpath = Path(a.corrections)
            data = json.loads(cpath.read_text(encoding="utf-8"))
            # 允许两种写法：{"fid": latex} 或 {"books": {stem: {"fid": latex}}}
            if "books" in data and isinstance(data["books"], dict):
                data = data["books"].get(stem) or data["books"].get(src.stem) or {}
            applied, missing, mismatched = apply_corrections(chapters, data)
            print(f"    套用修正 {applied} 条（来自 {cpath.name}）")
            if missing:
                print(f"    [!] {len(missing)} 条修正没找到对应公式（报告与 .pcex 不同步）："
                      f"{', '.join(missing[:8])}")
            if mismatched:
                print(f"    [!] {len(mismatched)} 条修正的原文与当前公式不符，已跳过（宁可不改也不能改错）："
                      f"{', '.join(mismatched[:8])}")

        # 编译失败的公式收尾：必须放在「套用修正」之后——一条被人工改好、因而
        # 能被 KaTeX 放行的公式，不该再被图顶替（那样白改）。块级退回裁剪图、
        # 行内退回文本；本机跑不了 KaTeX（无 node / 无 katex.js）会静默跳过，
        # 绝不让这步卡住整条流水线。
        if not a.no_auto_image:
            mark_failed_math(chapters, a.katex)

        toc_items = None
        if a.use_toc:
            try:
                toc_items = p.toc()
            except Exception as e:  # toc.xml 缺失或损坏时退回 heading 方案
                print(f"[warn] toc.xml 解析失败，改用 heading：{e}")
        stats = p.stats()
        lang = a.language or meta.get("language") or "zh"

        print(f"[i] {src.name}")
        print(f"    章 {stats['chapters']} · 段 {stats['blocks']['para']} · "
              f"块级公式 {stats['blocks']['math']} · 行内公式 {stats['inline_math_runs']} · "
              f"插图 {stats['blocks']['figure']} · 编号公式 {stats['numbered_equations']}")
        stitched = getattr(p, "stitched_environments", 0)
        if stitched:
            print(f"    缝合被版面切成两块的公式 {stitched} 处（还原成书上的一条公式）")

        # 抽取图片资产
        assets: dict[str, bytes] = {}
        used_hashes: set[str] = set()
        for ch in chapters:
            for b in ch.blocks:
                if isinstance(b, Figure) and b.asset_hash:
                    used_hashes.add(b.asset_hash)
                # 内容其实是图的公式块，同样需要把裁剪图打进包里
                if isinstance(b, MathBlock) and b.asset_hash and (
                    is_image_placeholder(b.latex) or b.force_image
                ):
                    used_hashes.add(b.asset_hash)
        for h in sorted(used_hashes):
            data = p.read_asset(h)
            if data:
                assets[f"{h}.png"] = data
        print(f"[i] 内嵌图片 {len(assets)} 张")

        produced: list[Path] = []

        # --- EPUB ---
        if "epub" in formats:
            book_uuid = stable_uuid(title + "|" + "|".join(authors) + "|" + str(len(chapters)))
            pages = [
                (f"text/ch{i:03d}.xhtml", chapter_to_xhtml(ch, title, lang, a.trace))
                for i, ch in enumerate(chapters, 1)
            ]
            nav = build_nav(chapters, toc_items, title, lang, a.toc_max_level)
            css = DEFAULT_CSS.read_text(encoding="utf-8") if DEFAULT_CSS.exists() else ""
            opf = build_opf(
                title, authors, lang, book_uuid, len(chapters),
                list(used_hashes),
                {"publisher": a.publisher, "description": meta.get("description")},
                has_math=stats["blocks"]["math"] + stats["inline_math_runs"] > 0,
            )
            epub_path = outdir / f"{stem}.epub"
            write_epub(epub_path, pages, nav, css, opf, assets)
            produced.append(epub_path)
            print(f"[+] EPUB  {epub_path}  ({epub_path.stat().st_size/1024:.0f} KB)")

        # --- LaTeX ---
        if "tex" in formats:
            tex = build_tex(chapters, title, authors, include_figures=not a.no_figures)
            tex_path = outdir / f"{stem}.tex"
            tex_path.write_text(tex, encoding="utf-8")
            produced.append(tex_path)
            print(f"[+] LaTeX {tex_path}  ({tex_path.stat().st_size/1024:.0f} KB)")
            if assets and not a.no_figures:
                imgdir = outdir / "images"
                imgdir.mkdir(exist_ok=True)
                for name, data in assets.items():
                    (imgdir / name).write_bytes(data)
                print(f"[+]        {imgdir}  ({len(assets)} 张图)")

        # --- Markdown ---
        if "md" in formats:
            md = build_markdown(chapters, title, authors)
            md_path = outdir / f"{stem}.md"
            md_path.write_text(md, encoding="utf-8")
            produced.append(md_path)
            print(f"[+] MD    {md_path}  ({md_path.stat().st_size/1024:.0f} KB)")

        # --- HTML 预览（KaTeX 渲染） ---
        if "html" in formats:
            from html_preview import build_preview_html  # 延迟导入

            katex_dir = Path(a.katex) if a.katex else None
            # 本地 KaTeX → 复制到输出目录旁，让预览页可以完全离线打开。
            # 增量同步而不是 rmtree + copytree：后者是「先毁再建」，重复生成同一个
            # 预览时会撞上环境里的批量删除守卫（一次 80+ 个文件就触发），而且它没有
            # 换来任何东西——预览页只在乎文件在不在，多几个陈旧文件无害。
            if katex_dir and (katex_dir / "katex.min.css").exists():
                dest = outdir / "katex"
                if katex_dir.resolve() != dest.resolve():
                    dest.mkdir(parents=True, exist_ok=True)
                    for item in katex_dir.rglob("*"):
                        if not item.is_file():
                            continue
                        target = dest / item.relative_to(katex_dir)
                        if not target.is_file() or target.stat().st_size != item.stat().st_size:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, target)

            body_parts = []
            for i, ch in enumerate(chapters, 1):
                p0 = next((getattr(b, "page", None) for b in ch.blocks if getattr(b, "page", None)), None)
                body_parts.append(f'<article class="chapter" id="ch{i}">')
                body_parts.append(
                    f'<div class="muted" style="font-family:system-ui,sans-serif;'
                    f'font-size:.72rem;letter-spacing:.04em">第 {i} 节'
                    + (f" · 原书第 {p0} 页起" if p0 else "")
                    + "</div>"
                )
                body_parts.append("\n".join(block_to_xhtml(b, a.trace) for b in ch.blocks))
                body_parts.append("</article>")

            css_text = DEFAULT_CSS.read_text(encoding="utf-8") if DEFAULT_CSS.exists() else ""
            toc_links = []
            for i, ch in enumerate(chapters, 1):
                head = next((b for b in ch.blocks if isinstance(b, Heading)), None)
                label = head.text if head else f"第 {i} 节"
                toc_links.append(f'<a href="#ch{i}">{md_bold(xesc(label))}</a>')

            html_doc = build_preview_html(
                title=title,
                authors=authors,
                body="\n".join(body_parts),
                katex_dir=katex_dir,
                lang=lang,
                reader_css=css_text,
                toc_html="\n    ".join(toc_links),
            )
            html_path = outdir / f"{stem}.html"
            html_path.write_text(html_doc, encoding="utf-8")
            produced.append(html_path)
            print(f"[+] HTML  {html_path}  ({html_path.stat().st_size/1024:.0f} KB)")

    ph = PLACEHOLDER_STATS
    if any(ph.values()):
        print(
            "[i] 内容其实是图的公式：%d 处已还原成裁剪图，%d 处连图都没有只能留空，"
            "%d 条行内公式的图标记被丢弃"
            % (ph["display_restored"], ph["display_lost"], ph["inline_dropped"])
        )

    print("[✓] 完成")
    for f in produced:
        print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 7. pipeline/texfix.py

```python
r"""把 OCR 弄坏的 LaTeX 修回 KaTeX 能编译的样子。

和 `pcexlib` 里那几条修复的分工：那边修的是**块的组装**（哪几行属于同一个
公式、caption 该归位到哪里），这里修的是**一条公式自己的文本**——环境名配错、
花括号不成对、命令拼错。两者失败的样子很像（喂给 KaTeX 都是红色报错），
但可用的证据完全不同，混在一起写迟早互相踩。

## 为什么必须修，而不是「编译不过就退回原图」

块级公式有退路：`asset_hash` 那张裁剪图还在，渲染端把公式换成原图就是了，
内容一字不差。**行内公式没有退路**——它在 `.pcex` 里只是一段文本，OCR 阶段
就没留图。KaTeX 认不出来，正文里留下的就是一段红色报错。所以行内失败的
优先级比块级高，而这里修的正是所有公式共用的那一层。

## 每条规则的形状

都是**结构**修复，不是猜测内容：环境名的配对由 `\begin`/`\end` 自己的栈给出，
花括号由计数给出，`\left`/`\right` 由深度给出。这种修复的判据来自公式自身，
不来自对书里写了什么的猜测，所以它可以批量跑在全库上而不用担心改坏——
一套「本来就正确」的公式过一遍规则必须**一字不变**，这是
`work/check_texfix.py` 要守住的门禁。

反过来，那些需要猜的（`\llangledenth` 到底是哪个符号）一律不做：猜错会静默
改坏公式，比留一段红字更糟。那种留给按 `fid` 写的修正文件。

## 顺序

先删 `$`——它会让后面每一步的扫描都把断句看错；再换命令别名；然后配环境名、
`\left/\right`；最后才是花括号计数。花括号放最后是因为它是**唯一**一条会
增删 `{}` 的规则，前面那些规则产生的 `\text{...}`、`\operatorname{...}` 必须
先落地，否则计数会按半成品算。
"""
import re
from typing import Optional

# ---------------------------------------------------------------------------
# 1. KaTeX 不认的命令 → 它认的等价写法
# ---------------------------------------------------------------------------

# 只收**语义确定**的。判据是「看到这个拼写，作者一定是想写右边那个」：
#   \sinc \sgn \card \im \ftr  都是算子名，OCR 把 \operatorname{..} 拆成了
#                              「反斜杠 + 名字」，补回 \operatorname 即可；
#   \bnabla                    是 \nabla 的多打了一个 b；
#   \minus \nerrow \llq        是箭头/关系符拼写走形，逐字可复原。
#
# 有意不收的（拼写走形到无法确定原意，交给按 fid 的修正文件）：
#   \llangledown \llangledenth \llangledimeq  可能是 \langle / \lfloor / \le
#   \mathdovev \mathmin \vdivmatrix \BOXb     不知道对应什么符号
#   \arc{eff}                                可能是 \arccos 被切碎
CMD_ALIASES = {
    r"\sinc": r"\operatorname{sinc}",
    r"\sgn": r"\operatorname{sgn}",
    r"\card": r"\operatorname{card}",
    r"\im": r"\operatorname{Im}",
    r"\ftr": r"\operatorname{tr}",
    r"\bnabla": r"\nabla",
    r"\minus": r"-",
    r"\nerrow": r"\nearrow",
    r"\llq": r"\ll",
    r"\llfloor": r"\lfloor",
    r"\gRightarrow": r"\Rightarrow",
    r"\ast": r"\ast",
    r"\*": r"\ast",
    r"\half": r"\tfrac{1}{2}",
    r"\½": r"\tfrac{1}{2}",
    r"\textsc": r"\text",
    r"\mbox": r"\text",
    r"\hbox": r"\text",
    r"\textbackslash": r"\backslash",
    # 带圈数字：OCR 把 \textcircled{1} 压成了 \circled1，`\textcircled` 本身
    # KaTeX 是认的（线性代数五讲里那个 `\textcircled{6}` 就编译得过）。
    r"\circled": r"\textcircled",
    r"\downarrowarrow": r"\downarrow",
    r"\uprho": r"\rho",
    # 同余式：`\tmod{m}` / `\tmod{8}` 是 `\pmod` 被念歪，参数形式正好对得上。
    r"\tmod": r"\pmod",
    # `\napprox` 是把 `\not\approx`（不约等于）的两截粘成了一条命令。
    r"\napprox": r"\not\approx",
    # 中文标点被当成命令转义了（`a\、 b\、 c` 是顿号分隔的枚举）。它们不是
    # 数学符号，换回半角标点正是书上那个式子。
    r"\、": ",",
    r"\，": ",",
    r"\。": ".",
    r"\；": ";",
    r"\：": ":",
}

# KaTeX 里只在 display 可用的环境名 → 两种模式都能用的等价物。改完在 display
# 下渲染不变（`aligned`/`gathered` 本来就是它俩的「内部版」），在行内则从
# 报错变成可用——行内公式里出现 align 是 OCR 把独立公式塞进了句子中间。
ENV_RENAME = {
    "align": "aligned",
    "align*": "aligned",
    "alignat": "aligned",
    "alignat*": "aligned",
    "eqnarray": "aligned",
    "eqnarray*": "aligned",
    "gather": "gathered",
    "gather*": "gathered",
}

# 命令扫描：`\` 后跟一串字母，或跟任意单个字符（`\½`、`\*` 这类）。
_TOKEN_RE = re.compile(r"\\([A-Za-z]+|.)", re.S)


_TEXT_CMD_RE = re.compile(
    r"\\(?:text|textrm|textnormal|textsf|texttt|textbf|textit|textmd|textup"
    r"|textsl|emph|mbox|hbox|fbox)\s*\{"
)


def _text_mode_spans(tex: str) -> list[bool]:
    r"""标出「文本模式」区间内的每个字符位置。

    `\text{...}` 里 `$...$` 是**合法且必要**的：它切回数学模式，是用来在文本
    里嵌公式的。`\text{$\mathbb{Z}$}` 完全能编译，删掉那两个 `$` 反而把
    `\mathbb` 留在了文本模式里 —— 这是全库唯一一条被规则改坏的公式，教训是
    删 `$` 之前必须先知道自己在不在 `\text` 里。
    """
    keep = [False] * len(tex)
    for m in _TEXT_CMD_RE.finditer(tex):
        depth = 1
        i = m.end()
        while i < len(tex) and depth:
            c = tex[i]
            if c == "\\":
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        for j in range(m.end(), min(i, len(tex))):
            keep[j] = True
    return keep


def _drop_dollars(tex: str) -> str:
    r"""删掉公式体里的 `$`——但 `\text{...}` 里的留下。

    整个字符串已经在数学模式里了，`$` 只会让 KaTeX 报 "Can't use function
    '$' in math mode"。它们是 OCR 把「行内公式的边界」也一起吐了出来：
    `f:\mathbb{R}^{d}\rightarrow [$0$,$+\infty$]` 里那两个 `$` 毫无用处，
    删掉正好还原成 `[0,+\infty]`。

    >>> _drop_dollars(r"[$0$,$+\infty$]")
    '[0,+\\infty]'
    >>> _drop_dollars(r"\text{$\mathbb{Z}$}")   # 文本模式里的 $ 是切换符
    '\\text{$\\mathbb{Z}$}'
    >>> _drop_dollars(r"a\$b")                  # 转义的 \$ 是内容，不动
    'a\\$b'
    """
    if "$" not in tex:
        return tex
    keep = _text_mode_spans(tex)
    out = []
    for i, c in enumerate(tex):
        if c == "$" and not keep[i] and not (i and tex[i - 1] == "\\"):
            continue
        out.append(c)
    return "".join(out)


def _sub_aliases(tex: str) -> str:
    r"""按 `CMD_ALIASES` 换命令名。

    >>> _sub_aliases(r"a \sinc(\pi p a)")
    'a \\operatorname{sinc}(\\pi p a)'
    >>> _sub_aliases(r"x \nrightarrow y")   # 不在表里，原样
    'x \\nrightarrow y'
    """
    if not any(k in tex for k in CMD_ALIASES):
        return tex
    return _TOKEN_RE.sub(lambda m: CMD_ALIASES.get(m.group(0), m.group(0)), tex)


def _rename_envs(tex: str) -> str:
    r"""把只在 display 可用的环境换成两种模式都能用的。

    >>> _rename_envs(r"\begin{align*}a &= b\end{align*}")
    '\\begin{aligned}a &= b\\end{aligned}'
    """
    if "\\begin" not in tex and "\\end" not in tex:
        return tex

    def swap(m: re.Match) -> str:
        name = m.group(2)
        # group(1) 含反斜杠本身——`\b` 在 replacement 里会被 Python 再解释一遍，
        # 拼接时必须整段带上，否则 `\begin` 会变成退格符。
        return "%s{%s}" % (m.group(1), ENV_RENAME.get(name, name))

    return re.sub(r"(\\(?:begin|end))\s*\{([^}]*)\}", swap, tex)


_BEGIN_RE = re.compile(r"\\begin\s*\{([^}]*)\}")
_END_RE = re.compile(r"\\end\s*\{([^}]*)\}")


def _pair_envs(tex: str) -> str:
    r"""让每个 `\end{...}` 与它配对的 `\begin{...}` 同名。

    用栈，不猜内容：OCR 常把环境名念歪（`\begin{vmatrix}` 配
    `\end{vdivmatrix}`），但**开合的顺序它没弄错**，所以栈顶就是答案。
    栈空时遇到的 `\end` 是凭空多出来的，删掉——留着一定是 KaTeX 报错。

    >>> _pair_envs(r"\begin{vmatrix}a\end{vdivmatrix}")
    '\\begin{vmatrix}a\\end{vmatrix}'
    >>> _pair_envs(r"\begin{gathered}a\begin{array}{c}b\end{array}\end{gathered}")
    '\\begin{gathered}a\\begin{array}{c}b\\end{array}\\end{gathered}'
    >>> _pair_envs(r"a\end{array}b")        # 多余的 \end
    'ab'
    """
    if "\\begin" not in tex and "\\end" not in tex:
        return tex
    out: list[str] = []
    stack: list[str] = []
    i, n = 0, len(tex)
    while i < n:
        m = _BEGIN_RE.match(tex, i)
        if m:
            stack.append(m.group(1).strip())
            out.append(m.group(0))
            i = m.end()
            continue
        m = _END_RE.match(tex, i)
        if m:
            if stack:
                want = stack.pop()
                if m.group(1).strip() == want:
                    out.append(m.group(0))
                else:
                    out.append("\\end{%s}" % want)
            # 栈空 → 这条 \end 没有对应的 \begin，丢掉
            i = m.end()
            continue
        out.append(tex[i])
        i += 1
    return "".join(out)


def _pair_delims(tex: str) -> str:
    r"""让 `\left` 与 `\right` 成对。

    只有深度信息可用：多出来的 `\right` 删掉（`a\right\right|` 里第一个是
    重复的），缺 `\right` 的在**末尾**补 `\right.`——`\right.` 是隐形定界符，
    补在末尾不会把后面的内容关进括号里，是唯一不会改变公式结构的补法。

    >>> _pair_delims(r"\left(a\right\right|")
    '\\left(a\\right|'
    >>> _pair_delims(r"\left(a + b")
    '\\left(a + b\\right.'
    >>> _pair_delims(r"a\right) b")
    'a) b'
    """
    if "\\left" not in tex and "\\right" not in tex:
        return tex
    out: list[str] = []
    depth = 0
    i, n = 0, len(tex)
    while i < n:
        if tex.startswith("\\left", i) and not (i + 5 < n and tex[i + 5].isalpha()):
            depth += 1
            out.append("\\left")
            i += 5
            continue
        if tex.startswith("\\right", i) and not (i + 6 < n and tex[i + 6].isalpha()):
            if depth == 0:
                i += 6          # 多余的 \right
                continue
            depth -= 1
            out.append("\\right")
            i += 6
            continue
        out.append(tex[i])
        i += 1
    if depth > 0:
        out.append("\\right." * depth)
    return "".join(out)


_TAG_RE = re.compile(r"\\tag\s*\{[^}]*\}")


def _dedupe_tag(tex: str, inline: bool) -> str:
    r"""`\tag` 只留一个；行内公式则一个都不留。

    OCR 会把同一编号吐两遍（`\tag{11-43}` 两次），KaTeX 直接报
    "Multiple \tag"。而行内公式里的 `\tag` 在 KaTeX 里是禁止的，无论几个。

    >>> _dedupe_tag(r"a \tag{1}b\tag{2}", False)
    'a \\tag{1}b'
    >>> _dedupe_tag(r"a\tag{1}", True)
    'a'
    """
    hits = _TAG_RE.findall(tex)
    if not hits:
        return tex
    if inline:
        return _TAG_RE.sub("", tex)
    if len(hits) == 1:
        return tex
    first = hits[0]
    seen = False

    def keep_first(m: re.Match) -> str:
        nonlocal seen
        if seen:
            return ""
        seen = True
        return first

    return _TAG_RE.sub(keep_first, tex)


_HLINE_RE = re.compile(r"\\hline\b")


def _drop_stray_hline(tex: str) -> str:
    r"""删掉不在 array 里的 `\hline`。

    `\hline` 只在 array/matrix 系环境里有意义，KaTeX 在别处直接报错。判定不
    靠猜：把环境栈走一遍，出现 `\hline` 时栈顶不是阵列类就删。

    在阵列里的 `\hline` 则要**先补行分隔**再留着：它的语义是「横线独占一行」，
    KaTeX 要求它出现在行首；OCR 常把它前面的 `\\` 吃掉（折成单个 `\ `），于是
    横线落到了行中间，KaTeX 照样报 "valid only within array environment"——报的
    是位置而不是环境，照字面去查环境会白费功夫。

    >>> _drop_stray_hline(r"a\\\hline b")
    'a\\\\ b'
    >>> _drop_stray_hline(r"\begin{array}{c}a\\\hline b\end{array}")
    '\\begin{array}{c}a\\\\\\hline b\\end{array}'
    >>> _drop_stray_hline(r"\begin{array}{c|c}B & n \ \hline D & E\end{array}")
    '\\begin{array}{c|c}B & n \\ \\\\\\hline D & E\\end{array}'
    """
    if "\\hline" not in tex:
        return tex
    stack: list[str] = []
    out: list[str] = []
    i, n = 0, len(tex)
    while i < n:
        m = _BEGIN_RE.match(tex, i)
        if m:
            stack.append(m.group(1).strip())
            out.append(m.group(0))
            i = m.end()
            continue
        m = _END_RE.match(tex, i)
        if m:
            if stack:
                stack.pop()
            out.append(m.group(0))
            i = m.end()
            continue
        m = _HLINE_RE.match(tex, i)
        if m:
            top = stack[-1].lower() if stack else ""
            if not top.startswith(("array", "matrix", "tabular", "cases")):
                i = m.end()      # 不在阵列里，删
                continue
            if not "".join(out).rstrip().endswith("\\\\"):
                out.append("\\\\")     # 横线不在行首，替它把行断开
            out.append(m.group(0))
            i = m.end()
            continue
            out.append(m.group(0))
            i = m.end()
            continue
        out.append(tex[i])
        i += 1
    return "".join(out)


_FBOX_RE = re.compile(r"\\fbox\s*\{")


def _fix_fbox(tex: str) -> str:
    r"""`\fbox{...}` 里的数学命令包进 `$...$`。

    `\fbox` 是文本模式命令（画个框住文字的框），而里面的 `\binom` 是数学命令。
    KaTeX 的报错是 "Can't use function '\binom' in text mode"——指的是**模式**，
    而修法必须在里面就地切回数学模式，不能把 `\fbox` 换掉：书上那一条确实是
    「一个框住 $\binom{8}{3}$ 的框」，框是内容的一部分。

    >>> _fix_fbox(r"\fbox{\binom{8}{3}} = 56")
    '\\fbox{$\\binom{8}{3}$} = 56'
    >>> _fix_fbox(r"\fbox{abc}")          # 纯文字，不动
    '\\fbox{abc}'
    >>> _fix_fbox(r"\fbox{$x$}")          # 已经在数学模式里，不重复包
    '\\fbox{$x$}'
    """
    if "\\fbox" not in tex:
        return tex
    out: list[str] = []
    i, n = 0, len(tex)
    while i < n:
        m = _FBOX_RE.match(tex, i)
        if not m:
            out.append(tex[i])
            i += 1
            continue
        depth = 1
        j = m.end()
        while j < n and depth:
            if tex[j] == "\\":
                j += 2
                continue
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
            j += 1
        body = tex[m.end():j - 1] if depth == 0 else tex[m.end():]
        if "\\" in body and "$" not in body:
            body = "$" + body + "$"
        out.append(m.group(0) + body + (tex[j - 1] if depth == 0 else ""))
        i = j
    return "".join(out)


# KaTeX 里 `\dots` 有一组「按上下文选点号」的兄弟命令，后缀是这几个字母时
# 它本身就是一条合法命令，不能拆。
_DOTS_SIBLINGS = set("bcimo")
_DOTS_RE = re.compile(r"\\dots([A-Za-z])")


def _split_dots(tex: str) -> str:
    r"""`\dotsn` 拆成 `\dots n`——OCR 把点号和变量名粘在了一起。

    带个字母后缀看起来像命令，其实不是：书上是 `\dots, n` 或 `\dots n`，
    点号和 `n` 之间原本有空格。判据是**后缀必须不是 `\dots` 的合法兄弟**
    （`\dotsb \dotsc \dotsi \dotsm \dotso` 都是 KaTeX 的真命令，拆了就错）。

    >>> _split_dots(r"\dotsn,n\}")
    '\\dots n,n\\}'
    >>> _split_dots(r"\dotsj_{r}s")
    '\\dots j_{r}s'
    >>> _split_dots(r"\dotsb x")     # 是合法命令，不动
    '\\dotsb x'
    """
    if "\\dots" not in tex:
        return tex
    return _DOTS_RE.sub(
        lambda m: m.group(0) if m.group(1) in _DOTS_SIBLINGS else "\\dots " + m.group(1),
        tex,
    )


def _unwrap_right(tex: str) -> str:
    r"""`{\right)` 外层那对花括号是多余的。

    OCR 常把 `\right)` 写进一个分组里：`\left(x{\right)}=`。定界符一旦被包进
    `{}`，就不再和它对家配对（KaTeX 报 "Expected '\right', got '}'"），而那
    个分组本身没有任何作用。剥掉 `{` 之后，多出来的 `}` 由后面的括号配平
    收拾——两步合起来正好还原成 `\left(x\right)=`。

    **只剥 `\right`，不剥 `\left`**：`{\left|...\right|}` 里的 `{` 常常正是
    `\frac` / `\underset` 的参数分组（`\frac{\left|a\right|}{\left|b\right|}`），
    剥掉会得到 `\frac\left|a\right|`，KaTeX 报 "Expected group as argument"。
    试过按「`{` 前面是不是命令名」区分，但 `\frac{...}{\left...` 的第二个
    参数前面是个 `}`，一样躲不过——所以这条路整体放弃。真正写成
    `{\left|_{t_0}}\right)` 的少数几处留给修正文件。

    >>> _unwrap_right(r"\left(x{\right)}=")   # `{` 剥掉，多出的 `}` 交给配平
    '\\left(x\\right)}='
    >>> _unwrap_right(r"\frac{\left|a\right|}{\left|b\right|}")
    '\\frac{\\left|a\\right|}{\\left|b\\right|}'
    """
    if "{\\right" not in tex:
        return tex
    # 必须确认命令在此结束：`{\rightarrow}` 里也含 `{\right`，照字符串替换会把它
    # 切成 `\rightarrow}`——那是一条完全不同的命令被改坏，而且改完仍然编译得过，
    # 门禁的「改坏」计数看不见它。（这是实测抓到的，唯一一条被规则改哑的公式。）
    return re.sub(r"\{\\right(?![A-Za-z])", r"\\right", tex)


_REENV_RE = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")
_ROW_ENVS = ("array", "matrix", "pmatrix", "bmatrix", "vmatrix", "cases",
             "aligned", "gathered", "tabular", "alignedat")


def _match_env_end(tex: str, pos: int) -> Optional[int]:
    r"""从 `\begin{...}` 的 `}` 之后开始，找配对的 `\end{...}` 的结束位置。

    配不平返回 None（那种公式本来就还要靠别的规则救，这里不猜）。
    """
    depth = 1
    for m in _REENV_RE.finditer(tex, pos):
        if m.group(1) == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
    return None


def _split_direct(body: str) -> list[tuple[bool, str]]:
    r"""把环境体切成 `(是否本层直辖文本, 片段)`，子环境整段标成 False。

    判「本层的行分隔丢没丢」必须在挖掉子环境之后做：子环境里的 `\\` 是它自己
    的行分隔，留着会把证据冲淡——本层一条都没丢，却因为子环境有 `\\` 被判成
    「没丢」，正好漏掉要修的那一处。反过来，替换也只能动直辖文本，否则会把
    子环境里**故意**的 `\ `（插空格）一起改掉。

    >>> _split_direct(r"a \ b")
    [(True, 'a \\ b')]
    >>> [(d, s) for d, s in _split_direct(r"a\begin{array}{c}x \\ y\end{array}b")]
    [(True, 'a'), (False, '\\begin{array}{c}x \\\\ y\\end{array}'), (True, 'b')]
    """
    segs: list[tuple[bool, str]] = []
    i = 0
    for m in _REENV_RE.finditer(body):
        if m.group(1) != "begin" or m.start() < i:
            continue
        end = _match_env_end(body, m.end())
        if end is None:
            continue
        segs.append((True, body[i:m.start()]))
        segs.append((False, body[m.start():end]))
        i = end
    segs.append((True, body[i:]))
    return segs


def _restore_row_breaks(tex: str) -> str:
    r"""把阵列里被吃掉的行分隔 `\\` 补回来。

    OCR 会把 `\\` 折成单个 `\ `（一个反斜杠加一个空格）——在 LaTeX 里 `\ `
    恰好也是个合法命令（插一个空格），所以这种损坏**不会报"未知命令"**。
    症状是 KaTeX 说 "Expected & or \\ or \cr or \end"，而且**指向公式末尾**，
    看上去像少了 `\end`；真实位置在被折掉的那个地方。

    只在本层**一条 `\\` 都没有**时才动手：这是「分隔全丢了」的确证，此时每个
    `\ ` 都只能是它。若本层还有真的 `\\`，说明只丢了其中一处，而丢的是哪一处
    没有证据，猜错会把两行挤成一行——比留一段红字更坏。

    >>> _restore_row_breaks(r"\begin{array}{c}a \ b\end{array}")
    '\\begin{array}{c}a \\\\ b\\end{array}'
    >>> _restore_row_breaks(r"\begin{array}{c}a \\ b \ c\end{array}")   # 本层有真分隔
    '\\begin{array}{c}a \\\\ b \\ c\\end{array}'
    >>> _restore_row_breaks(r"\begin{array}{c}a \\ b\end{array}")
    '\\begin{array}{c}a \\\\ b\\end{array}'
    >>> _restore_row_breaks(r"\begin{array}{c}a\begin{array}{c}x \ y\end{array}b\end{array}")
    '\\begin{array}{c}a\\begin{array}{c}x \\\\ y\\end{array}b\\end{array}'
    """
    if "\\begin" not in tex:
        return tex
    starts = [m.start() for m in _REENV_RE.finditer(tex) if m.group(1) == "begin"]
    # 从后往前：改右边不会让左边的下标失效；而本层自己的结束位置每次重算，
    # 所以子环境被改长也不影响。
    for at in reversed(starts):
        m = _REENV_RE.match(tex, at)
        if not m.group(2).strip().lower().startswith(_ROW_ENVS):
            continue
        end = _match_env_end(tex, m.end())
        if end is None:
            continue
        segs = _split_direct(tex[m.end():end])
        direct = [s for is_own, s in segs if is_own]
        if any("\\\\" in s for s in direct):
            continue
        if not any("\\ " in s for s in direct):
            continue
        fixed = "".join(_break_or_space(s) if is_own else s for is_own, s in segs)
        tex = tex[:m.end()] + fixed + tex[end:]
    return tex


def _break_or_space(seg: str) -> str:
    r"""把本层文本里的 `\ ` 换成行分隔——但贴着 `&` 的那个不换。

    反例（全库抽样时抓到的一条）：`... x_3 & \ = b_1c_1 + b_2c_2 + b_3c_3.`
    这里 `&` 后面那个 `\ ` 是 OCR 抹不掉的空格噪音，作者的对齐点是 `&`，右边
    紧跟 `=`。换成 `\\` 就变成「在 `&` 处断行」，把一个两格的等式拆成了两行
    ——渲染出来还是能编译，只是不再是书上那个公式。对齐符后面不该有行分隔，
    所以这一处按原样留着。

    换出来的也是 `\\` **加一个空格**：`\\b` 会被读成 `\` + `\b`，多留那个空格
    不花任何代价。

    >>> _break_or_space(r"a \ b")
    'a \\\\ b'
    >>> _break_or_space(r"x_3 & \ = b")
    'x_3 & \\ = b'
    """
    out: list[str] = []
    i = 0
    while i < len(seg):
        if seg.startswith("\\ ", i):
            if "".join(out).rstrip().endswith("&"):
                out.append(seg[i:i + 2])
            else:
                out.append("\\\\ ")
            i += 2
            continue
        out.append(seg[i])
        i += 1
    return "".join(out)


def _balance_braces(tex: str) -> str:
    r"""让 `{}` 恰好配平：多余的右括号删掉，缺的在末尾补齐。

    计数跳过 `\{` `\}`（转义花括号是**内容**，不是分组）和 `\\`（行分隔），
    所以扫描器遇到反斜杠就整对吃掉。

    补的位置**在末尾的 `\end{...}` 之前**，不是最后。这是踩过的：
    `\begin{gathered}...\sqrt{9^{2}\end{gathered}` 缺的是 `\sqrt{` 的收尾，
    补在 `\end` 之后就成了 `\end{gathered}}`，KaTeX 报的是"Expected '}',
    got '\end'"——`\end` 落进了 `\sqrt` 的参数里。`\end` 必须留在环境最外层。

    >>> _balance_braces(r"1 \le q \le k}")
    '1 \\le q \\le k'
    >>> _balance_braces(r"\mathbf{v")
    '\\mathbf{v}'
    >>> _balance_braces(r"\sqrt{9^{2}\end{gathered}")
    '\\sqrt{9^{2}}\\end{gathered}'
    >>> _balance_braces(r"\{a\}")
    '\\{a\\}'
    >>> _balance_braces(r"\frac{a}{b}")
    '\\frac{a}{b}'
    """
    out: list[str] = []
    depth = 0
    i, n = 0, len(tex)
    while i < n:
        c = tex[i]
        if c == "\\" and i + 1 < n:
            out.append(tex[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                i += 1          # 没有与之配对的 {，删
                continue
            depth -= 1
        out.append(c)
        i += 1
    if depth <= 0:
        return "".join(out)
    text = "".join(out)
    tail = _TAIL_ENDS_RE.search(text)
    if tail:
        return text[:tail.start()] + "}" * depth + text[tail.start():]
    return text + "}" * depth


_TAIL_ENDS_RE = re.compile(r"(?:\\end\s*\{[^}]*\})+\s*$")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def fix_math(tex: str, inline: bool = False) -> tuple[str, list[str]]:
    r"""跑一遍全部结构规则，返回 (修复后, 实际改动的规则名)。

    `inline` 只影响 `\tag`：行内公式里 KaTeX 不接受编号，删掉。

    >>> fix_math(r"\begin{vmatrix}a\end{vdivmatrix}")[0]
    '\\begin{vmatrix}a\\end{vmatrix}'
    >>> fix_math(r"a \sinc(\pi p a)")[0]
    'a \\operatorname{sinc}(\\pi p a)'
    >>> fix_math(r"\frac{a}{b}")[0]          # 本来就好的公式必须一字不变
    '\\frac{a}{b}'
    >>> fix_math(r"\frac{a}{b}")[1]
    []
    """
    applied: list[str] = []

    def step(name: str, fn) -> None:
        nonlocal tex
        new = fn(tex)
        if new != tex:
            applied.append(name)
            tex = new

    step("drop_dollars", _drop_dollars)
    step("fix_fbox", _fix_fbox)
    step("unwrap_right", _unwrap_right)
    step("split_dots", _split_dots)
    step("alias", _sub_aliases)
    step("rename_env", _rename_envs)
    step("pair_env", _pair_envs)
    step("row_breaks", _restore_row_breaks)
    step("pair_delim", _pair_delims)
    step("dedupe_tag", lambda t: _dedupe_tag(t, inline))
    step("drop_hline", _drop_stray_hline)
    step("balance_braces", _balance_braces)
    return tex, applied


# ---------------------------------------------------------------------------
# 不是公式的那些
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CMD_RE = re.compile(r"\\[A-Za-z]+")
_VISIBLE_RE = re.compile(r"\\[A-Za-z]+|\\.|[{}^_&$~\\\s]")
_LONG_DIGITS_RE = re.compile(r"\d{6,}")


def is_not_math(tex: str) -> bool:
    r"""这条「公式」根本不是公式——OCR 把散文、表格或页码吞进了数学模式。

    **只在 KaTeX 已经编译失败之后问这个问题。** 这是整套判定里唯一的安全阀：
    编译得过就说明它至少是一条合法公式，无论看起来多怪都不该被退回文本
    （可计算性理论里那些三行的 `\begin{array}` 定义式很长、很像散文，它们是
    真公式）。只有在「KaTeX 说它不成立」**且**「看着不像公式」两个条件同时
    成立时，退回文本才比留着更有道理。

    判据都指向同一件事：数学模式里不该出现这些东西。
      * `$`——它自己就是数学模式的边界，出现在体内说明 OCR 多吐了一层；
      * 中文且占比过半——数学符号里混进的汉字；
      * 六位以上连续数字——表格里的行号或页码列；
      * 长而几乎没有命令——一长串纯文字被 `\mathrm{}` 包了进来。

    >>> is_not_math(r"f:\mathbb{R}^{d}\rightarrow [$0$,$+\infty$]")
    True
    >>> is_not_math(r"\mathrm{We~exercise~7.5.4.}}")
    False
    >>> is_not_math(r"\frac{a}{b}")          # 合法公式，无论如何都不算
    False
    """
    if "$" in tex:
        return True
    if _LONG_DIGITS_RE.search(tex):
        return True
    stripped = _VISIBLE_RE.sub("", tex)
    # 剥掉命令和结构符之后什么都不剩（`}`、`\Phi ^`、`\, \;`）：它连一个可读
    # 的字符都不含，不可能是公式。
    if not stripped.strip():
        return True
    cjk = len(_CJK_RE.findall(tex))
    if cjk >= 3 and cjk * 2 >= max(len(stripped), 1):
        return True
    if len(tex) > 60 and len(_CMD_RE.findall(tex)) < 3:
        return True
    if len(tex) > 24:
        structural = sum(tex.count(c) for c in "{}^_~")
        if structural / len(tex) > 0.45:
            return True
    # 长片段的字母数字占比：`\lim_\{\_{}} }{(x_r\:\wx` 这种碎片里，能当内容读的
    # 字符不到三成，其余全是错位的结构符。放这条要有长度门槛——短的碎片
    # （`A_\vec{x}`、`\Phi ^`）更像真公式被截断，退成文本反而是丢东西。
    if len(tex) > 20:
        alnum = sum(c.isalnum() for c in tex)
        if alnum / len(tex) < 0.45:
            return True
    return False


_WRAPPER_RE = re.compile(
    r"\\(?:text|textrm|textnormal|mathrm|mbox|hbox|textsf|texttt|operatorname)"
    r"\s*\{([^{}]*)\}"
)


def salvage_text(tex: str) -> str:
    r"""从一条「不是公式」的条目里尽量捞回能读的字。

    退回文本而不是删掉，是因为这些条目里有一部分**本来是完整的一句话**
    （`\mathrm{We~exercise~7.5.4.}}` 是「We exercise 7.5.4.」），丢掉就等于
    正文少一句。捞不回来的（纯符号乱码）返回空串，由调用方丢弃——那种东西
    摆在正文里只是噪音，留着红字和留着乱码一样没用。

    只从**文本包装命令**（`\text{}`/`\mathrm{}`）里捞，而且那一层里不能再有
    命令——`(\mathrm{why?})` 是短语，`\operatorname{if}((e)_{k})` 是公式，把
    后者也当成文字捞出来，得到的是 "if" 这种半截词。没有包装命令时整条都是
    碎片，一律返回空。

    还要「读得出词」：至少两个三字母以上的单词，或一个四字以上的中文片段。
    这是为了不让 `_{N}^{-}{a}_{N}` 这种碎片被当成文字塞回正文。

    >>> salvage_text(r"\mathrm{We~exercise~7.5.4.}}")
    'We exercise 7.5.4.'
    >>> salvage_text(r"_{N}^{-}{a}_{N}")     # 捞不出词，返回空
    ''
    >>> salvage_text(r"(\mathrm{why?})\right)")   # 包装层是短语但不是句子
    ''
    """
    # 没有包装命令就**直接放弃**，绝不回落到整条串去捞：整条串里那些
    # `\begin{array}` 的命令名会被当成单词（`array` 恰好是四个字母），捞出来
    # 是一串 "array c c g - F - F" 这样的东西，比留空还糟。
    chunks = [c for c in _WRAPPER_RE.findall(tex) if "\\" not in c]
    if not chunks:
        return ""
    text = " ".join(chunks)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}$\\^_&~]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) >= 2 or (words and len(text) >= 8) or len(_CJK_RE.findall(text)) >= 4:
        return text
    return ""


_MEANINGFUL_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def has_content(tex: str) -> bool:
    r"""修完的公式是否还有实质内容——**命令也算**。

    只看「剥掉命令之后还剩什么」是不够的：`)\rho` 剥完只剩一个 `)`，会被判成
    空，于是一个本来正确的修复被拒绝采纳。而 `\rho` 就是这条公式的全部内容，
    当然算数。（实测就是这里漏掉了 `)\uprho` 的修复：别名换得对，结果被这道
    门槛挡在门外，公式原样留到成品里。）

    >>> has_content(r"\frac{a}{b}"), has_content(r")\rho")
    (True, True)
    >>> has_content("{}"), has_content(r"\, \;")
    (False, False)
    """
    return bool(_MEANINGFUL_RE.search(_VISIBLE_RE.sub("", tex))
                or _CMD_RE.search(tex))


_LATEX_NOISE_RE = _VISIBLE_RE


if __name__ == "__main__":
    import doctest
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    fails, tests = doctest.testmod(verbose=False)
    print(f"doctest: {tests - fails}/{tests} 通过")
    raise SystemExit(1 if fails else 0)
```

### 8. pipeline/check_epub.py

```python
"""check_epub.py — 对一本产出的 EPUB 做体检。

关心三件事，按「会不会让读者当场看见问题」排序：

1. **畸形 XHTML**：严格阅读器会因此打不开整本书，浏览器却毫无反应——最致命也最隐蔽。
2. **残留的 OCR 标记**：`img_url` 这类占位符一旦漏进正文，读者看到的就是这串字面量。
3. **公式定界符**：`\\(` / `\\[` 的条数，是「KaTeX 到底有没有活干」的直接证据。

用法：python check_epub.py <book.epub> [...]
"""

from __future__ import annotations

import re
import sys
import zipfile
from xml.etree import ElementTree as ET

NOISE = [
    (re.compile(r"img_url"), "OCR 图片占位符漏进正文"),
    (re.compile(r"<!--\s*image\s*-->", re.I), "图片注释"),
]

BEGIN_RX = re.compile(r"\\begin\s*\{\s*([^}]*)\s*\}")
END_RX = re.compile(r"\\end\s*\{\s*([^}]*)\s*\}")


def env_balance(raw: str) -> tuple[list[str], list[str]]:
    """返回「未闭合的 \\begin」与「多余的 \\end」。

    不能只数 `\\begin{` 出现次数——那会把每一对合法的 `\\begin{array}...\\end{array}`
    都报成问题（第一版就犯了这个错，一本书报了 79 个假警报）。必须真配一次对。
    """
    stack: list[str] = []
    orphan_end: list[str] = []
    for m in re.finditer(r"\\(begin|end)\s*\{\s*([^}]*)\s*\}", raw):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif stack and stack[-1] == m.group(2):
            stack.pop()
        else:
            orphan_end.append(m.group(2))
    return stack, orphan_end


def check(path: str) -> int:
    print(f"=== {path}")
    problems = 0
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".xhtml")]
        images = [n for n in z.namelist() if n.startswith("OEBPS/images/")]
        inline = display = 0
        malformed = 0
        for n in names:
            raw = z.read(n).decode("utf-8", "replace")
            try:
                ET.fromstring(raw)
            except Exception as e:  # noqa: BLE001
                malformed += 1
                problems += 1
                print(f"    !! 畸形 XML {n}: {str(e)[:120]}")
            inline += raw.count("\\(")
            display += raw.count("\\[")
            for rx, label in NOISE:
                hits = len(rx.findall(raw))
                if hits:
                    problems += hits
                    print(f"    !! {label} × {hits}  ({n})")
            unclosed, orphan = env_balance(raw)
            if unclosed or orphan:
                problems += len(unclosed) + len(orphan)
                print(f"    !! 环境不配对 {n}: 未闭合 {unclosed[:4]} 多余 {orphan[:4]}")
        print(f"    xhtml={len(names)} 畸形={malformed} images={len(images)}")
        print(f"    行内公式 \\(...)  = {inline}")
        print(f"    块级公式 \\[...\\] = {display}")
    return problems


if __name__ == "__main__":
    total = 0
    for p in sys.argv[1:]:
        total += check(p)
    print(f"\n合计问题：{total}")
```

### 9. pipeline/check_math.py

```python
"""Gate the delivered EPUB: compile every formula in its XHTML with KaTeX.

Why this exists as a separate gate: verify.py judges the `.pcex`, which is the
*intermediate*. Everything that can still go wrong lives downstream of it --
a correction overlay that missed, a split formula stitched the wrong way, an
entity-escaped delimiter that no longer looks like math. The only artifact the
reader ever sees is the EPUB, so that is what must be gated.

Compiles with the same KaTeX build the reader injects, so "passes here" means
"renders there".

Usage:
    python check_math.py <book.epub> [more.epub ...]
"""
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# workbuddy 已卸载（binaries 没了）→ 系统 node；v24 实测跑 katex.min.js 正常
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
KATEX_JS = Path(r"E:\EPUB\pipeline\assets\katex\katex.min.js")

# The reader scans text nodes for these delimiters, so we must see exactly the
# same thing it sees: entity-decoded text, delimiters included.
SPAN = re.compile(
    r'<span class="(math-inline|math-body)"[^>]*>(.*?)</span>', re.DOTALL
)


def extract(epub: Path) -> list[tuple[str, str, str]]:
    """Return (chapter, class, latex_without_delimiters) for every formula."""
    out = []
    with zipfile.ZipFile(epub) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith((".xhtml", ".html")):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            for cls, inner in SPAN.findall(text):
                body = html.unescape(inner)
                m = re.match(r"^\\\((.*)\\\)$", body, re.DOTALL)
                if not m:
                    m = re.match(r"^\\\[(.*)\\\]$", body, re.DOTALL)
                if not m:
                    continue
                out.append((name, cls, m.group(1).strip()))
    return out


def katex_batch(items: list[str], display=None) -> list[str]:
    """Compile many formulas in one node process; returns one line per formula.

    `display` says which of them are display formulas -- either one bool for
    all of them, or one per item. **It is not a detail.** KaTeX refuses some
    constructs outside display mode (`align*` is the one that bites) and
    renders others differently, and the reader picks the mode from the
    delimiters: `\\[..\\]` is display, `\\(..\\)` is inline. Compiling a
    display block as inline therefore reports failures the reader never
    sees, and the gate starts "fixing" formulas that were never broken.
    """
    js = f"""
const katex = require({json.dumps(str(KATEX_JS))});
const items = require({json.dumps('__ITEMS__')});
const displays = require({json.dumps('__DISPLAY__')});
for (let i = 0; i < items.length; i++) {{
  try {{
    katex.renderToString(items[i], {{throwOnError: true, displayMode: !!displays[i], strict: false}});
    console.log('OK\\t');
  }} catch (e) {{ console.log('FAIL\\t' + String(e.message).slice(0, 180)); }}
}}
"""
    if display is None:
        flags = [False] * len(items)
    elif isinstance(display, bool):
        flags = [display] * len(items)
    else:
        flags = [bool(d) for d in display]
        if len(flags) != len(items):
            raise ValueError("display 标志个数与公式条数不一致")

    tmp = Path(tempfile.mkdtemp(prefix="katexgate-"))
    items_json = tmp / "items.json"
    items_json.write_text(json.dumps(items), encoding="utf-8")
    flags_json = tmp / "display.json"
    flags_json.write_text(json.dumps(flags), encoding="utf-8")
    script = tmp / "run.js"
    head = js.replace(json.dumps("__ITEMS__"), json.dumps(str(items_json)))
    script.write_text(head.replace(json.dumps("__DISPLAY__"), json.dumps(str(flags_json))),
                      encoding="utf-8")
    r = subprocess.run([NODE, str(script)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    lines = [ln for ln in (r.stdout or "").splitlines() if "\t" in ln]
    if len(lines) != len(items):
        lines += ["FAIL\t(no output)"] * (len(items) - len(lines))
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    grand_total = grand_bad = 0
    for arg in sys.argv[1:]:
        epub = Path(arg)
        rows = extract(epub)
        if not rows:
            print(f"{epub.name}: 没有提取到公式")
            continue
        # The reader typesets `math-body` in display mode and `math-inline`
        # in inline mode; the gate has to ask the same question.
        res = katex_batch([r[2] for r in rows],
                          [r[1] == "math-body" for r in rows])
        bad = [(r, line) for r, line in zip(rows, res) if line.startswith("FAIL")]
        grand_total += len(rows)
        grand_bad += len(bad)
        pct = 100 * (1 - len(bad) / len(rows))
        print(f"{epub.name}")
        print(f"  公式 {len(rows)} · KaTeX 失败 {len(bad)} · 干净率 {pct:.2f}%")
        for (chap, cls, latex), line in bad[:12]:
            print(f"    [{cls}] {chap}  {latex[:88]}")
            print(f"        {line.split(chr(9), 1)[1][:150]}")
        if len(bad) > 12:
            print(f"    ... 另有 {len(bad) - 12} 条")

    if grand_total:
        print(f"\n合计：公式 {grand_total} · 失败 {grand_bad} · "
              f"干净率 {100 * (1 - grand_bad / grand_total):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 10. pipeline/verify.py

````python
#!/usr/bin/env python3
"""verify.py — OCR 输出的三级校验流水线（实验功能）

设计依据
--------
「先廉价检查、只把可疑片段送 LLM 修正」是对的，但**不能只靠 Jev 一层**，因为
Jev（TypeSafe System One）有三条官方自曝的边界：

* **纯文本**——它看不到公式的原始裁剪图；
* **不可靠计数**——不能让它判断「括号是否配对」「符号个数对不对」；
* **不生成文本**——它只能判定，不能修正。

所以正确的分层是三层，各做各擅长的事：

===========  ==============================================  ==========  ==============
层            职责                                            成本        抓什么
===========  ==============================================  ==========  ==============
L0 确定性     括号/环境配对、KaTeX 无头编译、编号连续性        **0**       机械破损、语法错误
L1 Jev        「这条公式可疑吗」的语义判定（Noul/Choice）      ~$0.007/书  语义异常、上下文矛盾
L2 LLM/VLM    带原图裁剪 + 上下文，重识别并修正                ~$0.5/书    真正的公式错误
===========  ==============================================  ==========  ==============

L1 的关键技巧
-------------
Jev 一次调用可以打包多个问题并并行返回，所以**不要一条公式一次调用**，而是把
一整章的所有公式各做成一个 ``noul`` 问题、塞进同一次请求。200 条公式 = 1 次调用。

用法
----
    # 只跑 L0（不需要任何 key，立刻可用）
    python verify.py book.pcex -o report/ --level l0

    # 加 L1（需要 TYPESAFE_API_KEY，早期访问需排队）
    python verify.py book.pcex -o report/ --level l0,l1 --jev-key sk-...

    # 加 L2（需要 OpenAI 兼容端点 + 视觉模型）
    python verify.py book.pcex -o report/ --level l0,l1,l2 \\
        --llm-base https://api.siliconflow.cn/v1 --llm-key sk-... --llm-model Qwen/Qwen2.5-VL-72B-Instruct

    # 三维置信度（完整 / 正确 / 格式清楚）——L1 的另一种问法，按 p_ok 排序出可疑集
    python verify.py book.pcex -o report/ --level l1 --jev-dims all

`--jev-dims` 与默认的 `bad_/ok_` 对比值问法是**两条互斥的路**，不要同时用：
对比值只管「结构有没有坏」一个维度，三维问法把「完整」「正确」「格式清楚」分开问，
每条公式三个 score 问题。实测两者对结构损坏的召回接近，但三维能额外分开
「读错了」与「排得乱」——这两类的修法完全不同（前者要动文本，后者要动 IR 组装）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcexlib import (  # noqa: E402
    Chapter,
    Figure,
    Heading,
    MathBlock,
    MathRun,
    Para,
    Pcex,
    TextRun,
    iter_math,
)

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"ok": 0, "info": 1, "warn": 2, "error": 3}


@dataclass
class Issue:
    level: str          # L0 | L1 | L2
    code: str           # 机器可读的分类
    message: str        # 一句话说明
    detail: str = ""    # 补充信息（位置、概率等）


@dataclass
class Formula:
    """一条待校验的公式（块级或行内）。"""

    fid: str                        # 稳定 id：c04-m012 / c04-i007
    chapter: int
    kind: str                       # display | inline
    latex: str
    page: int | None = None
    bbox: str | None = None
    asset_hash: str | None = None   # 有值表示 pcex 里存了裁剪图，L2 可用
    context: str = ""               # 所在段落文本（L1/L2 的上下文）
    eqno: str | None = None
    issues: list[Issue] = field(default_factory=list)
    fixed_latex: str | None = None
    fixed_by: str | None = None
    confidence: float | None = None  # L1 给出的可疑概率
    # 三维置信度（只有 --jev-dims 会填）：{维度名: {"level": 0..4, "p_ok": 0..1, "probs": {...}}}
    # p_ok = 该维「没问题」的概率 = 等级分布里 LEVELS_OK_FROM 及以上各级的概率之和。
    # 刻意不用 Jev 自带的 confidence 字段：实测它跨类型不可比（探针里 B3 那条最大概率
    # 0.52 却返回 confidence=0.0），而 probabilities 是可信的原始分布。
    dims: dict = field(default_factory=dict)

    @property
    def severity(self) -> str:
        if not self.issues:
            return "ok"
        return max((i.level for i in self.issues), key=lambda lv: {"L0": 3, "L1": 2, "L2": 1}.get(lv, 0))

    @property
    def worst(self) -> str:
        order = ["ok", "info", "warn", "error"]
        rank = {"L0": 2, "L1": 3, "L2": 1}
        best = "ok"
        for i in self.issues:
            sev = "error" if rank.get(i.level, 0) >= 2 else "warn"
            if order.index(sev) > order.index(best):
                best = sev
        return best

    @property
    def effective_latex(self) -> str:
        return self.fixed_latex or self.latex


# ---------------------------------------------------------------------------
# L0 — 确定性检查
# ---------------------------------------------------------------------------

# 这些命令的「可选参数」不计入必填组数
_ZERO_ARG_CMDS = {
    "int", "sum", "prod", "oint", "iint", "iiint", "lim", "max", "min", "sup", "inf",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta",
    "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda",
    "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "infty", "partial", "nabla",
    "cdot", "cdots", "ldots", "dots", "quad", "qquad", "left", "right", "mathrm", "mathbf",
    "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "text", "operatorname", "displaystyle",
    "textstyle", "limits", "nolimits", "big", "Big", "bigg", "Bigg", "hat", "bar", "vec",
    "dot", "ddot", "tilde", "widehat", "widetilde", "overline", "underline", "begin", "end",
}


def _strip_text_blocks(latex: str) -> str:
    """把 \\text{...} / \\mathrm{...} 里的内容挖空，避免把里面的字符当成数学符号。"""
    out, i, n = [], 0, len(latex)
    while i < n:
        m = re.match(r"\\(text|mathrm|mbox|hbox|operatorname|mathbb|mathcal|mathbf|mathit)\s*\{", latex[i:])
        if not m:
            out.append(latex[i])
            i += 1
            continue
        i += m.end()
        depth = 1
        while i < n and depth:
            if latex[i] == "\\":
                i += 2
                continue
            if latex[i] == "{":
                depth += 1
            elif latex[i] == "}":
                depth -= 1
            i += 1
        out.append("\x00")  # 占位符：等长替换，保持索引稳定
    return "".join(out)


def l0_structural(latex: str, display: bool = True) -> list[Issue]:
    """语法/结构层面的确定性检查。不依赖任何模型，也基本不依赖 KaTeX。"""
    issues: list[Issue] = []
    s = _strip_text_blocks(latex)

    # --- 1. 花括号平衡（跳过 \{ \} 转义）---
    # 花括号是 LaTeX 的**结构性**括号，必须严格配对。
    depth, bad = 0, False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                bad = True
                break
        i += 1
    if bad or depth != 0:
        issues.append(Issue("L0", "unbalanced-brace", f"花括号不配对（剩余 {depth:+d}）", latex[:160]))

    # --- 2. 圆括号 / 方括号 ---
    # 数学里 [a,b) 与 (-∞, c] 这类**半开区间**是合法写法，逐类型配对必然误报
    # （实测会误伤 v(0)\\in [0,A) 这类）。所以只检查「开闭总数」是否平衡。
    n_open = n_close = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c in "([":
            n_open += 1
        elif c in ")]":
            n_close += 1
        i += 1
    if n_open != n_close:
        issues.append(
            Issue(
                "L0",
                "unbalanced-paren",
                f"圆/方括号开闭总数不等（{n_open} 开 / {n_close} 闭）",
                latex[:160],
            )
        )

    # --- 2. \left / \right 配对 ---
    n_left = len(re.findall(r"\\left(?![a-zA-Z])", s))
    n_right = len(re.findall(r"\\right(?![a-zA-Z])", s))
    if n_left != n_right:
        issues.append(
            Issue("L0", "unbalanced-left-right", f"\\left({n_left}) 与 \\right({n_right}) 数量不等", latex[:160])
        )

    # --- 3. \begin / \end 配对 ---
    begins = re.findall(r"\\begin\s*\{\s*([^}]*)\s*\}", s)
    ends = re.findall(r"\\end\s*\{\s*([^}]*)\s*\}", s)
    if len(begins) != len(ends):
        issues.append(
            Issue("L0", "unbalanced-env", f"\\begin({len(begins)}) 与 \\end({len(ends)}) 数量不等", latex[:160])
        )
    elif begins != ends:
        issues.append(Issue("L0", "mismatched-env", f"环境名不匹配：{begins} vs {ends}", latex[:160]))

    # --- 4. 孤立上下标 ---
    for m in re.finditer(r"([_^])\s*(?![{\\a-zA-Z0-9\s]|$)", s):
        issues.append(Issue("L0", "dangling-script", f"孤立的上/下标符号 {m.group(1)}", latex[:160]))

    # --- 5. 空分组 ---
    n_empty = len(re.findall(r"\{\s*\}", s))
    if n_empty:
        issues.append(Issue("L0", "empty-group", f"存在 {n_empty} 个空分组 {{}}", latex[:160]))

    # --- 6. 疑似根本不是公式 ---
    # pdf-craft 有时把章节符号当成行内公式，例如「§2.3」被识别成 \S 2.3。
    # 这种条目在 EPUB 里会渲染成一个奇怪的公式，值得单独标出来。
    if re.match(r"^\\S[\s\d.]*$", s.strip()):
        issues.append(
            Issue("L0", "suspect-non-math", "疑似不是公式（章节符号被误判为公式）", latex[:160])
        )

    # 命令名是否合法**交给 KaTeX 的 parser 判断**。
    # 本地维护命令白名单已被实测证明必然误报（\pm \mu \xi \gg \S 全被误判为残缺命令），
    # 而 KaTeX 有一张完整的宏表，比任何手写白名单都可靠。
    return issues


def l0_document(formulas: list[Formula], chapters: list[Chapter]) -> list[Issue]:
    """文档级确定性检查：公式编号连续性、正文引用是否有对应公式。"""
    issues: list[Issue] = []

    # 收集所有编号
    by_eqno: dict[str, list[Formula]] = {}
    for f in formulas:
        if f.eqno:
            by_eqno.setdefault(f.eqno, []).append(f)

    # 1. 编号重复
    for eqno, fs in by_eqno.items():
        if len(fs) > 1:
            for f in fs:
                f.issues.append(
                    Issue("L0", "duplicate-eqno", f"编号 {eqno} 出现 {len(fs)} 次", "、".join(x.fid for x in fs))
                )

    # 2. 正文引用了不存在的编号
    refs: dict[str, list[str]] = {}
    ref_re = re.compile(r"[（(](\d+(?:\.\d+)+)[）)]")
    for ci, ch in enumerate(chapters, 1):
        for b in ch.blocks:
            texts = []
            if isinstance(b, (Para, Heading)):
                texts.append("".join(r.text for r in b.runs if isinstance(r, TextRun)))
            if texts:
                for m in ref_re.finditer(" ".join(texts)):
                    refs.setdefault(m.group(1), []).append(f"c{ci:02d}")
    missing = {k: v for k, v in refs.items() if f"({k})" not in by_eqno}
    if missing:
        sample = "、".join(f"({k})" for k in list(missing)[:12])
        issues.append(
            Issue(
                "L0",
                "missing-referenced-eqno",
                f"正文引用了 {len(missing)} 个不存在的公式编号",
                sample,
            )
        )

    # 3. 编号不连续（同一前缀内跳号）
    groups: dict[str, list[int]] = {}
    for eqno in by_eqno:
        m = re.match(r"\((\d+)\.(\d+)\)", eqno)
        if m:
            groups.setdefault(m.group(1), []).append(int(m.group(2)))
    gaps = []
    for prefix, nums in sorted(groups.items()):
        nums = sorted(set(nums))
        for a, b in zip(nums, nums[1:]):
            if b - a > 1:
                gaps.append(f"{prefix}.{a}→{prefix}.{b}")
    if gaps:
        issues.append(Issue("L0", "eqno-gap", f"公式编号跳号 {len(gaps)} 处", "、".join(gaps[:10])))

    return issues


# ---------------------------------------------------------------------------
# L0 — KaTeX 无头编译（最强的语法检查）
# ---------------------------------------------------------------------------

_KATEX_JS = r"""
const katex = require(process.argv[2]);
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = {};
for (const it of input) {
  try {
    katex.renderToString(it.latex, {
      displayMode: !!it.display,
      throwOnError: true,
      strict: false,
      trust: false,
      maxSize: 100,
    });
    out[it.id] = { ok: true };
  } catch (e) {
    out[it.id] = {
      ok: false,
      name: e && e.name ? String(e.name) : 'Error',
      message: e && e.message ? String(e.message) : String(e),
      pos: e && typeof e.position === 'number' ? e.position : null,
    };
  }
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def find_katex_js(explicit: str | None = None) -> Path | None:
    """定位可用于 Node 的 katex.js（UMD 版）。"""
    cands = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get("KATEX_DIR")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve().parent
    cands += [
        here / "assets" / "katex" / "katex.js",
        here.parent / "work" / "katex_tmp" / "package" / "dist" / "katex.js",
    ]
    for p in cands:
        if p.is_file():
            return p
        if p.is_dir() and (p / "katex.js").is_file():
            return p / "katex.js"
    return None


def find_node() -> str | None:
    for name in ("node", "node.exe"):
        p = shutil.which(name)
        if p:
            return p
    for p in (
        r"C:\Users\hp\.workbuddy\binaries\node\versions\22.22.2-3\node.exe",
        r"C:\Program Files\nodejs\node.exe",
    ):
        if Path(p).is_file():
            return p
    return None


def l0_katex(formulas: list[Formula], katex_js: Path | None, node: str | None) -> tuple[bool, str]:
    """用 KaTeX 真正的 parser 跑一遍所有公式。返回 (是否执行, 说明)。"""
    if not katex_js or not node:
        return False, "未找到 KaTeX/katex.js 或 node，跳过编译校验"
    payload = [
        {"id": f.fid, "latex": f.latex, "display": f.kind == "display"} for f in formulas
    ]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "check.js").write_text(_KATEX_JS, encoding="utf-8")
        (td / "in.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            r = subprocess.run(
                [node, str(td / "check.js"), str(katex_js).replace("\\", "/"), str(td / "in.json"), str(td / "out.json")],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(td),
            )
        except Exception as e:  # noqa: BLE001
            return False, f"KaTeX 校验执行失败：{e}"
        if r.returncode != 0 or not (td / "out.json").exists():
            return False, f"KaTeX 校验失败：{(r.stderr or r.stdout or '')[:400]}"
        results = json.loads((td / "out.json").read_text(encoding="utf-8"))

    bad = 0
    for f in formulas:
        res = results.get(f.fid)
        if res and not res.get("ok"):
            bad += 1
            pos = res.get("pos")
            detail = res.get("message", "")
            if pos is not None and 0 <= pos < len(f.latex):
                detail += f"  ← 出错位置附近：…{f.latex[max(0, pos-30):pos+30]}…"
            f.issues.append(Issue("L0", "katex-parse-error", f"KaTeX 无法解析：{res.get('name')}", detail))
    return True, f"KaTeX 编译校验完成，{bad}/{len(formulas)} 条失败"


# ---------------------------------------------------------------------------
# L1 — Jev（TypeSafe System One）
# ---------------------------------------------------------------------------

JEV_DEFAULT_ENDPOINT = "https://aiask.me/v1/systemone"

# 用英文写 instructions：Jev 是 English-first。
#
# 关键：**问题必须点名是哪一条**。Jev 的 state 是整体喂进去的、questions 并行且互相隔离，
# 问题 id 不会送给模型（官方原话：ID is for your code）。所以「这条公式坏了吗」这种不点名的
# 问法，模型回答的是这句 instruction 的语气，不是那一条公式——
# 实测一条缺右花括号的公式和一条完好的公式都拿到 noul≈0.88，完全没有区分度。
JEV_BAD = (
    "Item '{fid}' is MALFORMED: its LaTeX has a structural defect -- an unbalanced brace, "
    "an unclosed environment or delimiter, a duplicated superscript, a command that does "
    "not exist, or text truncated in the middle."
)
JEV_OK = (
    "Item '{fid}' is WELL-FORMED: its LaTeX is complete and compiles exactly as written."
)

# 判定用「对比值」而不是单独的 p_bad。
#
# 标定实测（UCBH113，9 条真坏 + 10 条合成坏 + 10 条好）：单看 p_bad，真坏的平均只有 +0.14 左右，
# 阈值卡 0.5 会漏掉一大半；而 p_ok 更低，p_bad - p_ok 的对比值把两类的间隔拉开到
# 好 [-0.91,-0.67] 与 坏 [-0.48,+0.87] 完全分离（AUC 1.000）。
# 直觉：Jev 有一个偏高的「可疑」先验，问两个相反的问题让这个偏置相互抵消。
JEV_CONTRAST_FLAG = 0.0


def l1_jev(
    formulas: list[Formula],
    api_key: str,
    model: str = "jev-latest",
    batch_chars: int = 24000,
    threshold: float = JEV_CONTRAST_FLAG,
    timeout: int = 90,
    endpoint: str = JEV_DEFAULT_ENDPOINT,
    book_context: str = "A STEM textbook OCR'd from printed pages. Formulas are stored as LaTeX source.",
) -> tuple[bool, str]:
    """把一批公式打包成一次 Jev 调用，用成对的 Noul 问「哪一条结构有问题」。

    一次调用可并行回答多个问题，所以按字符预算分组、组内一次请求。
    每条公式花两个问题（bad_/ok_），拿两个概率的差值作为可疑度。
    """
    import urllib.error
    import urllib.request

    # 按「上下文 + 公式」的字符量分组，避免超 64K token 限制
    batches: list[list[Formula]] = []
    cur: list[Formula] = []
    size = 0
    for f in formulas:
        n = len(f.latex) + len(f.context)
        if cur and size + n > batch_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += n
    if cur:
        batches.append(cur)

    done = 0
    flagged = 0
    errs: list[str] = []
    for bi, batch in enumerate(batches, 1):
        state = {
            "book": book_context,
            "items": [
                {
                    "id": f.fid,
                    "page": f.page,
                    "kind": f.kind,
                    "paragraph": f.context[:1200],
                    "latex": f.latex,
                }
                for f in batch
            ],
        }
        questions: dict = {}
        for f in batch:
            questions[f"bad_{f.fid}"] = {
                "type": "noul",
                "instructions": JEV_BAD.format(fid=f.fid),
                "criteria": {
                    "true": "The item's LaTeX cannot compile, or is visibly truncated",
                    "false": "The item's LaTeX is complete and compiles as written",
                },
            }
            questions[f"ok_{f.fid}"] = {
                "type": "noul",
                "instructions": JEV_OK.format(fid=f.fid),
            }
        body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            errs.append(f"批次{bi}: HTTP {e.code} {e.read()[:200]!r}")
            continue
        except Exception as e:  # noqa: BLE001
            errs.append(f"批次{bi}: {type(e).__name__}: {e}")
            continue

        answers = data.get("answers") or data.get("results") or {}
        for f in batch:
            done += 1
            a_bad = answers.get(f"bad_{f.fid}") or {}
            a_ok = answers.get(f"ok_{f.fid}") or {}
            p_bad = a_bad.get("noul")
            p_ok = a_ok.get("noul")
            if p_bad is None:
                continue
            contrast = float(p_bad) - float(p_ok if p_ok is not None else 0.0)
            f.confidence = contrast
            if contrast >= threshold:
                flagged += 1
                f.issues.append(
                    Issue(
                        "L1",
                        "jev-suspicious",
                        f"Jev 判定可疑（对比值 {contrast:+.2f}，p_bad={float(p_bad):.2f}"
                        f"{f', p_ok={float(p_ok):.2f}' if p_ok is not None else ''}）",
                        "",
                    )
                )

    if errs:
        return False, "；".join(errs[:3])
    return True, (
        f"Jev 完成：{len(batches)} 次调用，{done} 条判定，标出 {flagged} 条可疑"
        f"（阈值对比值≥{threshold:+.2f}）"
    )


# ---------------------------------------------------------------------------
# L1b — 三维置信度（完整 / 正确 / 格式清楚）
# ---------------------------------------------------------------------------
#
# 为什么另起一套问法，而不是把 JEV_BAD 那句话扩写成三条：
# * `noul` 只给一个标量概率，问三维就得问三对、每条公式三个问题变成六个；
# * `score` 的 `criteria` 是**有序数组**，回答直接落在等级轴上（实测会给 1.01、2.74
#   这种级间值），而且**附带整条等级分布** `probabilities`——这才是「置信度」该用的载体：
#   把「3 级及以上」的概率加起来就是「这一维没问题的概率」，比单个标量信息量大得多。
# * 实测 15 个问题 1791 输入 token，跟 noul 同量级，但一条公式问三维只花三个问题。
#
# 三维必须分开问，不能合并成一句「这条公式好不好」。探针结果说明了原因：
# `\llangledown_{i} x_i`（命令不存在，KaTeX 直接报错）在「完整性」维度拿到 2.74/4、
# 概率 0.52 压在「肯定完整」那一级——它确实完整，只是错。合并成一问就会漏掉它。
JEV_DIM_ORDER = ("complete", "correct", "format")

# 等级 >= 这个值算「这一维没问题」。取 3（即「大体没问题」及以上）而不是 4，
# 是因为 4 级要求「绝对确定」，实测这会把大量正常公式压进低置信，失去区分度。
LEVELS_OK_FROM = 3

JEV_DIMS: dict[str, dict] = {
    "complete": {
        "name": "完整性",
        # 只问「源码本身是否自足」，不问对错——这一维的判据完全在 LaTeX 文本内部，
        # 所以它是三维里最可靠的一维，也是唯一能和 KaTeX 硬失败对标做标定的。
        "ask": (
            "Rate how COMPLETE the LaTeX source of item '{fid}' is: does the source contain the "
            "whole formula, with nothing lost, cut off, or left unclosed? Judge only the source "
            "text of item '{fid}'."
        ),
        "levels": [
            "Severely truncated: the formula is cut off and large parts are missing",
            "Clearly incomplete: a group, environment or delimiter is left open",
            "Uncertain: something may be missing, but it cannot be told from the source alone",
            "Nearly complete: nothing appears to be missing",
            "Definitely complete: the source holds the whole formula and is self-contained",
        ],
    },
    "correct": {
        "name": "正确性",
        # 这一维的**能力边界必须说清楚**：Jev 看不到扫描原图，所以它判的是
        # 「与上下文段落是否自洽」，能抓命令不存在、与正文矛盾、量纲明显不对这类；
        # **抓不到**「扫描件上是 7 被读成 1」这种纯图像层的误读。报告里要如实标注，
        # 否则这个数字会被当成「与书上一致」的概率——
        #
        # 末句那条「裸符号不算错」是**标定之后补上的规则**，不是随手写的客套话。
        # 没有它时，`z`、`\epsilon`、`2^{\circ}` 这类没有断言的片段会被 Jev 判成
        # 「无法判断」，而「无法判断」在 p_ok 口径里等于不通过——实测 120 条样本里
        # correct 低置信 23 条，逐条看几乎全是裸符号，把真错（`r_{2}^{i\theta_{2}}`
        # 这种把 e 吃进上标的）全淹了。加规则前后的对照见
        # `work/jev_dims_ab.py` 的输出。
        "ask": (
            "Rate how CORRECT item '{fid}' is as mathematics, judged against the surrounding "
            "paragraph quoted in the state: do its symbols, indices and relations agree with that "
            "text, or is a symbol plausibly misread? Judge only item '{fid}'. "
            "If item '{fid}' is a bare symbol, a single variable, or a fragment that asserts no "
            "relation, then it cannot be incorrect on its own: rate it as definitely correct "
            "unless that symbol itself is implausible in the surrounding text."
        ),
        "levels": [
            "Definitely wrong: it contradicts the surrounding paragraph or is mathematical nonsense",
            "Probably wrong: a symbol, index or operator is very likely misread",
            "Uncertain: it cannot be judged from the text available",
            "Probably correct: consistent with the surrounding paragraph",
            "Definitely correct: every symbol and relation fits the context",
        ],
    },
    "format": {
        "name": "格式清楚",
        # 「清楚」在这里是可操作的：分组无歧义、环境完好、没有把散文词塞进数学模式、
        # 没有把该分开的符号粘成一串。这些都是排版层的问题，与对错无关——
        # 一条读错数字的公式照样可以排得很清楚。
        "ask": (
            "Rate how CLEARLY FORMATTED item '{fid}' is: is its LaTeX unambiguous and readable, "
            "with well-formed grouping and with no prose or stray text trapped inside the math? "
            "Judge only item '{fid}'."
        ),
        "levels": [
            "Unreadable: symbols or words run together, delimiters tangled, no structure left",
            "Poorly formatted: ambiguous grouping, stray prose inside the math, or a broken environment",
            "Uncertain: it is unclear whether the formatting conveys the intended structure",
            "Mostly clear: readable, with at most a minor oddity",
            "Definitely clear: unambiguous grouping and well-formed structure",
        ],
    },
}

# 三维联合置信度取**最小值**而不是平均：平均会把「一维很糟、两维很好」抬到及格线，
# 而只要有任何一维塌了，这条公式就不该被当成可信。
JEV_DIM_FLAG = 0.5


def _dim_p_ok(answer: dict, n_levels: int) -> float:
    """从 score 答案的等级分布里取「这一维没问题」的概率。"""
    probs = answer.get("probabilities") or {}
    return float(sum(float(probs.get(str(i), 0.0)) for i in range(LEVELS_OK_FROM, n_levels)))


def _dim_level(answer: dict) -> float | None:
    """Jev 给的等级（可落在两级之间，比如 2.74）。"""
    v = answer.get("score")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def l1_jev_dims(
    formulas: list[Formula],
    api_key: str,
    model: str = "jev-latest",
    dims: tuple[str, ...] = JEV_DIM_ORDER,
    batch_chars: int = 16000,
    batch_max: int = 80,
    threshold: float = JEV_DIM_FLAG,
    timeout: int = 150,
    endpoint: str = JEV_DEFAULT_ENDPOINT,
    book_context: str = "A STEM textbook OCR'd from printed pages.",
    retries: int = 2,
    progress: bool = True,
    stats: dict | None = None,
) -> tuple[bool, str]:
    """对每条公式问三维 score 问题，写出 f.dims，并按最小 p_ok 标出低置信。

    分组同时卡字符数与条数：三维问法比原来的 noul 对多 50% 的问题 token，
    只按字符分组会顶到 64k 上限（官方限制：state + 全部问题 ≈ 64k tokens）。

    失败批次会重试（服务侧偶发 5xx），重试仍失败就跳过并记进 errs——
    宁可有几条没判定，也不能整本白跑。
    """
    import urllib.error
    import urllib.request

    dims = tuple(d for d in dims if d in JEV_DIMS)
    if not dims:
        return False, "没有可用的维度名"

    batches: list[list[Formula]] = []
    cur: list[Formula] = []
    size = 0
    for f in formulas:
        n = len(f.latex) + len(f.context)
        if cur and (size + n > batch_chars or len(cur) >= batch_max):
            batches.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += n
    if cur:
        batches.append(cur)

    done = 0
    errs: list[str] = []
    per_dim_low: dict[str, int] = {d: 0 for d in dims}
    tok_in = tok_out = 0

    for bi, batch in enumerate(batches, 1):
        state = {
            "book": book_context,
            "items": [
                {
                    "id": f.fid,
                    "page": f.page,
                    "kind": f.kind,
                    "paragraph": f.context[:1200],
                    "latex": f.latex,
                }
                for f in batch
            ],
        }
        questions: dict = {}
        for f in batch:
            for d in dims:
                questions[f"{d}__{f.fid}"] = {
                    "type": "score",
                    "instructions": JEV_DIMS[d]["ask"].format(fid=f.fid),
                    "criteria": JEV_DIMS[d]["levels"],
                }
        body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

        answers = None
        last_err = ""
        fatal_err = ""
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                answers = data.get("answers") or data.get("results") or {}
                u = data.get("usage") or {}
                # 记 token 是为了能报出真实成本。全库 12.8 万条公式，不记账不敢跑。
                tok_in += int(u.get("input_tokens") or 0)
                tok_out += int(u.get("output_tokens") or 0)
                break
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                last_err = f"HTTP {e.code} {raw[:160]}"
                # 余额/额度类错误**重试没有意义**，而且必须让整本停下：
                # 实测余额耗尽返回 `403 {"code":"INSUFFICIENT_BALANCE"}`，
                # 当时它被当成可重试错误，于是脚本在 403 上空转到预算耗尽——
                # 更糟的是会把「只判了几条」的残结果落盘，下次重跑直接跳过这本。
                if e.code in (401, 402, 403) or any(
                    k in raw.upper() for k in ("BALANCE", "QUOTA", "CREDIT")
                ):
                    fatal_err = last_err
                    break
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
        if answers is None:
            errs.append(f"批次{bi}({len(batch)}条): {last_err}")
            if fatal_err:
                errs.append("额度/鉴权类错误，已中止本书剩余批次")
                break
            continue

        for f in batch:
            got_any = False
            for d in dims:
                a = answers.get(f"{d}__{f.fid}") or {}
                if not a:
                    continue
                n_levels = len(JEV_DIMS[d]["levels"])
                f.dims[d] = {
                    "level": _dim_level(a),
                    "p_ok": round(_dim_p_ok(a, n_levels), 4),
                    "probs": a.get("probabilities") or {},
                }
                got_any = True
            if not got_any:
                continue
            done += 1
            weak = [(d, f.dims[d]["p_ok"]) for d in dims if d in f.dims and f.dims[d]["p_ok"] < threshold]
            f.confidence = round(min(v["p_ok"] for v in f.dims.values()), 4)
            for d, p in weak:
                per_dim_low[d] += 1
                f.issues.append(
                    Issue(
                        "L1",
                        f"jev-low-{d}",
                        f"{JEV_DIMS[d]['name']}置信度低（p_ok={p:.2f}，"
                        f"等级={f.dims[d]['level']}，共 {len(JEV_DIMS[d]['levels'])} 级）",
                        "",
                    )
                )
        if progress:
            print(f"      [L1b] 批次 {bi}/{len(batches)} 完成，累计 {done} 条", flush=True)

    detail = "，".join(f"{d} 低置信 {per_dim_low[d]}" for d in dims)
    if stats is not None:
        stats.update(
            batches=len(batches), judged=done, tokens_in=tok_in, tokens_out=tok_out,
            per_dim_low=per_dim_low, errors=errs[:5],
        )
    if errs:
        return False, f"三维置信度部分失败：{'；'.join(errs[:3])}（已判定 {done} 条，{detail}）"
    return True, (
        f"三维置信度完成：{len(batches)} 次调用，{done} 条判定；"
        f"{detail}（阈值 p_ok<{threshold}）；"
        f"tokens 入 {tok_in:,} / 出 {tok_out:,}"
    )


# ---------------------------------------------------------------------------
# L2 — LLM/VLM 修正
# ---------------------------------------------------------------------------


def l2_fix(
    formulas: list[Formula],
    api_base: str,
    api_key: str,
    model: str,
    pcex: Pcex | None = None,
    outdir: Path | None = None,
    max_items: int = 200,
    timeout: int = 120,
) -> tuple[bool, str]:
    """只对 L0/L1 标出的可疑公式调用 LLM；有裁剪图时走视觉模型。"""
    import urllib.error
    import urllib.request

    targets = [f for f in formulas if f.issues][:max_items]
    if not targets:
        return True, "L2：没有需要修正的公式"

    system = (
        "你是数学排版专家。用户会给你一段来自扫描教材的 LaTeX 公式，可能还有原图裁剪。"
        "如果你的确能判断它写错了，请输出修正后的 LaTeX；如果不确定或原文正确，"
        "只输出原样 LaTeX。不要解释，不要加代码块，只输出 LaTeX 源码。"
    )
    fixed = 0
    errs: list[str] = []
    for f in targets:
        user_parts = [f"上下文段落：{f.context[:800] or '（无）'}", f"当前 LaTeX：{f.latex}"]
        content: list = [{"type": "text", "text": "\n".join(user_parts)}]

        # 有裁剪图 → 附上，让视觉模型真正看图
        img_b64 = None
        if pcex is not None and f.asset_hash:
            raw = pcex.read_asset(f.asset_hash)
            if raw:
                import base64

                img_b64 = base64.b64encode(raw).decode("ascii")
        if img_b64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            )
            if outdir is not None:
                imgdir = outdir / "crops"
                imgdir.mkdir(parents=True, exist_ok=True)
                import base64 as _b64

                (imgdir / f"{f.fid}.png").write_bytes(_b64.b64decode(img_b64))

        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": 1024,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            errs.append(f"{f.fid}: HTTP {e.code} {e.read()[:160]!r}")
            continue
        except Exception as e:  # noqa: BLE001
            errs.append(f"{f.fid}: {e}")
            continue
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            errs.append(f"{f.fid}: 响应结构异常")
            continue
        text = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", text, flags=re.M).strip()
        if text and text != f.latex:
            f.fixed_latex = text
            f.fixed_by = model
            f.issues.append(Issue("L2", "llm-suggested-fix", "模型给出了修正版本", text[:200]))
            fixed += 1

    if errs:
        return False, f"L2 部分失败：{'；'.join(errs[:3])}"
    return True, f"L2 完成：检查 {len(targets)} 条，其中 {fixed} 条给出修正"


# ---------------------------------------------------------------------------
# 收集 + 报告
# ---------------------------------------------------------------------------


def collect(pcex: Pcex) -> tuple[list[Formula], list[Chapter]]:
    """收集公式。

    编号一定走 pcexlib.iter_math —— 渲染端套用修正时用的是同一份实现，
    两边的 id 因此不可能对不上。这里只负责补上「上下文段落」这一项。
    """
    chapters = pcex.chapters(attach_toc_titles=False)
    formulas: list[Formula] = []
    for r in iter_math(chapters):
        ch = chapters[r.chapter - 1]
        bi = r.block
        blk = ch.blocks[bi]
        # 前后各一个文本块充当上下文，给 L1/L2 判断用
        ctx = ""
        for j in (bi - 1, bi + 1):
            if 0 <= j < len(ch.blocks) and isinstance(ch.blocks[j], (Para, Heading)):
                nb = ch.blocks[j]
                ctx += "".join(x.text for x in nb.runs if isinstance(x, TextRun))

        if r.kind == "display":
            formulas.append(
                Formula(
                    fid=r.fid,
                    chapter=r.chapter,
                    kind="display",
                    latex=r.latex,
                    page=blk.page,
                    bbox=blk.bbox,
                    asset_hash=blk.asset_hash,
                    context=ctx.strip(),
                    eqno=blk.eqno,
                )
            )
        else:
            formulas.append(
                Formula(
                    fid=r.fid,
                    chapter=r.chapter,
                    kind="inline",
                    latex=r.latex,
                    page=blk.page,
                    bbox=blk.bbox,
                    context="".join(x.text for x in blk.runs if isinstance(x, TextRun)).strip(),
                )
            )
    return formulas, chapters
    return formulas, chapters


def dim_stats(formulas: list[Formula], dim: str, low: float = JEV_DIM_FLAG) -> dict:
    """某一维的分布统计。没跑三维就返回空字典。

    只报均值和最低值会骗人：p_ok 是个重尾分布（绝大多数贴 1.0），均值看着很高，
    真正要看的其实是左尾有多少条。所以中位与「低于阈值的条数」必须一起报。
    """
    vals = sorted(f.dims[dim]["p_ok"] for f in formulas if dim in f.dims)
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": vals[len(vals) // 2],
        "low": sum(1 for v in vals if v < low),
        "min": vals[0],
    }


def write_reports(formulas: list[Formula], chapters: list[Chapter], outdir: Path, notes: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "verify.json").write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(formulas),
                    "with_issues": sum(1 for f in formulas if f.issues),
                    "l0_issues": sum(1 for f in formulas if any(i.level == "L0" for i in f.issues)),
                    "l1_flagged": sum(1 for f in formulas if any(i.level == "L1" for i in f.issues)),
                    "l2_fixed": sum(1 for f in formulas if f.fixed_latex),
                    "dims": {
                        d: dim_stats(formulas, d)
                        for d in JEV_DIM_ORDER
                        if any(d in f.dims for f in formulas)
                    },
                },
                "notes": notes,
                "formulas": [
                    {
                        **{k: v for k, v in asdict(f).items() if k != "issues"},
                        "issues": [asdict(i) for i in f.issues],
                        "severity": f.worst,
                    }
                    for f in formulas
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bad = [f for f in formulas if f.issues]
    lines = [
        "# OCR 公式校验报告",
        "",
        "## 摘要",
        "",
        f"- 公式总数：**{len(formulas)}**（块级 {sum(1 for f in formulas if f.kind=='display')}，"
        f"行内 {sum(1 for f in formulas if f.kind=='inline')}）",
        f"- 有问题：**{len(bad)}**",
        f"- L0 命中：{sum(1 for f in formulas if any(i.level=='L0' for i in f.issues))}",
        f"- L1 标出：{sum(1 for f in formulas if any(i.level=='L1' for i in f.issues))}",
        f"- L2 修正：{sum(1 for f in formulas if f.fixed_latex)}",
        "",
    ]
    if notes:
        lines += ["## 执行记录", ""] + [f"- {n}" for n in notes] + [""]

    # --- 三维置信度 ---------------------------------------------------
    dims_used = [d for d in JEV_DIM_ORDER if any(d in f.dims for f in formulas)]
    if dims_used:
        lines += [
            "## 三维置信度（Jev score 问法）",
            "",
            "`p_ok` = Jev 判定该维「没问题」的概率（5 级量表里 3 级及以上的概率之和）。",
            "联合置信度取**三维 p_ok 的最小值**——任何一维塌了，这条公式就不该被当成可信。",
            "",
            "| 维度 | 判定条数 | 均 p_ok | 中位 | 低置信条数 | 最低 |",
            "|---|---|---|---|---|---|",
        ]
        for d in dims_used:
            s = dim_stats(formulas, d)
            lines.append(
                f"| {d}（{JEV_DIMS[d]['name']}） | {s['n']} | {s['mean']:.3f} | "
                f"{s['median']:.3f} | {s['low']} | {s['min']:.3f} |"
            )
        lines += [
            "",
            "> **`correct` 维的能力边界**：Jev 看不到扫描原图，它判的是「与上下文段落是否自洽」，",
            "> 能抓出命令不存在、与正文矛盾、量纲明显不对这类；**抓不到**纯图像层的误读",
            "> （例如扫描件上是 7 被读成 1）。所以它读作「言语自洽度」，不是「与书上一致的概率」。",
            "",
        ]
        worst = sorted(
            (f for f in formulas if f.dims),
            key=lambda f: min(v["p_ok"] for v in f.dims.values()),
        )[:40]
        lines += ["### 置信度最低的 40 条", ""]
        for f in worst:
            combo = min(v["p_ok"] for v in f.dims.values())
            cell = " · ".join(
                f"{d} {f.dims[d]['p_ok']:.2f}" for d in dims_used if d in f.dims
            )
            lines.append(f"- `{f.fid}` · {f.kind} · 第 {f.page or '?'} 页 · 联合 {combo:.2f} — {cell}")
            lines.append(f"  - `{f.latex[:160]}`")
        lines.append("")

    by_code: dict[str, int] = {}
    for f in formulas:
        for i in f.issues:
            by_code[i.code] = by_code.get(i.code, 0) + 1
    if by_code:
        lines += ["## 问题类型分布", "", "| 类型 | 数量 |", "|---|---|"]
        for k, v in sorted(by_code.items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    if bad:
        lines += ["## 明细（仅列有问题的）", ""]
        for f in bad:
            tag = "★" if f.fixed_latex else "·"
            lines.append(f"### {tag} `{f.fid}` · {f.kind} · 第 {f.page or '?'} 页")
            lines.append("")
            lines.append(f"```latex\n{f.latex}\n```")
            for i in f.issues:
                lines.append(f"- **[{i.level}] {i.code}** — {i.message}")
                if i.detail:
                    lines.append(f"  - `{i.detail}`")
            if f.fixed_latex:
                lines.append("")
                lines.append("修正建议：")
                lines.append(f"```latex\n{f.fixed_latex}\n```")
            lines.append("")

    (outdir / "verify.md").write_text("\n".join(lines), encoding="utf-8")


def jev_api_key() -> str:
    """先环境变量，再用户私有文件——和 batch.py 取 SF key 的约定一致。

    密钥不落项目目录、不进日志。"""
    k = (os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or "").strip()
    if k:
        return k
    key_file = Path.home() / ".workbuddy" / "secrets" / "aiask.key"
    if key_file.is_file():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="OCR 公式三级校验")
    ap.add_argument("pcex")
    ap.add_argument("-o", "--outdir", default="verify_out")
    ap.add_argument("--level", default="l0", help="逗号分隔：l0,l1,l2")
    ap.add_argument("--katex-dir", default=None, help="含 katex.js 的目录")
    ap.add_argument("--jev-key", default=None)
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--jev-endpoint", default=JEV_DEFAULT_ENDPOINT,
                    help="System One 端点；换网关时改这里")
    ap.add_argument("--jev-threshold", type=float, default=JEV_CONTRAST_FLAG,
                    help="可疑度阈值，判据是 p_bad-p_ok 的对比值（不是 p_bad 本身）")
    ap.add_argument("--jev-dims", default=None,
                    help="走三维置信度模式：逗号分隔 complete,correct,format（或 all）。"
                         "给了这个就不再跑 bad_/ok_ 对比值问法")
    ap.add_argument("--jev-dim-threshold", type=float, default=JEV_DIM_FLAG,
                    help="三维模式下，某一维 p_ok 低于此值即标出（默认 0.5），"
                         "联合置信度取三维 p_ok 的最小值")
    ap.add_argument("--llm-base", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--llm-key", default=os.environ.get("LLM_API_KEY"))
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--l2-max", type=int, default=50, help="L2 最多修正多少条（控成本）")
    ap.add_argument("--max-formulas", type=int, default=0,
                    help="只校验前 N 条公式（0=全部）。冒烟测试与控成本用；"
                         "被截断会在报告里注明，免得把抽样结果当成全量结论")
    a = ap.parse_args()

    if a.jev_key is None:
        a.jev_key = jev_api_key()

    levels = [x.strip().lower() for x in a.level.split(",") if x.strip()]
    outdir = Path(a.outdir)
    notes: list[str] = []

    with Pcex(a.pcex) as p:
        formulas, chapters = collect(p)
        print(f"[i] 收集到 {len(formulas)} 条公式")
        if a.max_formulas and len(formulas) > a.max_formulas:
            n_all = len(formulas)
            formulas = formulas[:a.max_formulas]
            notes.append(
                f"⚠ 抽样：只校验了前 {a.max_formulas} / {n_all} 条公式，"
                f"以下所有比例都只是样本，不是全书结论"
            )
            print(f"[i] 抽样：只校验前 {a.max_formulas} 条（共 {n_all} 条）")

        if "l0" in levels:
            for f in formulas:
                f.issues.extend(l0_structural(f.latex, f.kind == "display"))
            doc_issues = l0_document(formulas, chapters)
            notes.append(f"L0 文档级检查：{len(doc_issues)} 项")
            for i in doc_issues:
                print(f"    [L0] {i.code} — {i.message} {i.detail}")

            katex_js = find_katex_js(a.katex_dir)
            node = find_node()
            ok, msg = l0_katex(formulas, katex_js, node)
            notes.append(f"L0 KaTeX：{msg}")
            print(f"    {msg}")
            n0 = sum(1 for f in formulas if any(i.level == "L0" for i in f.issues))
            print(f"[L0] 命中 {n0} 条")

        if "l1" in levels:
            if not a.jev_key:
                notes.append("L1 跳过：未提供 JEV key")
                print("[!] L1 需要 Jev key（--jev-key，或环境变量 TYPESAFE_API_KEY / JEV_API_KEY）")
            else:
                ctx = (
                    f"A STEM textbook (title: {Path(a.pcex).stem}) OCR'd from printed pages. "
                    f"Formulas are stored as LaTeX source."
                )
                if a.jev_dims:
                    want = JEV_DIM_ORDER if a.jev_dims.strip().lower() == "all" else tuple(
                        x.strip() for x in a.jev_dims.split(",") if x.strip()
                    )
                    unknown = [d for d in want if d not in JEV_DIMS]
                    if unknown:
                        print(f"[!] 未知维度 {unknown}，可用：{list(JEV_DIMS)}")
                        notes.append(f"L1 跳过：未知维度 {unknown}")
                    else:
                        print(f"[L1b] 三维置信度模式：{list(want)}，共 {len(formulas)} 条公式")
                        ok, msg = l1_jev_dims(
                            formulas, a.jev_key, a.jev_model, dims=want,
                            threshold=a.jev_dim_threshold, endpoint=a.jev_endpoint,
                            book_context=ctx,
                        )
                        notes.append(f"L1b 三维置信度：{msg}")
                        print(f"[L1b] {msg}")
                        for d in want:
                            s = dim_stats(formulas, d)
                            if s:
                                print(f"     {d:<9} n={s['n']:<6} 均 p_ok {s['mean']:.3f}  "
                                      f"中位 {s['median']:.3f}  <{a.jev_dim_threshold} 的 {s['low']} 条  "
                                      f"最低 {s['min']:.3f}")
                else:
                    ok, msg = l1_jev(
                        formulas, a.jev_key, a.jev_model,
                        threshold=a.jev_threshold, endpoint=a.jev_endpoint,
                        book_context=ctx,
                    )
                    notes.append(f"L1 Jev：{msg}")
                    print(f"[L1] {msg}")
                # 打印可疑度分布，阈值要按真实分布调，不能拍脑袋
                conf = sorted(
                    (f.confidence for f in formulas if f.confidence is not None), reverse=True
                )
                if conf:
                    def _pct(p: float) -> float:
                        return conf[min(len(conf) - 1, int(len(conf) * p))]
                    print(f"     对比值分布：最大 {conf[0]:+.2f}  "
                          f"top1% {_pct(0.01):+.2f}  top5% {_pct(0.05):+.2f}  "
                          f"中位 {_pct(0.5):+.2f}  最小 {conf[-1]:+.2f}")

        if "l2" in levels:
            if not (a.llm_base and a.llm_key and a.llm_model):
                notes.append("L2 跳过：未提供 --llm-base/--llm-key/--llm-model")
                print("[!] L2 需要 OpenAI 兼容端点 + 模型名（视觉模型效果最好）")
            else:
                ok, msg = l2_fix(
                    formulas, a.llm_base, a.llm_key, a.llm_model,
                    pcex=p, outdir=outdir, max_items=a.l2_max,
                )
                notes.append(f"L2：{msg}")
                print(f"[L2] {msg}")

        write_reports(formulas, chapters, outdir, notes)

    print(f"[✓] 报告：{outdir/'verify.md'}  /  {outdir/'verify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

### 11. work/autopilot.py

```python
"""autopilot —— 全库 PDF→EPUB 的可中断推进器。

## 为什么需要它

本机**无法让进程脱离会话**：`Start-Process` 起出来的进程在工具调用结束时即被杀
（2026-09-22 实测：pid 4040，心跳文件只写下 start 行、零 tick，进程表里也消失）。
所以"续航"不能靠常驻守护，只能靠**每次调用都幂等可续跑**：

- 会话中断只损失"正在跑的那几本"，已完成的书由 `batch.py` 逐本落盘进 manifest；
- 下一次只要再跑一次本脚本，自动从断点继续，不需要任何人工记账。

## 设计要点

1. **锁**：`work/autopilot.lock` 记 pid + 心跳。心跳超过 `--stale-min` 分钟视为僵尸锁，
   可抢占 —— 否则一次崩溃就会把后续所有调用永久挡在门外。
2. **小书优先**：按页数升序（`--sort small` 是 batch.py 默认）。先做小的，
   每段运行都能真正"出成品"，而不是几小时后手里还是半本大书。
3. **切片推进**：每轮只取 `--slice` 本交给 batch.py。批处理本身逐本落盘，
   切片只是把崩溃面收窄到"当前这一小片"。
4. **时间预算**：`--hours` 到点即退，把控制权还给调用者（会话/自动化），下次自动续。
5. **失败不阻塞**：失败的书留在 manifest 里（status=failed），不重试；
   等健康队列清空后再由 `--retry-failed` 统一过一遍。

## 用法

    python -u autopilot.py                    # 跑 3 小时或直到全库完成
    python -u autopilot.py --hours 1 --jobs 6 # 短跑
    python -u autopilot.py --status           # 只看剩余量，不动手
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PY = sys.executable
WORK = Path(r"E:\EPUB\work")
PIPELINE = Path(r"E:\EPUB\pipeline")
BOOKS_WORK = Path(r"E:\EPUB\work\mathbook")
LOCK = WORK / "autopilot.lock"
STATE = WORK / "autopilot_state.json"
LOG = WORK / "autopilot.log"
SLICE_LIST = WORK / "autopilot_slice.txt"
RESERVED = WORK / "mineru_reserved.txt"   # 划给 MinerU 的书，pdf-craft 不碰（避免重复烧额度）
SURVEY = WORK / "survey.json"
MANIFEST = BOOKS_WORK / "manifest.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid: int) -> bool:
    try:
        # 必须 errors="replace"：进程表里若有非 UTF-8 名字（如某些中文/带 0xd0 字节的进程），
        # text=True 会解码抛错 → stdout=None → 后面 `in None` 直接 TypeError 崩掉。
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=15).stdout or ""
    except Exception:  # noqa: BLE001
        return False
    return str(pid) in out


def acquire_lock(stale_min: int) -> bool:
    if LOCK.is_file():
        try:
            info = json.loads(LOCK.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            info = {}
        pid = int(info.get("pid") or 0)
        age = time.time() - float(info.get("ts") or 0)
        if pid and pid_alive(pid) and age < stale_min * 60:
            log(f"[锁] 已有实例在跑（pid={pid}，心跳 {age / 60:.1f} 分钟前）→ 退出")
            return False
        log(f"[锁] 抢占僵尸锁（pid={pid} 存活={pid_alive(pid)}，心跳 {age / 60:.1f} 分钟前）")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}), encoding="utf-8")
    return True


def heartbeat(stop: threading.Event) -> None:
    while not stop.wait(30):
        try:
            LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def load(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def remaining(a, include_failed: bool = False) -> list[dict]:
    """未完成的书，按页数升序（小的先做）。

    ★ `failed` 的书**默认不进"健康队列"**。原因（2026-09-22 踩到死循环）：
      `batch.py` 对上次失败的书是**跳过**的（要 `--retry-failed` 才重做），
      而旧版这里只排除 `done` → 失败书永远留在 `rest` 里 →
      每轮都被选中、每轮都被跳过、`rest` 永不为空 →
      「健康队列清空才转重试」那个分支永远不触发 → **一轮 1 秒地空转烧光预算**
      （实测 10 分钟跑了 2400+ 轮，0 本产出）。
      现在：失败书只在 `include_failed=True` 的重试轮出现。
    """
    survey = load(SURVEY, [])
    manifest = load(MANIFEST, {})
    reserved = set()
    if RESERVED.is_file():
        reserved = {ln.strip() for ln in RESERVED.read_text(encoding="utf-8").splitlines()
                    if ln.strip()}
    out = []
    for b in survey:
        rel, pages = b.get("rel"), int(b.get("pages") or 0)
        if not rel or not pages:
            continue
        if rel in reserved:
            continue
        kind = b.get("kind") or "unknown"
        if a.kind != "all" and kind != a.kind:
            continue
        if a.min_pages and pages < a.min_pages:
            continue
        if a.max_pages and pages > a.max_pages:
            continue
        rec = manifest.get(rel)
        if rec:
            status = rec.get("status")
            if status == "done":
                epub = rec.get("epub") or ""
                if epub and Path(epub).is_file():
                    continue
            elif status == "failed" and not include_failed:
                continue
        out.append({"rel": rel, "pages": pages, "kind": kind})
    out.sort(key=lambda b: b["pages"])
    return out


def run_batch(book_list: Path, jobs: int, retry_failed: bool, budget: float,
              max_pages: int = 0) -> int:
    # ★ 单本超时**绝不能**跟着剩余预算缩。
    #
    # 原实现是 `max(30, budget // 60)`，而 autopilot 在 run_batch 里是 `p.wait()`
    # **阻塞**的 —— deadline 到了也不会中断这一批。于是这个 timeout 唯一的实际
    # 效果，就是让「开在预算尾段的那一轮」拿到一个很小的值：预算只剩 5 分钟时
    # timeout-min = max(30, 5) = 30，于是一本 364 页的书（实测 12~15 秒/页
    # ⇒ 需 ~91 分钟）会在 30 分钟处被**腰斩**。页 XML 会留下、下轮能续跑，
    # 但那一轮的槽位和时间已经白烧了。
    #
    # 正确判据是「这一轮最大的那本书要跑多久」，与预算无关：48 路实测 p50
    # 21.4 秒/页（_bench_scale.py），2026-09-23 21:10 有 20 本大书（351~401p）
    # 因旧常量 16s/页被齐杀（>150 min）—— 页 XML 已落盘，重跑逐页 SKIP。
    # 取 22 秒/页 + 40 分钟余量（覆盖导出 EPUB、重抽轮次和磁盘抖动）。
    per_book_min = max(60, int(max_pages * 22 / 60) + 40)
    cmd = [PY, "-u", str(PIPELINE / "batch.py"), "--only", str(book_list),
           "--jobs", str(jobs), "--sort", "small",
           "--timeout-min", str(per_book_min)]
    if retry_failed:
        cmd.append("--retry-failed")
    log(f"[批] {' '.join(cmd[2:])}  ← 单本超时按最大书 {max_pages}p 定，与剩余预算无关")
    t0 = time.time()
    with LOG.open("a", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=str(PIPELINE), stdout=lf, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        rc = p.wait()
    log(f"[批] 退出码 {rc} · 用时 {(time.time() - t0) / 60:.1f} 分钟")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--slice", type=int, default=8, help="每轮交给 batch.py 的书数")
    ap.add_argument("--hours", type=float, default=3.0, help="本次调用最多跑多久")
    ap.add_argument("--min-pages", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--kind", default="all", choices=["all", "digital", "mixed", "scan"])
    ap.add_argument("--stale-min", type=int, default=10)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    rest = remaining(a)
    total_pages = sum(b["pages"] for b in rest)
    if a.status:
        print(f"剩余 {len(rest)} 本 / {total_pages:,} 页")
        for b in rest[:20]:
            print(f"  {b['pages']:5d}p  {b['kind']:8s}  {b['rel'][:70]}")
        if len(rest) > 20:
            print(f"  … 其余 {len(rest) - 20} 本")
        return 0
    if not rest:
        log("[✓] 队列已空，全部完成")
        return 0

    if not acquire_lock(a.stale_min):
        return 3
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()

    deadline = time.time() + a.hours * 3600
    log(f"[起] pid={os.getpid()} · 剩余 {len(rest)} 本 / {total_pages:,} 页 · "
        f"jobs={a.jobs} slice={a.slice} 预算={a.hours}h")
    try:
        round_no = 0
        retry_done = False
        no_progress = 0
        while time.time() < deadline:
            rest = remaining(a, include_failed=retry_done)
            if not rest:
                if retry_done:
                    log("[✓] 队列已空（含重试），全部完成")
                    break
                retry_done = True
                log("[→] 健康队列已清空，转 --retry-failed 统一过一遍失败的书")
                continue
            round_no += 1
            batch = rest[: a.slice]
            SLICE_LIST.write_text("\n".join(b["rel"] for b in batch) + "\n", encoding="utf-8")
            pages = sum(b["pages"] for b in batch)
            log(f"[轮 {round_no}] 取 {len(batch)} 本 / {pages:,} 页"
                f"（{batch[0]['pages']}~{batch[-1]['pages']}p，剩余 {len(rest)} 本）")
            left = deadline - time.time()
            run_batch(SLICE_LIST, a.jobs, retry_failed=retry_done, budget=left,
                      max_pages=batch[-1]["pages"])
            STATE.write_text(json.dumps({
                "ts": time.time(), "round": round_no, "remaining": len(rest),
                "last_batch": [b["rel"] for b in batch],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            # ★ 无进展守卫：正常一轮要几十到上百分 钟，绝不该出现"秒回且一本没动"。
            #   一旦连着两轮队列长度不降，就停下——宁可少跑，也不能空转烧光预算
            #   （2026-09-22 就是因为没有这道闸，10 分钟空转了 2400+ 轮）。
            after = len(remaining(a, include_failed=retry_done))
            if after >= len(rest):
                no_progress += 1
                log(f"[!] 本轮无进展（{len(rest)} → {after}），连续 {no_progress} 轮")
                if no_progress >= 2:
                    log("[停] 连续两轮无进展，退出（避免空转）")
                    break
            else:
                no_progress = 0
        else:
            log(f"[停] 到达时间预算 {a.hours}h，交还控制权（下次调用自动续跑）")
    finally:
        stop.set()
        try:
            LOCK.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    rest = remaining(a)
    log(f"[尾] 剩余 {len(rest)} 本 / {sum(b['pages'] for b in rest):,} 页")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 12. work/diag_alive.py

```python
#!/usr/bin/env python
"""主线存活诊断：不信任日志 mtime，也不信任进程表（沙盒里枚举不到进程）。

判据（本项目踩过的坑）：
1. `autopilot.lock` 的心跳；
2. **books/ 下真实的写盘**（page_*.xml / assets 的 mtime）—— 这是唯一不会骗人的证据；
3. 两者取较新的那个。

用法：python -u E:\\EPUB\\work\\diag_alive.py [--minutes 30]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(r"E:\EPUB\work\mathbook\books")
WORK = pathlib.Path(r"E:\EPUB\work")


def age_min(ts: float, now: float) -> float:
    return (now - ts) / 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    a = ap.parse_args()
    now = time.time()
    print(f"now = {time.strftime('%Y-%m-%d %H:%M:%S')}")

    lock = WORK / "autopilot.lock"
    if lock.is_file():
        d = json.loads(lock.read_text())
        print(f"lock: pid={d.get('pid')}  age={age_min(d.get('ts', 0), now):.1f} min")
    else:
        print("lock: 无")

    for name in ("autopilot.log", "mineru_all.log", "glm_scan_zai2.log", "sf_qa.log",
                 "vlm_fix_all.log", "maas_scan.log"):
        p = WORK / name
        if p.is_file():
            print(f"log {name:<18} age={age_min(p.stat().st_mtime, now):7.1f} min  "
                  f"{p.stat().st_size} B")

    # 全局：最近被写入的文件（books 下）
    recent: list[tuple[float, str]] = []
    per_book: dict[str, tuple[float, int]] = {}
    scanned = 0
    for p in ROOT.rglob("*"):
        scanned += 1
        try:
            st = p.stat()
        except OSError:
            continue
        if not p.is_file():
            continue
        t = st.st_mtime
        if age_min(t, now) > a.minutes:
            continue
        if p.suffix in (".xml", ".png") or p.name.endswith(".pcex"):
            recent.append((t, str(p.relative_to(ROOT))))
            bk = str(p.relative_to(ROOT)).split("\\")[0]
            old = per_book.get(bk)
            if old is None or t > old[0]:
                per_book[bk] = (t, (old[1] if old else 0) + 1)

    print(f"\n扫描 {scanned} 个条目；{a.minutes:.0f} 分钟内有写入的产出文件 {len(recent)} 个，"
          f"涉及 {len(per_book)} 本书")
    for bk, (t, n) in sorted(per_book.items(), key=lambda x: -x[1][0]):
        print(f"  {age_min(t, now):6.1f} min  {n:5d} 个   {bk[:72]}")
    print("\n最近写入的 12 个文件：")
    for t, rel in sorted(recent, key=lambda x: -x[0])[:12]:
        print(f"  {age_min(t, now):6.1f} min  {rel[:100]}")


if __name__ == "__main__":
    main()
```

### 13. work/scan_math_defects.py

```python
# -*- coding: utf-8 -*-
"""pcex 级公式缺陷扫描：对全部 done 书逐条公式跑缺陷 pattern，产出 fid 级清单。

为什么不用 15:19 的 census 页清单：EPUB 已重渲染，页级 flag 对不上当前产物；
且页级 flag 混着「页上有好公式也有坏公式」。fid 级扫描直接给出要修的每一条。

缺陷类（按杀伤力排序）：
  grid_garbage   矩阵被 OCR 成 | 3 \\ | 0 \\ | 0 竖线网格（视觉垃圾但 KaTeX 能编译）
  empty          latex 为空/纯空白
  img_ref        公式内容其实是图片引用 ![](...) 或 http 链接
  dollar_leak    latex 里混入 $ 定界符
  html_escape    latex 里有 &lt; &gt; &amp; 实体
  cjk_bare       CJK 裸奔在数学里（不在 \\text{}/\\mathrm{} 内）
  replacement    � 替换符
  brace_unbal    非转义花括号不配对（保守：只报不改）

输出 work/math_defect_fids.json:
  [{"book": <epub名>, "key": <manifest key>, "pcex": <路径>, "fid": ..., "kind": ...,
    "page": ..., "bbox": ..., "latex": <现值>, "flags": [...]}...]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"E:\EPUB\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import pcexlib as P  # noqa: E402

MANIFEST = Path(r"E:/EPUB/work/mathbook/manifest.json")
OUT = Path(r"E:/EPUB/work/math_defect_fids.json")

GRID = re.compile(r"(?:\|\s*\S{1,12}\s*\\\\[\s]*){3,}")
IMGREF = re.compile(r"!\[\]\(|https?://", re.I)
DOLLAR = re.compile(r"\$")
ESCAPE = re.compile(r"&(?:lt|gt|amp|quot|#\d+);")
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def cjk_outside_text(latex: str) -> bool:
    """CJK 出现在 \\text{...}/\\mathrm{...}/\\mbox{...} 之外才算缺陷。"""
    s = re.sub(r"\\(?:text|mathrm|mbox|textrm|textbf|operatorname)\s*\{[^{}]*\}", " ", latex)
    s = re.sub(r"\\(?:text|mathrm|mbox|textrm|textbf|operatorname)\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", s)
    return bool(CJK.search(s))


def brace_unbalanced(latex: str) -> bool:
    s = latex.replace(r"\{", "").replace(r"\}", "")
    s = re.sub(r"\\[{}]", "", s)
    return s.count("{") != s.count("}")


def flags_of(latex: str) -> list[str]:
    f: list[str] = []
    if not latex or not latex.strip():
        f.append("empty")
        return f
    if GRID.search(latex):
        f.append("grid_garbage")
    if IMGREF.search(latex):
        f.append("img_ref")
    if DOLLAR.search(latex):
        f.append("dollar_leak")
    if ESCAPE.search(latex):
        f.append("html_escape")
    if cjk_outside_text(latex):
        f.append("cjk_bare")
    if "\ufffd" in latex:
        f.append("replacement")
    if brace_unbalanced(latex):
        f.append("brace_unbal")
    return f


def main() -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    done = [(k, v) for k, v in man.items()
            if v.get("status") == "done" and v.get("pcex") and Path(v["pcex"]).is_file()]
    print(f"done 且 pcex 在: {len(done)} 本")

    out: list[dict] = []
    overflow: list[dict] = []
    t0 = time.time()
    n_books_hit = 0
    for bi, (k, v) in enumerate(done):
        pcex = Path(v["pcex"])
        try:
            with P.Pcex(str(pcex)) as book:
                chs = book.chapters()
                # 页码越界检测：块页码超过 manifest pages ⇒ 旧版流水线陈旧锚（内容无损）
                mp = int(v.get("pages") or 0)
                if mp:
                    mx = 0
                    for ch in chs:
                        for b in ch.blocks:
                            p = getattr(b, "page", None)
                            if p and p > mx:
                                mx = p
                    if mx > mp:
                        overflow.append({"book": k, "manifest_pages": mp, "pcex_max_page": mx})
                for r in P.iter_math(chs):
                    fl = flags_of(r.latex)
                    if not fl:
                        continue
                    obj = r.obj
                    page = getattr(obj, "page", None)
                    bbox = getattr(obj, "bbox", None)
                    if page is None and r.kind == "inline":
                        parent = chs[r.chapter - 1].blocks[r.block]
                        page = getattr(parent, "page", None)
                        bbox = getattr(parent, "bbox", None)
                    out.append({
                        "book": Path(v["epub"]).name if v.get("epub") else k,
                        "key": k, "pcex": str(pcex), "fid": r.fid, "mkind": r.kind,
                        "page": page, "bbox": bbox, "latex": r.latex, "flags": fl,
                    })
        except Exception as e:  # noqa: BLE001
            print(f"[!] pcex 解析失败 {k[:60]}: {type(e).__name__}: {e}")
            continue
        if bi % 20 == 0:
            print(f"  ...{bi + 1}/{len(done)} 本，累计缺陷 {len(out)}  ({time.time() - t0:.0f}s)", flush=True)

    n_books_hit = len({d["key"] for d in out})
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    import collections
    cf = collections.Counter(f for d in out for f in d["flags"])
    ck = collections.Counter(d["mkind"] for d in out)
    print(f"\n完成 {len(done)} 本 / {time.time() - t0:.0f}s")
    print(f"缺陷公式 {len(out)} 条，涉及 {n_books_hit} 本")
    OF = OUT.with_name("math_defect_page_overflow.json")
    OF.write_text(json.dumps(overflow, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"页码越界书 {len(overflow)} 本（陈旧锚，内容无损）→ {OF.name}")
    print("按 flag:", dict(cf.most_common()))
    print("按公式型:", dict(ck))
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 14. work/collect_katex_failures.py

```python
# -*- coding: utf-8 -*-
"""全库 KaTeX 失败收集器：扫全部 done EPUB，产出结构化失败清单。

输出 work/katex_failures.json:
  [{"epub": <路径>, "chapter": <xhtml名>, "cls": "math-inline|math-body",
    "latex": <失败 latex>, "book_key": <manifest key>}...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"E:\EPUB\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import check_math as CM  # noqa: E402

MAN = json.load(open(r"E:/EPUB/work/mathbook/manifest.json", encoding="utf-8"))
OUT = Path(r"E:/EPUB/work/katex_failures.json")

done = [(k, v) for k, v in MAN.items()
        if v.get("status") == "done" and v.get("epub") and Path(v["epub"]).is_file()]
print(f"done EPUB: {len(done)}")

fails: list[dict] = []
total = bad = 0
for i, (k, v) in enumerate(done):
    epub = Path(v["epub"])
    try:
        rows = CM.extract(epub)
    except Exception as e:  # noqa: BLE001
        print(f"[!] extract 失败 {epub.name[:50]}: {e}")
        continue
    if not rows:
        continue
    res = CM.katex_batch([r[2] for r in rows],
                         [r[1] == "math-body" for r in rows])
    n_bad = 0
    for (chap, cls, latex), line in zip(rows, res):
        total += 1
        if line.startswith("FAIL"):
            bad += 1
            n_bad += 1
            fails.append({
                "epub": str(epub), "chapter": chap, "cls": cls,
                "latex": latex, "err": line.split("\t", 1)[1][:160],
                "book_key": k,
            })
    if n_bad:
        print(f"  [{i + 1}/{len(done)}] {epub.name[:52]} 失败 {n_bad}/{len(rows)}", flush=True)
    if i % 40 == 0:
        print(f"  ...{i + 1}/{len(done)} 累计失败 {bad}", flush=True)

OUT.write_text(json.dumps(fails, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n完成：公式 {total} · KaTeX 失败 {bad} · 涉及 EPUB {len({f['epub'] for f in fails})}")
print("->", OUT)
```

### 15. work/map_katex_failures.py

```python
# -*- coding: utf-8 -*-
"""KaTeX 失败 → fid 映射 + 页分组 + 缓存状态盘点。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"E:\EPUB\pipeline")
sys.path.insert(0, r"E:\EPUB\work")
sys.stdout.reconfigure(encoding="utf-8")
import pcexlib as P  # noqa: E402
from glmocr_refine import bhash, CACHE  # noqa: E402

FAILS = json.load(open(r"E:/EPUB/work/katex_failures.json", encoding="utf-8"))
MAN = json.load(open(r"E:/EPUB/work/mathbook/manifest.json", encoding="utf-8"))

by_book: dict[str, list[dict]] = {}
for f in FAILS:
    by_book.setdefault(f["book_key"], []).append(f)

mapped: list[dict] = []
n_nomap = 0
for bk, items in by_book.items():
    v = MAN.get(bk)
    if not v or not v.get("pcex") or not Path(v["pcex"]).is_file():
        n_nomap += len(items)
        continue
    pcex = Path(v["pcex"])
    corr_p = Path(r"E:/EPUB/work/mathbook/corrections") / f"{pcex.stem}.json"
    corr = json.loads(corr_p.read_text(encoding="utf-8")) if corr_p.is_file() else {}
    try:
        with P.Pcex(str(pcex)) as book:
            chs = book.chapters()
            if corr:
                P.apply_corrections(chs, corr)
            for r in P.iter_math(chs):
                pass
            # 建 latex→refs 索引（套用修正后的当前值）
            lat2refs: dict[str, list] = {}
            for r in P.iter_math(chs):
                lat2refs.setdefault(r.latex, []).append(r)
            for it in items:
                lat = it["latex"]
                refs = lat2refs.get(lat)
                if not refs:
                    # 宽松：去空白比较
                    key = re.sub(r"\s+", "", lat) if (re := __import__("re")) else lat
                    for L, rs in lat2refs.items():
                        if re.sub(r"\s+", "", L) == key:
                            refs = rs
                            break
                if not refs:
                    n_nomap += 1
                    continue
                r0 = refs[0]
                page = getattr(r0.obj, "page", None)
                if page is None and r0.kind == "inline":
                    parent = chs[r0.chapter - 1].blocks[r0.block]
                    page = getattr(parent, "page", None)
                mapped.append({
                    "book_key": bk, "pcex": str(pcex), "fid": r0.fid,
                    "kind": r0.kind, "page": page, "latex": it["latex"],
                    "err": it["err"], "n_same": len(refs),
                })
    except Exception as e:  # noqa: BLE001
        print(f"[!] {bk[:50]}: {e}")
        n_nomap += len(items)

out = Path(r"E:/EPUB/work/katex_failures_mapped.json")
out.write_text(json.dumps(mapped, ensure_ascii=False, indent=1), encoding="utf-8")

pages = {(m["pcex"], m["page"]) for m in mapped if m.get("page")}
cached = sum(1 for pc, pg in pages if (CACHE / bhash(pc) / f"p{pg}.json").is_file())
kinds = {}
for m in mapped:
    kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
print(f"失败 {len(FAILS)} → 映射 {len(mapped)}（未映射 {n_nomap}）| 公式型 {kinds}")
print(f"涉及页 {len(pages)}：缓存命中 {cached}，需新抓 {len(pages) - cached}")
print("->", out)
```

### 16. work/glmocr_refine.py

```python
# -*- coding: utf-8 -*-
r"""GLM-OCR 精修引擎（2026-09-23 拍板实现）。

架构决定（拍板人：本会话 AI；授权：主人"待拍板是你做决定"）：
  * **不走 page_N.xml 替换/重 extract**（会重跑 toc/chapters，ID 漂移风险大、
    且 done 书重抽浪费算力），**走 corrections 叠加层**：
    GLM-OCR 重 OCR 缺陷页 → 按 bbox IoU 对齐 pcex 公式 → KaTeX 预校验 →
    合并写 `mathbook/corrections/<stem>.json`（既有条目优先）→
    pcex2x 纯渲染覆写 EPUB（零 manifest 写入，与产线零冲突）。
  * 原始响应全落盘缓存 `work/glmocr_refine/cache/`——后续改进匹配算法零 token。
  * 预算闸门：默认硬顶 30M tok（包的 60%）；429 退避；1113 余额不足 → 全停。

用法：
    python glmocr_refine.py fetch --limit 3      # 试跑 3 页（渲染+调用+缓存）
    python glmocr_refine.py fetch                # 全量 1,110 页
    python glmocr_refine.py patch                # 缓存 → 对齐 → 校验 → corrections
    python glmocr_refine.py apply                # 受影响书重渲染 + 门禁报告
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, r"E:\EPUB\pipeline")
sys.stdout.reconfigure(encoding="utf-8")

import httpx  # noqa: E402

import pcexlib as P  # noqa: E402
from check_math import katex_batch  # noqa: E402  # 复用产线同款 KaTeX 门禁

# ── 常量 ────────────────────────────────────────────────────────────────────
KEY = Path(r"C:\Users\hp\.workbuddy\secrets\glmocr.key").read_text(encoding="ascii").strip()
URL = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
PDF_ROOT = Path(r"E:\mathbook")
POPPLER = Path(r"E:\EPUB\tools\poppler\Library\bin\pdftoppm.exe")
MATHBOOK = Path(r"E:\EPUB\work\mathbook")
OUTROOT = Path(r"E:\EPUB\out\mathbook")
DEFECTS = Path(r"E:\EPUB\work\math_defect_fids.json")
WORKROOT = Path(r"E:\EPUB\work\glmocr_refine")
CACHE = WORKROOT / "cache"
LEDGER = WORKROOT / "ledger.json"

CONC = 8
BUDGET_CAP = 30_000_000
CALL_TIMEOUT = 240.0
BACKOFFS = [3, 6, 12, 24, 48]

SEV = {"grid_garbage": 6, "empty": 6, "img_ref": 5, "replacement": 5,
       "html_escape": 5, "dollar_leak": 3, "cjk_bare": 2, "katex_fail": 5}
STRONG = {"grid_garbage", "empty", "img_ref", "replacement", "html_escape", "dollar_leak", "katex_fail"}
FAILMAP = Path(r"E:/EPUB/work/katex_failures_mapped.json")

_lock = threading.Lock()


def ledger_load() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"tokens": 0, "calls": 0, "ok": 0, "e429": 0, "fail": 0, "aborted": ""}


def ledger_add(d: dict, **kw) -> None:
    with _lock:
        d.update({k: d.get(k, 0) + v for k, v in kw.items() if isinstance(v, (int, float))})
        LEDGER.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def bhash(pcex: str) -> str:
    import hashlib
    return hashlib.sha1(pcex.encode("utf-8")).hexdigest()[:12]


# ── 缺陷再判（与 scan_math_defects.py 同一套 pattern）──────────────────────
GRID = re.compile(r"(?:\|\s*\S{1,12}\s*\\\\[\s]*){3,}")
IMGREF = re.compile(r"!\[\]\(|https?://", re.I)
DOLLAR = re.compile(r"\$")
ESCAPE = re.compile(r"&(?:lt|gt|amp|quot|#\d+);")
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def cjk_outside_text(latex: str) -> bool:
    s = re.sub(r"\\(?:text|mathrm|mbox|textrm|textbf|operatorname)\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", latex)
    return bool(CJK.search(s))


def flags_of(latex: str) -> list[str]:
    f: list[str] = []
    if not latex or not latex.strip():
        return ["empty"]
    if GRID.search(latex):
        f.append("grid_garbage")
    if IMGREF.search(latex):
        f.append("img_ref")
    if DOLLAR.search(latex):
        f.append("dollar_leak")
    if ESCAPE.search(latex):
        f.append("html_escape")
    if cjk_outside_text(latex):
        f.append("cjk_bare")
    if "\ufffd" in latex:
        f.append("replacement")
    return f


def is_pure_prose(latex: str) -> bool:
    """公式块其实是纯中文散文（'所以'、'[利用…得出证明…]'）→ 应变 \\text{}。"""
    s = re.sub(r"\\(?:mathbf|mathrm|text|textrm|boldsymbol)\s*\{[^{}]*\}", " ", latex)
    if len(CJK.findall(s)) < 2:
        return False
    core = CJK.sub(" ", s)
    core = re.sub(r"[\s\[\]()，。：；、！？“”‘’《》\.,:;!?'\"]", "", core)
    return not re.search(r"[0-9a-zA-Z+\-=^_{}|/<>]", core)


# ── bbox 工具 ───────────────────────────────────────────────────────────────
def normalize_candidate(s: str) -> str:
    """GLM 候选规范化：LaTeX 控制序列只能是 ASCII，反斜杠后跟非 ASCII 字符
    （\升、\§ 等）必是模型错误转义 ⇒ 剥掉反斜杠（KaTeX strict:false 可渲染 Unicode）。"""
    return re.sub(r"\\(?=[^\x00-\x7f])", "", s)


def pbbox(s: str | None):
    if not s:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in s.split(","))
        return (x1, y1, x2, y2)
    except Exception:  # noqa: BLE001
        return None


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ── fetch ───────────────────────────────────────────────────────────────────
def targets(from_failures: bool = False) -> list[dict]:
    if from_failures:
        mp = json.loads(FAILMAP.read_text(encoding="utf-8"))
        by_page: dict[tuple, dict] = {}
        for it in mp:
            pg = it.get("page")
            if not pg:
                continue
            key = (it["pcex"], int(pg))
            t = by_page.setdefault(key, {"pcex": it["pcex"], "page": int(pg),
                                         "book": it["book_key"], "sev": 5, "n": 0})
            t["n"] += 1
        return sorted(by_page.values(), key=lambda t: (-t["n"],))
    defects = json.loads(DEFECTS.read_text(encoding="utf-8"))
    by_page: dict[tuple, dict] = {}
    for it in defects:
        pg = it.get("page")
        if not pg:
            continue
        key = (it["pcex"], int(pg))
        t = by_page.setdefault(key, {"pcex": it["pcex"], "page": int(pg),
                                     "book": it["book"], "sev": 0, "n": 0})
        t["sev"] = max(t["sev"], max((SEV.get(f, 1) for f in it["flags"]), default=1))
        t["n"] += 1
    out = sorted(by_page.values(), key=lambda t: (-t["sev"], -t["n"]))
    return out


def render_page(pdf: Path, page: int, png_out: Path) -> Path:
    png_out.mkdir(parents=True, exist_ok=True)
    for old in png_out.glob("pg*"):
        old.unlink(missing_ok=True)
    r = subprocess.run([str(POPPLER), "-png", "-r", "300", "-f", str(page), "-l", str(page),
                        str(pdf), str(png_out / "pg")],
                       capture_output=True, timeout=120)
    got = list(png_out.glob("pg*.png"))
    if r.returncode != 0 or not got:
        raise RuntimeError(f"pdftoppm rc={r.returncode} {r.stderr[:120]!r}")
    return got[0]


def call_glm(png: Path, ledger: dict) -> dict:
    b64 = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    body = {"model": "glm-ocr", "file": b64}
    headers = {"Authorization": f"Bearer {KEY}"}
    last = ""
    for attempt, wait in enumerate([0] + BACKOFFS):
        if wait:
            time.sleep(wait + random.random() * 2)
        try:
            with httpx.Client(trust_env=False, timeout=CALL_TIMEOUT) as c:
                r = c.post(URL, json=body, headers=headers)
            if r.status_code == 200:
                j = r.json()
                ledger_add(ledger, ok=1, calls=1,
                           tokens=int((j.get("usage") or {}).get("total_tokens") or 0))
                return j
            last = f"HTTP {r.status_code}: {r.text[:150]}"
            ledger_add(ledger, calls=1)
            if r.status_code == 429 or "1302" in r.text[:200]:
                ledger_add(ledger, e429=1)
                continue
            if "1113" in r.text[:200]:
                ledger_add(ledger, fail=1)
                with _lock:
                    ledger["aborted"] = "余额不足(1113)"
                    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
                raise SystemExit("GLM-OCR 余额不足(1113)——全停")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:150]
            ledger_add(ledger, calls=1)
    raise RuntimeError(f"重试耗尽: {last}")


def phase_fetch(a) -> int:
    ledger = ledger_load()
    if ledger.get("aborted"):
        print(f"[!] ledger 带中止标记: {ledger['aborted']}（删 ledger.json 可重置）")
        return 1
    ts = targets(bool(getattr(a, "from_failures", False)))
    print(f"目标 {len(ts)} 页；当前已耗 {ledger['tokens']:,} tok，硬顶 {BUDGET_CAP:,}")
    todo = []
    n_cached = 0
    for t in ts:
        cdir = CACHE / bhash(t["pcex"])
        if (cdir / f"p{t['page']}.json").is_file():
            n_cached += 1
            continue
        todo.append(t)
    n_before_limit = len(todo)
    if a.limit:
        todo = todo[:a.limit]
    if a.book:
        todo = [t for t in todo if a.book in t["book"]]
    print(f"缓存命中 {n_cached}，待取 {n_before_limit}（本轮限 {len(todo)}）")

    def work(t):
        if ledger["tokens"] >= BUDGET_CAP:
            return ("budget", t, "")
        pdf = PDF_ROOT / MAN[t["pcex"]]
        if not pdf.is_file():
            return ("no_pdf", t, str(pdf))
        png_dir = WORKROOT / "png" / bhash(t["pcex"]) / f"p{t['page']}"  # 每页独立子目录：同书并发不互删
        try:
            png = render_page(pdf, t["page"], png_dir)
        except Exception as e:  # noqa: BLE001
            return ("render", t, str(e)[:120])
        try:
            j = call_glm(png, ledger)
            cdir = CACHE / bhash(t["pcex"])  # 局部计算，别用外层循环残留的 cdir
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / f"p{t['page']}.json").write_text(
                json.dumps(j, ensure_ascii=False), encoding="utf-8")
            return ("ok", t, "")
        finally:
            png.unlink(missing_ok=True)

    MAN = {v.get("pcex"): k for k, v in
           json.loads((MATHBOOK / "manifest.json").read_text(encoding="utf-8")).items()}
    stats: dict[str, int] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as pool:
        futs = [pool.submit(work, t) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            st, t, msg = f.result()
            stats[st] = stats.get(st, 0) + 1
            if st not in ("ok",):
                print(f"  [{st}] {t['book'][:40]} p{t['page']} {msg}", flush=True)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} · tok={ledger['tokens']:,} · "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
            if ledger["tokens"] >= BUDGET_CAP:
                print("[!] 预算硬顶触及，停止提交新页")
                break
    print(f"fetch 完成 {stats} · 总 tok={ledger['tokens']:,}")
    return 0 if stats.get("ok", 0) + len(todo) == 0 or stats.get("ok", 0) else 1


# ── patch ───────────────────────────────────────────────────────────────────
def strip_dollars(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\\\[(.*)\\\]$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"^\$\$(.*)\$\$$", r"\1", s, flags=re.DOTALL)
    return s.strip()


INLINE_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)


def phase_patch(a) -> int:
    if getattr(a, "from_failures", False):
        mp = json.loads(FAILMAP.read_text(encoding="utf-8"))
        defects = [{"book": it["book_key"], "key": it["book_key"], "pcex": it["pcex"],
                    "fid": it["fid"], "mkind": it["kind"], "page": it.get("page"),
                    "bbox": None, "latex": it["latex"], "flags": ["katex_fail"]}
                   for it in mp]
    else:
        defects = json.loads(DEFECTS.read_text(encoding="utf-8"))
    by_book: dict[str, list[dict]] = {}
    for it in defects:
        by_book.setdefault(it["pcex"], []).append(it)

    n_cand = n_written = n_skip_iou = n_skip_katex = 0
    report: list[str] = []
    for pcex_s, items in sorted(by_book.items()):
        pcex = Path(pcex_s)
        stem = pcex.stem
        corr_path = MATHBOOK / "corrections" / f"{stem}.json"
        existing = json.loads(corr_path.read_text(encoding="utf-8")) if corr_path.is_file() else {}
        need_pages = {int(it["page"]) for it in items if it.get("page")}
        cached = {pg for pg in need_pages if (CACHE / bhash(pcex_s) / f"p{pg}.json").is_file()}
        if not cached:
            continue
        try:
            with P.Pcex(str(pcex)) as book:
                chs = book.chapters()
                refs = P.iter_math(chs)
                by_fid = {r.fid: r for r in refs}
                # 每页的 GLM blocks
                new_corr: dict[str, dict] = {}
                cand_variants: dict[str, list[str]] = {}
                for pg in sorted(cached):
                    raw = json.loads((CACHE / bhash(pcex_s) / f"p{pg}.json").read_text(encoding="utf-8"))
                    det = raw.get("layout_details") or []
                    flat = det[0] if (det and isinstance(det[0], list)) else det
                    fblocks = [(b.get("bbox_2d"), strip_dollars(b.get("content") or ""))
                               for b in flat if b.get("label") == "formula" and b.get("bbox_2d")]
                    tblocks = [(b.get("bbox_2d"), (b.get("content") or ""))
                               for b in flat if b.get("label") == "text" and b.get("bbox_2d")]
                    page_items = [it for it in items if int(it.get("page") or 0) == pg]
                    for it in page_items:
                        fid = it["fid"]
                        # 失败通道：fid 已有修正但正死于门禁 ⇒ 必须允许覆盖（merge 时 keep 后写胜出）
                        if fid in existing and "katex_fail" not in (it.get("flags") or []):
                            continue
                        if fid in new_corr:
                            continue
                        r = by_fid.get(fid)
                        if r is None:
                            continue
                        cur = r.latex
                        fl = set(it["flags"]) or set(flags_of(cur))
                        bb = pbbox(it.get("bbox"))
                        if bb is None and r.kind == "display":
                            bb = pbbox(getattr(r.obj, "bbox", None))
                        variants: list[str] = []
                        if r.kind == "display":
                            if is_pure_prose(cur) and not fblocks:
                                # 纯散文块：找同位 text 块包 \text{}
                                best, bc = 0.0, ""
                                for tb, tc in tblocks:
                                    v = iou(bb, tb)
                                    if v > best:
                                        best, bc = v, tc
                                txt = re.sub(r"<[^>]+>", "", bc).strip() or cur.strip()
                                if best > 0.15 or not tblocks:
                                    variants = [r"\text{" + txt + "}"]
                            else:
                                # 主策略：中心点落在 pcex 块内的 GLM 公式块（允许多个——
                                # pcex 常把多行公式合并一块而 GLM 拆开），拼接成变体组
                                contained: list[str] = []
                                if bb:
                                    pad = 12
                                    for fbb, fc in fblocks:
                                        cx = (fbb[0] + fbb[2]) / 2
                                        cy = (fbb[1] + fbb[3]) / 2
                                        if bb[0] - pad <= cx <= bb[2] + pad and bb[1] - pad <= cy <= bb[3] + pad:
                                            contained.append(strip_dollars(fc))
                                if len(contained) == 1:
                                    variants = [contained[0]]
                                elif len(contained) > 1:
                                    variants = ["\n".join(contained), " \\\\ ".join(contained),
                                                "".join(contained)] + contained
                                if not variants:
                                    # 兕底：最佳 IoU ≥ 0.30 的单个块
                                    best, bv = None, 0.30
                                    for fbb, fc in fblocks:
                                        v = iou(bb, fbb)
                                        if v > bv:
                                            best, bv = strip_dollars(fc), v
                                    if best:
                                        variants = [best]
                        else:  # inline：强缺陷才修，且要求 $ 序长度对齐
                            if not (fl & STRONG):
                                continue
                            parent = chs[r.chapter - 1].blocks[r.block]
                            pbb = pbbox(getattr(parent, "bbox", None))
                            best, bv = None, 0.25
                            for tb, tc in tblocks:
                                v = iou(pbb, tb)
                                if v > bv:
                                    best, bv = tc, v
                            if not best:
                                n_skip_iou += 1
                                continue
                            seq = INLINE_RE.findall(best)
                            runs = [x for x in parent.runs if isinstance(x, P.MathRun)]
                            cand = None
                            if len(seq) == len(runs) and runs:
                                idx = next((i for i, x in enumerate(runs) if x is r.obj), -1)
                                if idx >= 0:
                                    cand = seq[idx].strip()
                            elif len(runs) == 1 and len(seq) == 1:
                                # v2 宽松：单公式段 + 唯一 $ 项（无序信息也不歧义）
                                cand = seq[0].strip()
                            if not cand:
                                n_skip_iou += 1
                                continue
                            variants = [cand]
                        variants = [normalize_candidate(v) for v in variants if v and v.strip()]
                        if not variants:
                            n_skip_iou += 1
                            continue
                        n_cand += 1
                        cand_variants[fid] = variants
                        new_corr[fid] = {"latex": variants[0], "original": cur, "source": "glmocr"}
        except Exception as e:  # noqa: BLE001
            report.append(f"[!] {stem[:50]} pcex 解析失败 {type(e).__name__}: {e}"[:140])
            continue
        if not cand_variants:
            continue
        # KaTeX 批量预校验：多变体取首个通过的
        flat_items: list[tuple[str, str]] = []
        disp_flags: list[bool] = []
        for f, vs in cand_variants.items():
            for v in vs:
                flat_items.append((f, v))
                disp_flags.append("-m" in f)  # cNN-mNNN=display，cNN-iNNN=inline（不能混：align 只能在 display 模式编译）
        try:
            res = katex_batch([v for _, v in flat_items], display=disp_flags)
        except Exception as e:  # noqa: BLE001
            report.append(f"[!] {stem[:50]} KaTeX 门禁异常 {e}"[:120])
            continue
        first_ok: dict[str, str] = {}
        for (f, v), rr in zip(flat_items, res):
            if rr.startswith("OK") and f not in first_ok:
                first_ok[f] = v
        keep: dict[str, dict] = {}
        for f, vs in cand_variants.items():
            if f in first_ok:
                keep[f] = {"latex": first_ok[f], "original": new_corr[f]["original"],
                           "source": "glmocr"}
            else:
                n_skip_katex += 1
                report.append(f"  [katex] {stem[:36]} {f}: 全部变体失败（{len(vs)} 个）")
        if not keep:
            continue
        merged = dict(existing)
        merged.update(keep)  # existing 键早已排除，不会覆盖人工成果
        corr_path.parent.mkdir(parents=True, exist_ok=True)
        corr_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
        n_written += len(keep)
        report.append(f"[+] {stem[:56]} 写入 {len(keep)}/{len(new_corr)} 条（累计 {len(merged)}）")

    print(f"\npatch 完成：候选 {n_cand} · 写入 {n_written} · IoU/结构跳过 {n_skip_iou} · KaTeX 拒绝 {n_skip_katex}")
    for line in report[:60]:
        print(line)
    return 0


# ── apply ───────────────────────────────────────────────────────────────────
def phase_apply(a) -> int:
    PY = sys.executable
    corr_dir = MATHBOOK / "corrections"
    man = json.loads((MATHBOOK / "manifest.json").read_text(encoding="utf-8"))
    rel_by_stem = {}
    for k, v in man.items():
        pc = v.get("pcex") or ""
        if pc and Path(pc).is_file():
            rel_by_stem[Path(pc).stem] = (pc, v.get("rel") or k, v.get("epub") or "")
    files = sorted(corr_dir.glob("*.json"))
    done = 0
    for cf in files:
        stem = cf.stem
        hit = rel_by_stem.get(stem)
        if not hit:
            print(f"[skip] manifest 无此 pcex: {stem[:60]}")
            continue
        pcex_s, rel, epub = hit
        pcex = Path(pcex_s)
        if a.book and a.book not in stem:
            continue
        outdir = Path(epub).parent if epub else (OUTROOT / Path(rel).parent)
        outdir.mkdir(parents=True, exist_ok=True)
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", stem) else "en"
        cmd = [PY, r"E:\EPUB\pipeline\pcex2x.py", str(pcex),
               "--outdir", str(outdir), "--formats", "epub", "--stem", stem,
               "--language", lang, "--trace",
               "--katex", r"E:\EPUB\pipeline\assets\katex",
               "--corrections", str(cf)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
        ok = r.returncode == 0
        done += 1
        m = re.search(r"套用修正 (\d+)", r.stdout or "")
        print(f"[{'OK' if ok else 'FAIL'}] {stem[:56]}  applied: {m.group(1) if m else '?'}")
        if not ok:
            print("   stderr:", (r.stderr or "")[-300:])
    print(f"\napply 完成 {done} 本（重渲染即覆写 EPUB；corrections 已被 batch.py 约定目录自动拾取）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fetch", "patch", "apply"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--book", default=None)
    ap.add_argument("--from-failures", action="store_true",
                    help="数据源改用 katex_failures_mapped.json（门禁失败行内公式修复通道）")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    WORKROOT.mkdir(parents=True, exist_ok=True)
    return {"fetch": phase_fetch, "patch": phase_patch, "apply": phase_apply}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
```

