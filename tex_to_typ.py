import re

# ========== 定义核心正则模式 ==========
# 模式1-1：匹配\(任意内容\)（支持多行），单独区分行内公式
pattern_inline = r'\\\((.*?)\\\)'
# 模式1-2：匹配\[任意内容\]（支持多行），单独区分多行公式
pattern_multiline = r'\\\[(.+?)\\\]'
# 模式2：匹配\任意字母/数字{任意内容}（支持多行），如\adgsrg{sddfagr}、\jhrfg{sgr}等
pattern_remove = r'\\\w+\{.*?\}'

# 读取1.txt的内容，保留所有原始格式（包括换行、空格）
try:
    with open('1.txt', 'r', encoding='utf-8') as f:
        original_content = f.read()
except FileNotFoundError:
    print("错误：未找到1.txt文件，请确认文件存在且路径正确！")
    exit(1)

# ========== 第一步：提取所有括号内内容到2.txt（核心修改：换行换空格） ==========
# 合并两种括号模式，提取所有内容
pattern_bracket_all = r'\\\((.*?)\\\)|\\\[(.+?)\\\]'
matches = re.findall(pattern_bracket_all, original_content, flags=re.DOTALL | re.MULTILINE)
# 整理提取的内容：过滤空值，将内部换行替换为空格
extracted_content = []
for match in matches:
    # 提取非空的匹配内容
    content = match[0].strip() if match[0] else match[1].strip()
    if content:  # 排除空内容的情况
        # 核心修改：将内容内部的所有换行符替换为单个空格
        content = content.replace('\n', ' ')
        # 可选：将多个连续空格合并为一个（避免换行替换后多个空格）
        content = re.sub(r'\s+', ' ', content).strip()
        extracted_content.append(content)
# 写入2.txt，不同内容之间用回车键分隔
with open('2.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(extracted_content))

# ========== 第二步：生成3.txt（保留in_line/multi_line标记逻辑） ==========
# 步骤2.1：先将\(...)替换为in_line（行内公式标记）
temp_content = re.sub(pattern_inline, 'in_line', original_content, flags=re.DOTALL | re.MULTILINE)
# 步骤2.2：再将\[...)替换为multi_line（多行公式标记）
temp_content = re.sub(pattern_multiline, 'multi_line', temp_content, flags=re.DOTALL | re.MULTILINE)
# 步骤2.3：删除所有\任意字符{任意内容}格式的内容
final_3_content = re.sub(pattern_remove, '', temp_content, flags=re.DOTALL | re.MULTILINE)
# 写入最终的3.txt
with open('3.txt', 'w', encoding='utf-8') as f:
    f.write(final_3_content)

# ========== 输出处理结果 ==========
print("处理完成！")
print(f"共提取到 {len(extracted_content)} 段内容，已保存到2.txt")
print("2.txt格式说明：")
print("  - 同一括号内的多行内容已将换行替换为空格")
print("  - 不同括号内容之间用回车键分隔")
print("3.txt已生成：")
print("  - \(...\)位置标记为in_line，\[...\]位置标记为multi_line")
print("  - 所有\\xxx{xxx}格式的内容已删除")




import re

# ========== 核心配置 ==========
# 定义需要匹配的标记正则（匹配in_line或multi_line，不匹配其他包含这些字符的内容）
pattern_marker = r'\b(in_line|multi_line)\b'

# ========== 步骤1：读取4.txt的内容（按行读取，匹配标记顺序） ==========
try:
    with open('4.txt', 'r', encoding='utf-8') as f:
        # 读取所有行，去除每行首尾的换行/空格，过滤空行
        lines_4 = [line.strip() for line in f.readlines() if line.strip()]
    # 打印4.txt的有效行数，方便校验
    print(f"4.txt中读取到 {len(lines_4)} 行有效内容")
except FileNotFoundError:
    print("错误：未找到4.txt文件，请确认文件存在且路径正确！")
    exit(1)

# ========== 步骤2：读取3.txt的内容 ==========
try:
    with open('3.txt', 'r', encoding='utf-8') as f:
        content_3 = f.read()
except FileNotFoundError:
    print("错误：未找到3.txt文件，请确认文件存在且路径正确！")
    exit(1)

# ========== 步骤3：统计3.txt中的标记数量并校验 ==========
# 找到所有in_line/multi_line标记
markers = re.findall(pattern_marker, content_3)
marker_count = len(markers)
print(f"3.txt中找到 {marker_count} 个标记（in_line+multi_line）")

# 校验标记数量和4.txt行数是否一致
if marker_count != len(lines_4):
    print(f"错误：标记数量({marker_count})与4.txt行数({len(lines_4)})不匹配！")
    print("请检查4.txt的行数是否为267行，或3.txt的标记数量是否正确。")
    exit(1)

# ========== 步骤4：按顺序替换标记 ==========
# 创建迭代器，按顺序取4.txt的行内容
line_4_iter = iter(lines_4)
# 定义替换函数：根据标记类型返回对应格式的内容
def replace_marker(match):
    marker_type = match.group(1)
    # 取4.txt的下一行内容
    line_content = next(line_4_iter)
    if marker_type == 'in_line':
        # in_line替换为 $内容$
        return f"${line_content}$"
    elif marker_type == 'multi_line':
        # multi_line替换为 $\n内容\n$（\n为换行符）
        return f"$\n{line_content}\n$"

# 执行替换（按标记出现顺序替换）
content_5 = re.sub(pattern_marker, replace_marker, content_3)

# ========== 步骤5：保存替换后的内容到5.txt ==========
with open('5.txt', 'w', encoding='utf-8') as f:
    f.write(content_5)

# ========== 输出结果 ==========
print("替换完成！")
print(f"共替换 {marker_count} 个标记，替换后的内容已保存到5.txt")
