import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 1. 配置参数 --------------------------
excel_path = r"E:/Work/工作簿1.xlsx"
sheet_name = "Sheet1"
cancer_col = "癌种"
case_col = "全球女性新发病例（万）"

# -------------------------- 2. 读取+清洗数据 --------------------------
df = pd.read_excel(excel_path, sheet_name=sheet_name)
# 强制转数字+删除无效行
df[case_col] = pd.to_numeric(df[case_col], errors='coerce')
df = df.dropna(subset=[case_col])
# 降序排序（大数值在前，和原图一致）
df = df.sort_values(by=case_col, ascending=False).reset_index(drop=True)

# 提取核心数据
cancers = df[cancer_col].tolist()
cases = df[case_col].tolist()
total_cases = df[case_col].sum()
n = len(cancers)
max_case = max(cases)

bar_width = 2 * np.pi / n  # 所有条形角度宽度一致
bar_widths = np.full(n, bar_width)
# 计算每个条形的起始角度、中心角度
center_thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
start_thetas = center_thetas - bar_width / 2

inner_radius = max_case * 0.2

# -------------------------- 5. 绘图全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(111, polar=True)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_ylim(0, max_case * 1.3)  # 上限设为最大值的1.3倍，预留文字空间

custom_cmap = LinearSegmentedColormap.from_list(
    'cancer_cmap',
    ['#7A208F', '#B83A8F', '#D05A9F', '#E0789F', '#F09090', '#F8A080', '#FFB060', '#60B0B0', '#50A0A0', '#409090'],
    N=n
)
bar_colors = custom_cmap(np.linspace(0, 1, n))

# -------------------------- 7. 绘制带空心的等宽极坐标条形图 --------------------------
bars = ax.bar(
    x=center_thetas,       # 每个条形的中心角度
    height=cases,           # 条形的长度（数值大小，和参考图一致）
    width=bar_widths,       # 等宽条形
    bottom=inner_radius,    # 从内半径开始，实现中间空心
    color=bar_colors,
    edgecolor='white',      # 条形间白色分隔
    linewidth=0.8
)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['polar'].set_visible(False)
ax.grid(False)

for idx, (cancer_name, case_num, center_theta) in enumerate(zip(cancers, cases, center_thetas)):
    text_y = inner_radius + case_num*0.8
    theta_deg = np.rad2deg(center_theta)
    if idx >= 3 and idx < 5:
        theta_deg = theta_deg - 90
        text_y = inner_radius + case_num*0.5
    if idx == 5:
        theta_deg = theta_deg - 180
        text_y = inner_radius + case_num*1.25
    if idx >= 6:
        theta_deg = theta_deg + 90
        if idx == 6:
            text_y = inner_radius + case_num*1.25
        if idx == 7:
            text_y = inner_radius + case_num*1.4
        if idx == 8:
            text_y = inner_radius + case_num*1.35
    if idx == 9:
        text_y = inner_radius + case_num*2.25
    # 动态字号：数值越大，字号越大，和原图一致
    font_size = 8 + (case_num / max_case) * 16
    # 添加文本（癌种在上、数值在下，和参考图排版一致）
    ax.text(
        x=center_theta,
        y=text_y,
        s=f'{cancer_name}\n{case_num}',
        ha='center',
        va='center',
        rotation=-theta_deg,  
        rotation_mode='anchor',  # 固定旋转锚点，避免文字偏移
        fontsize=font_size,
        fontweight='bold'
    )

# -------------------------- 10. 中心文本（正好在空心圆中间） --------------------------
ax.text(
    x=0, y=0,
    s=f'全球女性新发病例\n{total_cases:.2f}万人',
    ha='center', va='center',
    linespacing=1.5,
    fontsize=12,
    fontweight='bold'
)

# -------------------------- 12. 布局调整 --------------------------
plt.tight_layout()
# 显示+保存高清图
plt.savefig('癌症发病数玫瑰图.png', dpi=300, bbox_inches='tight')
plt.show()