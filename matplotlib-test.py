import matplotlib
# 无GUI必须加这行
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = ['Noto Sans Color Emoji']
plt.rcParams['font.family'] = [' Noto Color Emoji']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, '🚀 服务后台运行 ✅', fontsize=22, ha='center')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis('off')

# 保存图片，不用弹窗
# plt.savefig('emoji_plot.png', dpi=150, bbox_inches='tight')
plt.savefig("./out_images/test_emoji.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图片已保存为 emoji_plot.png")



