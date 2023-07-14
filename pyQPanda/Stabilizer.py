import matplotlib.pyplot as plt
 
x = [5,10,15,20,25,30]
y = [27,36,52,64,90,101]

plt.xlabel('depth')  # x轴标题
plt.ylabel('Time(s)')  # y轴标题


plt.title("1000 qubits random clifford circuit with 1 million shots")
plt.plot(x, y, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小

# 设置横纵坐标
plt.xticks(x)

# plt.yticks([0.84,0.86,0.88,0.90])
plt.yticks([20,40,60,80,100,120])
 
# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y3):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y4):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 
# plt.legend(['$\lambda_{2}$ = 0.2', '$\lambda_{2}$ = 0.4', '$\lambda_{2}$ = 0.6', '$\lambda_{2}$ = 0.8'], loc="lower right")  # 设置折线名称
 
plt.show()  # 显示折线图
# plt.savefig('./figure2/line_lamda_shs148k.jpg', dpi=200)