from pyqpanda import *
import matplotlib.pyplot as plt
if __name__=="__main__":

    # 同样可以申请云计算机器（采用真实芯片）
    qvm =  QCloud()
    qvm.set_configure(72,72)
    qvm.init_qvm("302e020100301006072a8648ce3d020106052b8104001c041730150201010410d0513887336bab9e1fdc4a2376746e8c/11028")
    qvm.set_qcloud_api("https://qcloud4test.originqc.com")

    qv = qvm.qAlloc_many(72)
    # qubits = [qv[45],qv[46],qv[52],qv[53],qv[54],qv[58]]
    qubits = [qv[46]]

    # 设置随机线路中clifford门集数量
    range = [ 5,10,15,20,25,30,35,40,45,50 ]
    # range = [ 5,10,15,20]

    result_list = []
    for qubit in qubits:
        #res = double_qubit_rb(qvm, qv[0], qv[1], range, 10, 1000)
        res = single_qubit_rb(qvm, qubit, range, 1, 1000, 72)
        print(res)
        result_list.append(res)
    
    qvm.finalize()


# 创建图表和子图
fig, ax = plt.subplots()

# 遍历数据列表，为每个数据字典绘制折线图
for i, data in enumerate(result_list):
    # 提取横坐标和纵坐标数据
    x_values = list(data.keys())
    y_values = list(data.values())

    # 绘制折线图
    ax.plot(x_values, y_values, 'o-', label=f'RB sinle qubit {i+1}')

# 设置横轴和纵轴标签
ax.set_xlabel('clifford_list_num')
ax.set_ylabel('fidelity')

# 设置图表标题
ax.set_title('RB Line Chart')

# 显示图例
ax.legend()

# 显示图表
plt.show()