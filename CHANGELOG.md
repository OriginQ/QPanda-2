# Changelog

所有针对QPanda的修改都会记录在本文件中.

> **修改的类型:**
>
> -   **Added**: 添加新的功能.
> -   **Changed**: 对现有的功能进行修改.
> -   **Deprecated**: 即将删除的功能.
> -   **Removed**: 删除的功能.
> -   **Fixed**: Bug修复.

[UNRELEASED](https://github.com/OriginQ/QPanda-2/compare/v2.1.7...HEAD)
========================================================================

[v2.1.7](https://github.com/OriginQ/QPanda-2/compare/v2.1.5...v2.1.7) - 2020-7-27
==================================================================================

QPanda
-----------

Added
*********

- 虚拟机添加同时对多种量子逻辑门噪声设置接口


Changed
*********

- 修改control信息遍历方式
- 修改PMeasure算法
- 修改GTest测试框架
- Grover测试用例整改


pyqpanda
-------------

Added
*********

- 虚拟机添加同时对多种量子逻辑门噪声设置接口


Changed
*********

- 修改control信息遍历方式
- 修改PMeasure算法
- OBMT_mapping映射算法接口优化
- HHL应用程序优化


[v2.1.6](https://github.com/OriginQ/QPanda-2/compare/v2.1.5...v2.1.6) - 2020-6-17
==================================================================================

QPanda
-----------

Added
*********

- 单振幅虚拟机添加概率测量类接口
- 部分振幅虚拟机添加概率测量类接口
- 增加模式匹配的并行化功能
- 添加直接通过矩阵构造U3门方法
- 添加Toffli门支持

Changed
*********

- 单振幅虚拟机算法增加路径优化功能
- 拓展Var支持逻辑门
- 修改线路dagger()和control()函数的内部实现
- 在量子程序或量子线路中插入节点时修改为深拷贝的方式
- bmt及sabre映射算法插入swap门时，直接转换成U3+cz门
- 独立单门优化方法
- 全振幅虚拟机算法优化
- mingw支持大文件debug编译
- 量子比特和经典寄存器与虚拟机解耦

pyqpanda
-------------

Added
*********

- 单振幅虚拟机添加概率测量类接口
- 部分振幅虚拟机添加概率测量类接口
- 增加模式匹配的并行化功能
- 添加直接通过矩阵构造U3门方法
- bmt及sabre映射算法插入swap门时，直接转换成U3+cz门
- 添加OBMT_mapping映射算法Python接口


Changed
*********

- 单振幅虚拟机算法增加路径优化功能
- 拓展Var支持逻辑门
- 修改线路dagger()和control()函数的内部实现
- 在量子程序或量子线路中插入节点时修改为深拷贝的方式
- 独立单门优化方法
- 全振幅虚拟机算法优化
- 量子比特和经典寄存器与虚拟机解耦

[v2.1.5](https://github.com/OriginQ/QPanda-2/compare/v2.1.4...v2.1.5) - 2020-2-25
==================================================================================

QPanda
-----------

Added
*********

- 添加XEB实验功能
- ClassicalCondition添加获取地址信息接口
- 添加用户自定义逻辑门类型的随机线路生成接口
- 获取QProg和QCircuit线路属性包含使用量子比特及经典寄存器个数、测量量子比特等信息属性
- 添加获取线路期望值功能

Changed
*********

- 修改U4门转换后，矩阵不一致问题
- 优化多控门分解：单方向旋转单控门直接通过配置文件进行线路替换
- QPE代码优化
- 修正QFT量子比特顺序错误的问题，并修正了Shor算法涉及QFT的部分代码
- 修改QProg转QASM双门丢失问题
- 修改字符画写文件错位问题

pyqpanda
-------------

Added
*********

- 添加XEB实验功能
- ClassicalCondition添加获取地址信息接口
- 添加用户自定义逻辑门类型的随机线路生成接口
- 获取QProg和QCircuit线路属性包含使用量子比特及经典寄存器个数、测量量子比特等信息属性
- 添加获取线路期望值功能

Changed
*********

- 修改U4门转换后，矩阵不一致问题
- 优化多控门分解：单方向旋转单控门直接通过配置文件进行线路替换
- QPE代码优化
- 修正QFT量子比特顺序错误的问题，并修正了Shor算法涉及QFT的部分代码
- 修改QProg转QASM双门丢失问题
- 修改字符画写文件错位问题

[v2.1.4](https://github.com/OriginQ/QPanda-2/compare/v2.1.3...v2.1.4) - 2020-9-29
==================================================================================

QPanda
-----------

Added
*********

- 添加线路适配异常处理
- MPS 振幅模拟功能
- 添加多控门分解独立接口
- 云量子虚拟机支持链接真实芯片功能

Changed
*********

- 配置文件接口可直接接收json字符串形式的配置信息
- 修改windows控制台字符画乱码问题
- 优化了测量部分
- 优化Shor算法结果处理，使量子算法结果到输出逆元这一步更为可靠
- applications 中 HHL 实例修改
- 修改线路替换时，qubit内存错误问题
- 更新originir支持PI，自然对数
- 修改受控单门无法在多控门分解接口分解的问题

pyqpanda
-------------

Added
*********

- 优化字符画Python接口，可直接print量子程序
- 添加量子线路适配芯片拓扑结构接口
- 添加Grover算法接口
- 添加获取量子线路所用到的qubit的接口
- 添加线路适配异常处理
- MPS 振幅模拟功能
- 添加多控门分解独立接口
- 云量子虚拟机支持链接真实芯片功能

Changed
*********

- 修改windows控制台字符画乱码问题
- 优化了测量部分
- 优化Shor算法结果处理，使量子算法结果到输出逆元这一步更为可靠
- 更新originir支持PI，自然对数


[v2.1.3](https://github.com/OriginQ/QPanda-2/compare/v2.1.2...v2.1.3) - 2020-6-19
==================================================================================

QPanda
--------

Added
*******

- 添加 `SU4` 线路映射功能
- 添加含噪声虚拟机中添加角度旋转误差接口 `set_rotation_angle_error`
- 添加通过泡利矩阵设置噪声模型的方法 `set_noise_kraus_matrix`
- 添加通酉矩阵和概率设置噪声的方法 `set_noise_unitary_matrix`
- 添加生成随机线路的功能 `RandomCircuit`
- 添加 `Base_QCircuit` 文件夹存放基础量子线路，`QFT`，`QPE` 等
- 添加 `HHL` 算法
- 添加 `QARM` 算法
- 添加 `QSVM` 算法
- 添加 `QITE `算法

Changed
*********

- `convert_qasm_to_qprog` 支持科学记数表达式：如 `1.0e-10`
- 修改 `runwithconfiguration` 返回结果的显示方式
- 修复 `free qubit` 内存泄漏
- 修复 `U4 gamma` 值为nan的问题
- 更新线路优化算法
- 去掉噪声虚拟机的默认噪声参数的设置
- 修复 `Psi4Wrapper` 中成员变量未赋初值的bug
- 添加 `QGate::remap` 接口，映射逻辑门量子bit到不同的量子bit

pyQPanda
----------

Added
*******

- 添加 `SU4` 线路映射功能
- 添加含噪声虚拟机中添加角度旋转误差接口 `set_rotation_angle_error`
- 添加通过泡利矩阵设置噪声模型的方法 `set_noise_kraus_matrix`
- 添加通酉矩阵和概率设置噪声的方法 `set_noise_unitary_matrix`
- 添加生成随机线路的功能 `RandomCircuit`
- 添加 `Base_QCircuit` 文件夹存放基础量子线路，`QFT`，`QPE` 等
- 添加 `HHL` 算法
- 添加 `QARM` 算法
- 添加 `QSVM` 算法
- 添加 `QITE `算法

Changed
*********

- `convert_qasm_to_qprog` 支持科学记数表达式：如 `1.0e-10`
- 修改 `runwithconfiguration` 返回结果的显示方式
- 修复 `free qubit` 内存泄漏
- 修复 `U4 gamma` 值为nan的问题
- 更新线路优化算法
- 去掉噪声虚拟机的默认噪声参数的设置
- 修复 `Psi4Wrapper` 中成员变量未赋初值的bug
- 添加 `QGate::remap` 接口，映射逻辑门量子bit到不同的量子bit

[v2.1.2](https://github.com/OriginQ/QPanda-2/compare/v2.1.1...v2.1.2) - 2020-3-31
==================================================================================

QPanda
--------

Added
*******

- `QAdder` 量子加法器功能
- `amplitude_encode` 实现经典数据的量子态振幅编码
- `run_with_configuration` 添加测量次数的接口
- `QCodar` 一种用于各种NISQ设备的上下文持续时间感知的Qubit映射

Changed
*********

- 修改 `QCloudMachine` 接口
- 修改 `SQISWAP` 、`U2` 、`U3` 门中的bug
- 调整 `topology_match` 功能，使QVec完成物理比特映射

pyQPanda
----------

Added
*******

- `QAdder` 量子加法器功能
- `amplitude_encode` 实现经典数据的量子态振幅编码
- `run_with_configuration` 添加测量次数的接口
- `QCodar` 一种用于各种NISQ设备的上下文持续时间感知的Qubit映射

Changed
*********

- 修复 `Shor` 算法测试代码的错误
- 调整 `topology_match` 功能，使QVec完成物理比特映射
- 修改虚拟机中调用 `pmeasure` 系列接口出错的问题


[v2.1.1](https://github.com/OriginQ/QPanda-2/compare/v2.1.0...v2.1.1) - 2020-1-15
==================================================================================

QPanda
--------

Added
*******

- `QCloudMachine` 添加商业云功能

Changed
*********

- 修改 `GTEST` 测试框架
- `ChemiQ` 可以生成动态库
- 修改 `NoiseQVM` 中的算法错误
- 修改 `QIF` 和 `QWHILE` 中的执行错误
- 修改注释部分的乱码引起的编译错误

pyQPanda
----------

Added
*******
- 添加接口: `set_noise_model`: 设置噪声模型
- `QCloudMachine` 添加商业云功能

Changed
*********

- 修改 `NoiseQVM` 中的算法错误
- 修改 `QIF` 和 `QWHILE` 中的执行错误


[v2.1.0](https://github.com/OriginQ/QPanda-2/compare/v2.0.0...v2.1.0) - 2019-11-8
==================================================================================

QPanda
--------

Added
*******

- 添加逻辑门：`I`门.
- 添加接口：`fill_qprog_by_I`：通过I门填充QProg
- 添加接口：`cast_qprog_qgate`：转换Qprog到QGate
- 添加接口：`cast_qprog_qmeasure`：转换Qprog到QMeasure
- 添加接口：`cast_qprog_qcircuit`：转换Qprog到QCircuit
- 添加接口：`NoiseModel::set_noise_model`:设置NoiseModel配置接口
- 添加接口：`flatten`：展开量子程序中的嵌套节点的功能
- 新增功能：单振幅量子虚拟机中可运行`SWAP`门
- 添加接口：`convert_qprog_to_binary`：转换QProg到二进制
- 添加接口：`convert_binary_data_to_qprog`：转换二进制到QProg
- 添加接口：`convert_originir_to_qprog`：转换Qoriginir到QProg
- 添加接口：`convert_qasm_to_qprog`：新增QASM转QProg的方法
- 添加新的含噪声虚拟机模型:`DECOHERENCE_KRAUS_OPERATOR_P1_P2`, `BITFLIP_KRAUS_OPERATOR`, `DEPOLARIZING_KRAUS_OPERATOR`, `BIT_PHASE_FLIP_OPRATOR`, `PHASE_DAMPING_OPRATOR`
- 添加接口：`convert_qprog_to_originir`：转换QProg到Qoriginir
- 添加接口：`convert_qprog_to_quil`：转换QProg到QUil
- 添加接口：`convert_qprog_to_qasm`：转换QProg到QASM

Changed
*********

- 原字符画接口`print_prog`改为：`draw_qprog`
- 原`QVM::setConfigure`为 `QVM::setConfig`
- 通过重载std::cout，直接输出目标线路的字符画

pyQPanda
----------

Added
*******

- 添加逻辑门：`I`门
- 添加接口：`fill_qprog_by_I`：通过I门填充QProg
- 添加接口：`cast_qprog_qgate`：转换Qprog到QGate
- 添加接口：`cast_qprog_qmeasure`：转换Qprog到QMeasure
- 添加接口：`cast_qprog_qcircuit`：转换Qprog到QCircuit，遇到流控节点或者测量节点，返回false
- 添加接口：`flatten`：添加量子程序或线路展开功能的python接口
- 添加接口：`convert_qprog_to_binary`：转换QProg到二进制
- 添加接口：`convert_binary_data_to_qprog`：转换二进制到QProg
- 添加接口：`convert_originir_to_qprog`：转换Qoriginir到QProg
- 添加接口：`convert_qprog_to_originir`：转换QProg到Qoriginir
- 添加接口：`convert_qprog_to_quil`：转换QProg到QUil
- 添加接口：`convert_qasm_to_qprog`：新增QASM转QProg的方法
- 添加接口：`convert_qprog_to_qasm`：转换QProg到QASM

Changed
*********

- 调整接口：打印字符画接口`print_qprog`修改为`draw_qprog`

[v2.0.0](https://github.com/OriginQ/QPanda-2/compare/v1.3.5...v2.0.0) - 2019-9-30
===============================================================================

QPanda
--------

Added
*******
- QPanda重构了项目框架把QPanda分为Applications、QAlg、Components、Core四层。
- 添加接口`getAllocateCMemNum`：获取申请经典比特的数量
- 添加接口`pMeasureNoIndex`：概率测量
- 添加接口`createEmptyCircuit`：创建空的量子线路
- 添加接口`QWhile::getClassicalCondition`： 获得经典表达式
- 添加接口`createWhileProg`：创建QWhile
- 添加接口`createIfProg`： 创建QIf
- 添加接口`createEmptyQProg`：创建量子程序
- 添加接口`QVM::setConfigure`: 设置比特数和经典寄存器数
- 添加接口`QVM:: qAlloc`: 申请量子比特
- 添加接口`QVM::qAllocMany`：申请多个量子比特
- 添加接口`QVM::getAllocateQubitNum`：获取申请的量子比特数
- 添加接口`QVM::getAllocateCMemNum` 获取申请的经典寄存器数
- 添加接口`QVM::cAlloc`: 申请一个经典寄存器
- 添加接口`QVM::cAllocMany`：申请多个经典寄存器
- 添加接口`SingleAmplitudeQVM：pMeasureBinIndex`： 通过二进制下标进行PMeasure操作
- 添加接口`SingleAmplitudeQVM：pMeasureDecIndex`： 通过十进制下标进行PMeasure操作
- 添加接口`CPUQVM:: pMeasureNoIndex`: PMeasure操作
- 添加接口`validateSingleQGateType`： 验证单量子逻辑门有效性
- 添加接口`validateDoubleQGateType`：验证双量子逻辑门有效性
- 添加接口`getUnsupportQGateNum`：统计量子程序（包含量子线路、QIF、QWHILE）中不支持的逻辑门的数量
- 添加接口`getQGateNum`：统计量子程序（包含量子线路、QIF、QWHILE）中逻辑门的数量
- 添加接口`transformBinaryDataToQProg`： 解析二进制数据转化为量子程序
- 添加接口`transformQProgToBinary`：量子程序转化为二进制数据

pyQPanda
----------

Added
*******

- 添加接口`cAlloc`: 申请一个固定位置上的经典比特
- 添加接口`cFree_all`：释放传入的所有经典寄存器
- 添加接口`get_allocate_qubit_num`： 获取申请量子比特的数量
- 添加接口`get_allocate_cmem_num`：获取申请经典比特的数量
- 添加接口`get_prob_tuple_list`：获得目标量子比特的概率测量结果， 其对应的下标为十进制，需先调用directlyRun
- 添加接口`get_prob_list`：获得目标量子比特的概率测量结果， 并没有其对应的下标，需先调用directlyRun
- 添加接口`get_prob_dict`：获得目标量子比特的概率测量结果， 其对应的下标为二进制，需先调用directlyRun
- 添加接口`pmeasure_no_index`：概率测量
- 添加接口`accumulate_probability`：累计概率
- 添加接口`QGate::set_dagger`：设置逻辑门转置共轭
- 添加接口`QGate::set_control`：设置逻辑门控制比特
- 添加接口`QCircuit::set_dagger`：设置线路转置共轭
- 添加接口`Circuit::set_control`：设置线路控制比特
- 添加接口`create_empty_circuit`： 申请空线路
- 添加接口`QWhileProg::get_true_branch`：获取正确分支
- 添加接口`QWhileProg::get_classical_condition` 获取判断表达式
- 添加接口`QIfProg::get_true_branch`：获取正确分支
- 添加接口`QIfProg::get_classical_condition`： 获取判断表达式
- 添加接口`QIfProg::get_false_branch`：获取失败分支
- 添加接口`create_If_prog`：创建QIf
- 添加接口`create_empty_qprog`：创建QProg
- 添加接口`QVM::allocate_qubit_through_phy_address`：通过量子比特物理地址申请量子比特
- 添加接口`QVM::allocate_qubit_through_vir_address`：通过量子比特虚拟地址申请量子比特
- 添加接口`QVM::get_result_map`：获取结果map
- 添加接口`QVM::get_allocate_qubit_num`：获取申请比特数
- 添加接口`QVM::get_allocate_cmem_num`：获取申请经典寄存器数
- 添加接口`QVM::init_qvm`：初始化量子虚拟机
- 添加接口`PartialAmplitudeQVM::pmeasure_subset`：获取量子态任意子集的结果
- 添加接口`validate_single_qgate_type`：验证单量子逻辑门有效性
- 添加接口`validate_double_qgate_type`: 验证双量子逻辑门有效性
- 添加接口`transform_qprog_to_originir`：量子程序转化OriginIR
- 添加接口`transform_originir_to_qprog`：OriginIR转化量子程序
- 添加接口`get_unsupport_qgate_num`：统计量子程序（包含量子线路、QIF、QWHILE）中不支持的逻辑门的数量
- 添加接口`get_qgate_num`：统计量子程序（包含量子线路、QIF、QWHILE）中逻辑门的数量
- 添加接口`get_qprog_clock_cycle`：统计量子程序时钟周期
- 添加接口`transform_binary_data_to_qprog`：解析二进制数据转化为量子程序
- 添加接口`transform_qprog_to_binary`：量子程序转化为二进制数据
- 添加接口`transform_qprog_to_qasm`：量子程序转化为QASM指令集
- 添加接口`transform_qprog_to_quil`：量子程序转化为Quil指令集

