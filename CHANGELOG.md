# Changelog

所有针对QPanda的修改都会记录在本文件中.

> **修改的类型:**
>
> -   **Added**: 添加新的功能.
> -   **Changed**: 对现有的功能进行修改.
> -   **Deprecated**: 即将删除的功能.
> -   **Removed**: 删除的功能.
> -   **Fixed**: Bug修复.

[UNRELEASED](https://github.com/OriginQ/QPanda-2/compare/v2.1.0...HEAD)
====================================================================

[v2.1.0](https://github.com/OriginQ/QPanda-2/compare/v2.0.0...v2.1.0) - 2019-11-8
===============================================================================

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

