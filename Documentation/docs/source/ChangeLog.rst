Change Log
================

1.3.0 - 2019-04-28 
-------------------------
add
*******
1. 添加量子机器学习框架VQNet。
2. 添加部分振幅量子虚拟机PartialAmplitudeQVM。
3. 添加单振幅量子虚拟机SingleAmplitudeQVM。
4. 添加含噪声量子虚拟机NoiseQVM。
5. 添加云虚拟机QCloudQVM。
6. 添加量子程序持久化存储功能QProgStored
7. 添加解析持久化存储的量子程序文件QProgDataParse
8. 添加统计量子程序中逻辑门个数的功能QGateCounter

update
***********

1. 更新QProgToQASM的对外接口为transformQProgToQASM
2. 更新QProgToQRunes的对外接口为transformQProgToQRunes
3. 更新量子虚拟机部分对外接口


1.2.0 - 2019-01-18 
-------------------------
add
*******

1. 用户可使用QProg()、QCircuit()、QWhileProg(...)、QIfProg(...)构造相关对象。
2. 对外隐藏CBit类型，统一替换为ClassicalCondition。
3. 实现ClassicalCondition 的+,-,*,/，>,>=,<,<=,=,==功能。
4. 实现经典运算在量子虚拟机使用的左值和右值引用功能；实现在量子程序中插入经典运算的功能。
5. 实现在操作量子逻辑门时，目标量子比特集合下边可为变量的功能。
6. 添加qAllocMany接口。此接口可以申请多个量子比特，输入参数为量子比特数，返回值类型修改为QVec。
7. 添加cAllocMany接口，此接口可以申请多个经典寄存器，返回值类型修改为std::vector<ClassicalCondition>。
8. cFreeAll的输入参数类型修改为std::vector<ClassicalCondition>。
9. 添加getProbTupleList接口。使用PMEASURE获取量子程序结果，并返回pair<量子态，概率>类型的vector。
10. 添加getProbList接口。使用PMEASURE获取量子程序结果，并返回概率值的vector。
11. 添加getProbDict接口。使用PMEASURE获取量子程序结果，并返回pair<量子态，概率>类型的map。
12. 添加probRunTupleList接口，根据输入的量子程序，使用PMEASURE获取量子程序结果，并返回pair<量子态，概率>类型的vector。
13. 添加probRunList接口，根据输入的量子程序，使用PMEASURE获取量子程序结果，并返回概率的vector。
14. 添加probRunDict接口，根据输入的量子程序，使用PMEASURE获取量子程序结果，并返回pair<量子态，概率>类型的map。
15. 添加runWithConfiguration接口，根据输入的量子程序和循环次数，统计测量结果，并返回pari<量子态，概率>的map。
16. 添加MeasureAll接口，返回一个量子程序，该量子程序对输入的所有量子比特进行Measure操作。
17. 添加initQuantumMachine接口，此接口可根据输入的QuantumMachine_type返回一个QuantumMachine类型的虚拟机指针。

update
***********

1. cAlloc和cAlloc(size_t)返回值修类型修改为ClassicalCondition
2. cFree的输入参数类型改为ClassicalCondition&
3. 删除getCBitValue接口
4. PMeasure接口的qubit_vector参数数据类型改为QVec
5. PMeasure_no_index接口的qubit_vector参数数据类型改为QVec
6. quick_measure接口的qubit_vector参数数据类型改为QVec
7. init接口添加QuantumMachine_type参数，可以设置生成的量子虚拟机的类型
8. CreateHadamardQCircuit接口的输入参数pQubitVector数据类型改为QVec
