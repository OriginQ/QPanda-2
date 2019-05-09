QRunes转化为量子程序
=======================
----

通过该功能模块，你可以解析QRunes文本文件，将其中的量子逻辑门操作信息提取出来，得到QPanda 2内部可操作的量子程序。

QRunes
>>>>>>>
----

QRunes的书写格式规范与例程可以参考量子程序转化QRunes模块中的 :ref:`QRunes介绍` 部分。

QPanda 2提供了QRunes文件转换工具接口 ``transformQRunesToQProg(std::string sFilePath, QProg& prog,QuantumMachine* qvm)`` 该接口使用非常简单，具体可参考下方示例程序。

实例
>>>>>>>
----

在使用该功能之前，需要先书写QRunes量子程序，以 :ref:`QRunes介绍` 中的文件格式作为例子

    ::

        QINIT 6
        CREG 2
        H 0
        CNOT 0,1
        CONTROL 1
        X 1
        Y 2
        ENDCONTROL 1
        DAGGER
        X 2
        CZ 0,1
        ENDDAGGER
        MEASURE 0,$0
        MEASURE 1,$1

接下来通过简单的接口调用演示了QRunes指令集转化量子程序的过程

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        const string sQRunesPath("D:\\QRunes");

        int main(void)
        {
            auto qvm = initQuantumMachine();
            auto prog = CreateEmptyQProg();
            transformQRunesToQProg("D:\\QRunes", prog,qvm);

            qvm->finalize();
            delete qvm;
            return 0;
        }


具体步骤如下:

 - 首先在主程序中用 ``initQuantumMachine()`` 初始化一个量子虚拟机对象，用于管理后续一系列行为

 - 接着用 ``CreateEmptyQProg()`` 创建一个空的量子程序，用于接收返回值

 - 然后调用 ``transformQRunesToQProg(sQRunesPath, prog，qvm)`` 转化

 - 最后用 ``finalize()`` 结束，并释放系统资源

   .. tip:: 我们可以调用量子程序转化QRunes函数接口transformQProgToQRunes(QProg &)来验证是否转化成功
    
    
错误提示
>>>>>>>>
----

假如在解析QRunes文件直到生成量子程序的过程中发生错误，你可以根据控制台打印的错误信息来判断出错的类型，以下是错误信息及描述。

===================    ================================================
运行错误代号              错误描述
===================    ================================================
| ``FileOpenError``      | 打开文件失败或文件不存在
| ``KeyWordsError``      | QRunes不支持的关键词
| ``MatchingError``      | 部分关键词找不到与之对应的关键词，如CONTROL等
| ``IsIntError``         | 操作参数错误，非整型数据
| ``IsDoubleError``      | 操作参数错误，非浮点型数据
| ``ExpressionError``    | 计算表达式格式错误
| ``FormalError``        | 其他QRunes语法格式上的问题
===================    ================================================
