QRunes转化量子程序
====================
----

通过该功能模块，你可以解析QRunes文本文件，将其中的量子逻辑门操作信息提取出来，得到QPanda2内部可操作的量子程序。

QRunes格式
>>>>>>>>>>>>
----

QRunes的书写格式规范与例程可以参考量子程序转化QRunes模块中的 :ref:`QRunes介绍` 部分,本模块对QRunes的书写格式添加了额外的功能拓展。主要有以下几点。

 -  ``RX`` 表示旋转门操作，第一个参数是目标量子比特，第二个参数可以是一个确定的角度值，也可以是一个数学表达式，支持浮点型数据的四则运算与括号等，
    如 ``RX q1,(-1.1+3)*(pi/2)`` 等。与之类似的量子逻辑门操作有 ``RY`` ， ``RZ`` 等。
 -  ``QIF`` 与 ``QWHILE`` 的作用是根据经典寄存器的值进行逻辑判断，后面的参数可以是一个经典寄存器，如 ``c0`` ，也可以是包含经典寄存器的逻辑运算表达式，
    如 ``c0+c1||c2+c0&c3`` 等。

功能函数接口
>>>>>>>>>>>>
----

你可以通过调用 ``qRunesToQProg(string sQRunesPath)`` 接口来调用该功能,该接口说明如下：
  
    .. cpp:function:: qRunesToQProg(string sQRunesPath)

       **功能**
        - 将QRunes转化为量子程序

       **参数**
        - QRunes文件路径

       **返回值**
        - QProg量子程序

使用例程
>>>>>>>>
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
            init(QuantumMachine_type::CPU);

            auto prog = CreateEmptyQProg();

            prog = qRunesToQProg(sQRunesPath);

            finalize();
            return 0;
        }


具体步骤如下:

 - 首先在主程序中用 ``init()`` 进行全局初始化

 - 接着用 ``CreateEmptyQProg()`` 创建一个空的量子程序，用于接收返回值

 - 然后调用 ``qRunesToQProg(sQRunesPath)`` 获取转化后的量子程序

 - 最后用 ``finalize()`` 结束，并释放系统资源

   .. tip:: 我们可以调用量子程序转化QRunes函数接口qProgToQRunes(QProg &)来验证是否转化成功
    
    
错误提示
>>>>>>>>
----

假如在解析QRunes文件直到生成量子程序的过程中发生错误，你可以根据控制台打印的错误信息来判断断出错的类型，以下是错误信息及描述。

===================    ================================================
运行错误代号              错误描述
===================    ================================================
``FileOpenError``        打开文件失败或文件不存在
``KeyWordsError``        QRunes不支持的关键词
``MatchingError``        部分关键词找不到与之对应的关键词，如CONTROL等
``IsIntError``           操作参数错误，非整型数据
``IsDoubleError``        操作参数错误，非浮点型数据
``ExpressionError``      计算表达式格式错误
``FormalError``          其他QRunes语法格式上的问题
===================    ================================================
