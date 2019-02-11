QWhile
==============
----

量子程序循环控制操作，输入参数为条件判断表达式，功能是执行while循环操作

QWhile语法规则
>>>>>>>>>>>>>>>>>>
----

QWHILE控制语句，起始标识是QWHILE,终止标识是 ENDWHILE。
QWHILE带有输入参数 条件判断表达式 。QWHILE与ENDWHILE标识之间为QWHILE的正确分支，QWhile不包含错误分支。
QWHILE中可嵌套QWHILE，也可包含QIF。

接口介绍
>>>>>>>>>>>>>
----

.. cpp:class:: QWhileProg

    该类用于表述一个QWhile节点的各项信息，同时包含多种可调用的接口。

    .. cpp:function:: NodeType getNodeType()

       **功能**
            获取节点类型
       **参数**
            无
       **返回值**
            节点类型

    .. cpp:function:: QNode* getTrueBranch()

       **功能**
            获取正确分支节点
       **参数**
            无
       **返回值**
            正确分支节点

    .. cpp:function:: ClassicalCondition getCExpr()

       **功能**
            获取逻辑判断表达式
       **参数**
            无
       **返回值**
            量子表达式


实例
>>>>>>>>>>
----

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            init();
            QProg prog;
            auto qvec = qAllocMany(3);
            auto cvec = cAllocMany(3);
            cvec[1].setValue(0);
            cvec[0].setValue(0);
            
            QProg prog_in;
            prog_in<<H(cvec) ;
            auto qwhile = CreateWhileProg(cvec[1]<3,&prog_in);
            prog<<qwhile;
            auto result = probRunTupleList(prog, qvec);

            for (auto & val : result)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            finalize();
            return 0;
        }