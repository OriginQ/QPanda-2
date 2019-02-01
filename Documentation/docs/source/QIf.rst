QIF
==========
----

QIF表示量子程序条件判断操作，输入参数为条件判断表达式，功能是执行条件判断

QIF语法规则
>>>>>>>>>>>>>>>>
----

QIF控制语句，起始标识是QIF,终止标识是 ENDQIF，分支分割标识为ELSE。
QIF 带有输入参数条件判断表达式 。QIF与ELSE标识之间为QIF的正确分支，ELSE与ENDQIF标识之间为QIF的错误分支。
QIF中可嵌套QIF，也可包含QWHILE

接口介绍
>>>>>>>>>>>
----

.. cpp:class:: QIfProg

    该类用于表述一个QIf节点的各项信息，同时包含多种可调用的接口。

    .. cpp:function:: NodeType getNodeType()

        **功能**
            获取节点类型
        **参数**
            无
        **返回值**
            节点类型

    .. cpp:function::  QNode* getTrueBranch()

       **功能**
            获取正确分支节点
       **参数**
            无
       **返回值**
            正确分支节点

    .. cpp:function:: QNode* getFalseBranch()

       **功能**
            获取错误分支节点
       **参数**
            无
       **返回值**
            错误分支节点

    .. cpp:function:: ClassicalCondition getCExpr()

       **功能**
            获取逻辑判断表达式
       **参数**
            无
       **返回值**
            量子表达式


实例
>>>>>>>>>
----

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            init();
            QProg prog;

            auto qvec = qAllocMany(5);
            auto cvec = cAllocMany(2);
            cvec[1].setValue(0);
            cvec[0].setValue(0);
            QProg branch_true;
            QProg branch_false;
            branch_true << (cvec[1]=cvec[1]+1) << H(qvec[cvec[0]]) << (cvec[0]=cvec[0]+1);
            branch_false << H(qvec[0]) << CNOT(qvec[0],qvec[1]) << CNOT(qvec[1],qvec[2])
                        << CNOT(qvec[2],qvec[3]) << CNOT(qvec[3],qvec[4]);
            auto qwhile = CreateIfProg(cvec[1]>5,&branch_true, &branch_false);
            prog<<qwhile;
            auto result = probRunTupleList(prog, qvec);

            for (auto & val : result)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            finalize();
            return 0;
        }

