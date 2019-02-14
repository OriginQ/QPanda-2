QIf
==========
----

QIf表示量子程序条件判断操作，输入参数为条件判断表达式，功能是执行条件判断

QIf语法规则
>>>>>>>>>>>>>>>>
----

QIf表示量子程序条件判断操作，输入参数为条件判断表达式，功能是执行条件判断。QIf的语法规则参考量子程序转化QRunes模块中的 :ref:`QRunes介绍` 部分。

.. _api_introduction:

接口介绍
>>>>>>>>>>>
----

.. cpp:class:: QIfProg

    QIfProg是一种和量子控制流相关的节点。QIf接受一个储存在测控设备中的变量，或者由这些变量构成的表达式，通过判断它的值为True/False，选择程序接下来的执行分支。

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

.. note:: QIfProg和普通的If， While截然不同的原因是这个判断过程仅仅在测控设备中执行，并且要求了极高的实时性。因此，所有的True和False分支都会被输入到QlfProg里面去执行。

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

