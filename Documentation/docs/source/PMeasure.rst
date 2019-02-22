概率测量
==================

PMeasure和 :ref:`Measure` 是完全不同的过程，Measure是执行了一次测量， 并返回一个确定的0/1结果。PMeasure是输出当前时刻的量子虚拟机的概率结果，PMeasure并不会改变量子态。

PMeasure的输入参数是 ``QVec``， 它指定了我们关注的量子比特。例如，一共有10个Qubit的系统，我们指定了前三个Qubit作为PMeasure的目标，就会输出一个长度为8的vector。

接口介绍
----------------

.. _getProbTupleList:
.. cpp:function:: std::vector<std::pair<size_t, double>> getProbTupleList(QVec &qvec,int selectMax=-1)

    **功能**
         获得PMeasure结果。
    **参数**
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
         PMeasure得到的结果， 返回下标和对应的概率。
    **实例**
        .. code-block:: c

            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto qvec = qAllocMany(4);
                auto cvec = cAllocMany(4);
                prog << H(qvec[0]) << CNOT(qvec[0],qvec[1])
                    << CNOT(qvec[1],qvec[2]) << CNOT(qvec[2],qvec[3]);

                load(prog);
                run();
                auto result = getProbTupleList(qvec, -1); // 对所有的目标比特做概率测量
                for (auto &val : result)  // 输出所有的PMeasure结果
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }

.. _getProbList:
.. cpp:function:: std::vector<double> getProbList(QVec& qvec, int selectMax = -1)

    **功能**
        获得PMeasure结果
    **参数**
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        PMeasure得到的概率
    **实例**
        .. code-block:: c
        
            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto qvec = qAllocMany(4);
                auto cvec = cAllocMany(4);
                prog << H(qvec[0]) << CNOT(qvec[0],qvec[1])
                    << CNOT(qvec[1],qvec[2]) << CNOT(qvec[2],qvec[3]);

                load(prog);
                run();
                auto result = getProbList(qvec); // 对所有的目标比特做概率测量
                for (auto &val : result) // 输出所有的PMeasure结果
                {
                    std::cout << val << std::endl;
                }

                finalize();
                return 0;
            }

.. _getProbDict:
.. cpp:function:: std::map<std::string, double>  getProbDict(QVec &qvec, int selectMax = -1)

    **功能**
        获得PMeasure结果
    **参数**
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        PMeasure得到结果， 下标的二进制和对应的概率
    **实例**
        .. code-block:: c

            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto qvec = qAllocMany(4);
                auto cvec = cAllocMany(4);
                prog << H(qvec[0]) << CNOT(qvec[0],qvec[1])
                    << CNOT(qvec[1],qvec[2]) << CNOT(qvec[2],qvec[3]);

                load(prog);
                run();
                auto result = getProbDict(qvec); // 对所有的目标比特做概率测量
                for (auto &val : result) // 输出所有的PMeasure结果
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }

.. cpp:function:: std::vector<std::pair<size_t, double>> probRunTupleList(QProg &prog,QVec &qvec, int selectMax = -1)
    
    **功能**
        获得PMeasure结果,不需要load和run
    **参数**
        - prog 量子程序
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        PMeasure得到结果， 下标的二进制和对应的概率
    **实例**
        .. code-block:: c

            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto qvec = qAllocMany(4);
                auto cvec = cAllocMany(4);
                prog << H(qvec[0]) << CNOT(qvec[0],qvec[1])
                    << CNOT(qvec[1],qvec[2]) << CNOT(qvec[2],qvec[3]);

                auto result = probRunTupleList(prog, qvec); // 对所有的目标比特做概率测量
                for (auto &val : result) // 输出所有的PMeasure结果
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }

**参照** getProbTupleList_

.. cpp:function:: std::vector<double> probRunList(QProg &prog,QVec &qvec , int selectMax = -1)
    
    **功能**
        获得PMeasure结果,不需要load和run
    **参数**
        - prog 量子程序
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        PMeasure得到概率
    **实例**
        .. code-block:: c

            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto qvec = qAllocMany(4);
                auto cvec = cAllocMany(4);
                prog << H(qvec[0]) << CNOT(qvec[0],qvec[1])
                    << CNOT(qvec[1],qvec[2]) << CNOT(qvec[2],qvec[3]);

                auto result = probRunList(prog, qvec); // 对所有的目标比特做概率测量
                for (auto &val : result) // 输出所有的PMeasure结果
                {
                    std::cout << val << std::endl;
                }

                finalize();
                return 0;
            }

**参照** getProbList_ 

.. cpp:function:: std::vector<double> probRunDict(QProg &prog,QVec &qvec, int selectMax = -1)
    
    **功能**
        获得PMeasure结果,不需要load和run
    **参数**
        - prog 量子程序
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        PMeasure得到结果， 下标的二进制和对应的概率
    **实例**
        .. code-block:: c

            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto qvec = qAllocMany(4);
                prog << H(qvec[0]) << CNOT(qvec[0],qvec[1])
                    << CNOT(qvec[1],qvec[2]) << CNOT(qvec[2],qvec[3]);

                auto result = probRunDict(prog, qvec); // 对所有的目标比特做概率测量
                for (auto &val : result) // 输出所有的PMeasure结果
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }

**参照** getProbDict_