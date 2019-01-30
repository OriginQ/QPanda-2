PMeasure接口
=================

PMeasure只能在量子虚拟机上实现，就是计算出末态在目标量子比特序列对应的空间的各个基矢上的投影(概率), 在量子线路中的表示符号与 :ref:`Measure` 相同，如下图表示：

.. image:: imagine/measure.svg
    :width: 65

接口介绍
----------------

.. cpp:function:: std::vector<std::pair<size_t, double>> getProbTupleList(QVec &qvec,int selectMax=-1)

    **功能**
        - 获得PMeasure结果
    **参数**
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        - PMeasure得到的结果， 返回下标和对应的概率
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

.. cpp:function:: std::vector<double> getProbList(QVec& qvec, int selectMax = -1)

    **功能**
        - 获得PMeasure结果
    **参数**
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        - PMeasure得到的概率
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

.. cpp:function:: std::map<std::string, double>  getProbDict(QVec &qvec, int selectMax = -1)

    **功能**
        - 获得PMeasure结果
    **参数**
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        - PMeasure得到结果， 下标的二进制和对应的概率
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
        - 获得PMeasure结果,不需要load和run
    **参数**
        - prog 量子程序
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        - PMeasure得到结果， 下标的二进制和对应的概率
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

.. cpp:function:: std::vector<double> probRunList(QProg &prog,QVec &qvec , int selectMax = -1)
    
    **功能**
        - 获得PMeasure结果,不需要load和run
    **参数**
        - prog 量子程序
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        - PMeasure得到概率
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

.. cpp:function:: std::vector<double> probRunDict(QProg &prog,QVec &qvec, int selectMax = -1)
    
    **功能**
        - 获得PMeasure结果,不需要load和run
    **参数**
        - prog 量子程序
        - qvec 目标量子比特
        - selectMax 需要PMeasure的目标量子比特个数，若为-1，则是所有的目标量子比特
    **返回值**
        - PMeasure得到结果， 下标的二进制和对应的概率
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





