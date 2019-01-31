.. _Measure:

量子测量
================

量子测量是指通过外界对量子系统进行干扰来获取需要的信息，测量门使用的是蒙特卡洛方法的测量。在量子线路中用如下图标表示：

.. image:: imags/measure.svg
    :width: 65

接口介绍
----------------

.. cpp:function:: QMeasure Measure(Qubit * targetQuBit, ClassicalCondition & cc)

       **功能**
        - 获得量子测量对象

       **参数**
        - targetQuBit 测量比特
        - cc 量子表达式，测量结果存储在量子表达式的经典寄存器中
       **返回值**
        - 量子测量对象
       **实例**

        .. code-block:: c

            #include <QPanda.h>
            USING_QPANDA


            int main(void)
            {
                init();
                QProg prog;
                auto q = qAlloc();
                auto c = cAlloc();

                auto gate = H(q); // 得到量子逻辑门
                auto measure = Measure(q, c); // 得到量子测量
                prog << gate << measure;
                load(prog);
                run();

                auto result = getResultMap();
                for (auto &val : result)
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }

.. cpp:function:: QProg MeasureAll(QVec& qvec, std::vector<ClassicalCondition> &cvec)

    **功能**
        测量所有目标比特
    **参数**
        - qvec 目标比特容器
        - cvec 量子表达式容器
    **返回值**
        量子程序    
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
            prog << H(qvec[0]) << CNOT(qvec[0], qvec[1]) << CNOT(qvec[1], qvec[2])
                << CNOT(qvec[2], qvec[3]);

            auto prog_measure = MeasureAll(qvec, cvec); // 得到测量所有目标比特的量子程序
            prog << prog_measure;
            load(prog);
            run();

            auto result = getResultMap();
            for (auto &val : result)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            finalize();
            return 0;
        } 

.. _runWithConfiguration:

.. cpp:function:: std::map<std::string, size_t> runWithConfiguration(QProg &prog, std::vector<ClassicalCondition> &cvec, int shorts)
    
    **功能**
        - 末态在目标量子比特序列在量子程序多次运行结果中出现的次数,不需要load和run
    **参数**
        - prog 量子程序
        - cvec 量子表达式vector
        - shorts 量子程序运行的次数
    **返回值**
        - 目标量子比特序列二进制及其对应的次数
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
                prog << H(qvec[0]) << H(qvec[1]) << H(qvec[2]) << H(qvec[3])
                    << Measure(qvec[0], cvec[0]) << Measure(qvec[1], cvec[1])
                    << Measure(qvec[2], cvec[2]) << Measure(qvec[3], cvec[3]);

                auto result = runWithConfiguration(prog, cvec, 1000);
                for (auto &val : result)
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }

.. cpp:function:: std::map<std::string, size_t> quickMeasure(QVec &qvec, int shorts);
    
    **功能**
        - 末态在目标量子比特序列在量子程序多次运行结果中出现的次数
    **参数**
        - qvec 目标量子比特
        - shorts 量子程序运行的次数
    **返回值**
        - 目标量子比特序列二进制及其对应的次数
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
                prog << H(qvec[0]) << H(qvec[1]) << H(qvec[2]) << H(qvec[3]);
                load(prog);
                run();

                auto result = quickMeasure(qvec, 1000);
                for (auto &val : result)
                {
                    std::cout << val.first << ", " << val.second << std::endl;
                }

                finalize();
                return 0;
            }


**see also** runWithConfiguration_