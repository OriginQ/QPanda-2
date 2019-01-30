.. _Measure:

量子测量
================

量子测量是指通过外界对量子系统进行干扰来获取需要的信息，测量门使用的是蒙特卡洛方法的测量。在量子线路中用如下图标表示：

.. image:: imagine/measure.svg
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

实例
---------

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