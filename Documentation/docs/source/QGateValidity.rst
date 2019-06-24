.. _QGateVaildity:

量子门的有效性
=================

量子门的分类
-----------------

量子门主要分为单量子门、双量子门和三量子门，根据量子芯片中元数据不同的单量子门或双量子门组合对单量子门和双量子门又分为了不同的类型，下面分别对单量子门和双量子门进行介绍。

单比特逻辑门判断
````````````````

单量子门可分为四种情况：任意旋转类，双连续类，单连续单离散类，双离散类。下面根据量子芯片的元数据判断单量子门所属类型。

1. **任意旋转类**：需要判断元数据提供的单量子门的容器中是否存在 ``U3`` 、 ``U2`` 的量子门类型，如果存在则是任意旋转类，如果不存在则不是任意旋转类；

2. **双连续类**：需要判断元数据提供的单量子门的容器中是否存在 ``RX+RZ`` 、 ``RX+RY`` 、 ``RY+RZ`` 这三种组合，如果存在则是双连续类，如果不存在则不是双连续类；

3. **单连续单离散类**：第一步，需要判断元数据提供的单量子门的容器中是否存在 ``RX`` 、 ``RY`` 、 ``RZ`` 这三种量子门；第二步，如果存在 ``RX``，不存在 ``RY`` 、 ``U1``，则查找容器中是否有 ``Y`` 、 ``H`` 、 ``Z`` 量子门中的一种，如果存在则是单连续单离散类，如果不存在则不是单连续单离散类。如果存在 ``RY``，不存在 ``RX`` 、 ``U1``，则查找容器中是否有 ``X`` 、 ``H`` 、 ``Z`` 量子门中的一种，如果存在则是单连续单离散类，如果不存在则不是单连续单离散类。如果存在 ``U1`` ，不存在 ``RX`` 、 ``RY`` ,则查找容器中是否存在 ``X`` 、 ``H`` 、 ``Y`` 中的一种，如果存在则是单连续单离散类，如果不存在则不是单连续单离散类。

4. **双离散类**：如果元数据提供的单量子门的容器中不存在 ``RX`` 、 ``RY`` 、 ``U1`` 这三种量子门，则判断 ``H+T`` 、 ``X+T`` 、 ``Y+T`` 的组合，如果存在则是双离散类，如果不存在则不是双离散类。

`当以上四种情况都不存在时，则当前芯片元数据提供的单逻辑门信息不是有效信息。`

双量子门判断
````````````````

双量子门需要判断元数据提供的双量子门的容器中是否 ``CNOT`` 、 ``SWAP`` 门，如果存在则元数据可用，如果不存在则元数据不可用。

判断量子门有效性的方法
--------------------------

判断量子门有效性是由 ``SingleGateTypeValidator`` 和 ``DoubleGateTypeValidator`` 两个类实现的。下面分别对这两个类中的接口介绍。

接口介绍
---------------

``SingleGateTypeValidator`` 是验证单量子逻辑门有效性的工具类，使用方式如下：

    .. code-block:: c

        std::vector<std::string> single_gate;
        std::vector<std::string> valid_single_gate; // 有效的量子逻辑门组合会输出到这个容器中
        single_gate.push_back("T");
        single_gate.push_back("H");
        single_gate.push_back("S");
        auto single_gate_type = SingleGateTypeValidator::GateType(single_gate, valid_single_gate); // 得到有效的量子逻辑门组合，和有效组合的类型

``DoubleGateTypeValidator`` 是验证双量子逻辑门有效性的工具类，使用方式如下：

    .. code-block:: c

        std::vector<std::string> double_gate;
        std::vector<std::string> valid_double_gate; // 有效的量子逻辑门组合会输出到这个容器中
        double_gate.push_back("CNOT");
        double_gate.push_back("SWAP");
        double_gate.push_back("CZ");
        auto double_gate_type = DoubleGateTypeValidator::GateType(double_gate, valid_double_gate); // 得到有效的量子逻辑门组合，和有效组合的类型

实例
------------

    .. code-block:: c
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            // 单量子门有效性验证
            std::vector<std::string> single_gate;
            std::vector<std::string> valid_single_gate; // 有效的量子逻辑门组合会输出到这个容器中
            single_gate.push_back("T");
            single_gate.push_back("H");
            single_gate.push_back("S");

            // 得到有效的量子逻辑门组合，和有效组合的类型
            auto single_gate_type = SingleGateTypeValidator::GateType(single_gate, valid_single_gate);

            std::cout << "SingleGateTransferType: " << single_gate_type << std::endl;
            for (auto &val : valid_single_gate)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            // 双量子门有效性验证
            std::vector<std::string> double_gate;
            std::vector<std::string> valid_double_gate; // 有效的量子逻辑门组合会输出到这个容器中
            double_gate.push_back("CNOT");
            double_gate.push_back("SWAP");
            double_gate.push_back("CZ");

            // 得到有效的量子逻辑门组合，和有效组合的类型
            auto double_gate_type = DoubleGateTypeValidator::GateType(double_gate, valid_double_gate);

            std::cout << "doubleGateTransferType: " << double_gate_type << std::endl;
            for (auto &val : valid_double_gate)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            return 0;
        }

    
运行结果：

    .. code-block:: c

        SingleGateTransferType: 3
        T H 
        doubleGateTransferType: 0
        CNOT 
