量子逻辑门
====================
----

经典计算中，最基本的单元是比特，而最基本的控制模式是逻辑门。我们可以通过逻辑门的组合来达到我们控制电路的目的。类似地，处理量子比特的方式就是量子逻辑门。
使用量子逻辑门，我们有意识的使量子态发生演化。所以量子逻辑门是构成量子算法的基础。

量子逻辑门由酉矩阵表示。最常见的量子门在一个或两个量子位的空间上工作，就像常见的经典逻辑门在一个或两个位上操作一样。

常见量子逻辑门矩阵形式
--------------------------------------

.. |H| image:: images/H.png
   :width: 50px
   :height: 50px

.. |X| image:: images/X.png
   :width: 50px
   :height: 50px

.. |Y| image:: images/Y.png
   :width: 50px
   :height: 50px
   
.. |Z| image:: images/Z.png
   :width: 50px
   :height: 50px

.. |RX| image:: images/X_Theta.png
   :width: 50px
   :height: 50px

.. |RY| image:: images/Y_Theta.png
   :width: 50px
   :height: 50px

.. |RZ| image:: images/Z_Theta.png
   :width: 50px
   :height: 50px

.. |CNOT| image:: images/+-01.png
   :width: 50px
   :height: 50px

.. |CR| image:: images/CR-01.png
   :width: 50px
   :height: 50px

.. |iSWAP| image:: images/iSWAP.png
   :width: 50px
   :height: 50px

.. |Toffoli| image:: images/Toff-01.png
   :width: 50px
   :height: 50px

单比特量子逻辑门：

============================================ ======================= =============================================================================
| |H|                                         | ``Hadamard``              | :math:`\begin{bmatrix} 1/\sqrt {2} & 1/\sqrt {2} \\ 1/\sqrt {2} & -1/\sqrt {2} \end{bmatrix}\quad`
| |X|                                         | ``Pauli-X``               | :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\quad`
| |Y|                                         | ``Pauli-Y``               | :math:`\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}\quad`
| |Z|                                         | ``Pauli-Z``               | :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}\quad`
| |RX|                                        | ``RX``                    | :math:`\begin{bmatrix} \cos(\theta/2) & -1i×\sin(\theta/2) \\ -1i×\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}\quad`
| |RY|                                        | ``RY``                    | :math:`\begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}\quad`
| |RZ|                                        | ``RZ``                    | :math:`\begin{bmatrix} \exp(-i\theta/2) & 0 \\ 0 & \exp(i\theta/2) \end{bmatrix}\quad`
============================================ ======================= =============================================================================

多比特量子逻辑门：

============================================ ======================= ========================================================================================================
| |CNOT|                                      | ``CNOT``                  | :math:`\begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}\quad`
| |CR|                                        | ``CR``                    | :math:`\begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & \exp(i\theta) \end{bmatrix}\quad`
| |iSWAP|                                     | ``iSWAP``                 | :math:`\begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 0 & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}\quad`
| |Toffoli|                                   | ``Toffoli``               | :math:`\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0  \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1  \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ \end{bmatrix}\quad`
============================================ ======================= ========================================================================================================

.. _api_introduction:

QPanda-2把所有的量子逻辑门封装为API向用户提供使用，并可获得QGate类型的返回值。比如，您想要使用Hadamard门，就可以通过如下方式获得：

     .. code-block:: c
          
          QGate h = H(qubit);

可以看到，H函数只接收一个qubit，qubit如何申请会在 :ref:`QuantumMachine` 部分介绍。

再比如，您想要使用RX门，可以通过如下方式获得：

     .. code-block:: c
          
          QGate rx = RX(qubit，PI);

如上所示，RX门接收两个参数，第一个是目标量子比特，第二个偏转角度。您也可以通过相同的方式使用RY，RZ门。

两比特量子逻辑门的使用和单比特量子逻辑门的用法相似，只不过是输入的参数不同，举个使用CNOT的例子：

     .. code-block:: c
          
          QGate cnot = CNOT(control_qubit，target_qubit);

CNOT门接收两个参数，第一个是控制比特，第二个是目标比特。


接口介绍
----------------

在本章的开头介绍过，所有的量子逻辑门都是酉矩阵，那么您也可以对量子逻辑门做转置共轭操作。QGate类型有两个成员函数可以做转置共轭操作：
dagger、setDagger。

setDagger的作用是根据输入参数更新当前量子逻辑门的dagger标记，在计算时计算后端会根据dagger判断当前量子逻辑门是否需要执行转置共轭操作。举个列子：

     .. code-block:: c
          
          QGate h_dagger = H(qubit).setDagger(true);

.. note:: setDagger有一个布尔类型参数，用来设置当前逻辑门是否需要转置共轭操作。

dagger的作用是复制一份当前的量子逻辑门，并更新复制的量子逻辑门的dagger标记。举个例子：

     .. code-block:: c
          
          QGate rx_dagger = RX(qubit,PI).dagger();

除了转置共轭操作，您也可以为量子逻辑门添加控制比特，添加控制比特后，当前量子逻辑门是否执行需要根据控制比特的量子态决定，而控制比特有可能处于叠加态，
所以当前量子逻辑门是否执行，不好说。QGate类型有两个成员函数帮助您添加控制比特：control、setControl。

setControl的作用是给当前的量子逻辑门添加控制比特，例如：

     .. code-block:: c
          
          QGate rx_control = RX(qubit,PI).setControl(qvec);



control的作用是复制当前的量子逻辑门，并给复制的量子逻辑门添加控制比特，例如：

     .. code-block:: c
          
          QGate rx_control = RX(qubit,PI).control(qvec);


.. note:: setControl、control都需要接收一个参数，参数类型为QVec，QVec是qubit的vector。

control的作用是复制当前的量子逻辑门，并给复制的量子逻辑门添加控制比特，例如：

     .. code-block:: c
          
          QGate rx_control = RX(qubit,PI).control(qvec);

.. note:: setControl、control都需要接收一个参数，参数类型为QVec，QVec是qubit的vector。

实例
----------------

以下实例主要是向您展现QGate类型接口的使用方式.

    .. code-block:: c

        #include "QPanda.h"
        using namespace QPanda

        int main(void)
        {
            init(QMachineType::CPU);
            auto q = qAllocMany(3);
            QVec qubits = {q[0],q[1]};
            
            auto prog = CreateEmptyQProg();
            prog << H(q[0])
                 << H(q[1]) 
                 << H(q[0]).dagger()
                 << X(q[2]).control(qubits);
            auto result = probRunTupleList(prog, q);
            for(auto & aiter : result)
            {
                std::cout << aiter.first << " : " << aiter.second << std::endl;
            }

            finalize();
            return 0;
        }

计算结果如下：

    .. code-block:: c
        
        000:0.5
        010:0.5