pyQPanda介绍
============
----

**pyQPanda是python版本的QPanda**。

我们通过pybind11工具，以一种直接和简明的方式，对QPanda2中的函数、类进行封装，并且提供了几乎完美的映射功能。

封装部分的代码在QPanda2编译时会生成为动态库，从而可以作为python的包引入。在windows下是后缀名为pyd的文件，在linux下是.so，在macOS下是.dylib。

如何安装
>>>>>>>>
----

如果你已经安装好了python环境和pip工具，可以直接通过下面的命令安装pyQPanda

    .. code-block:: python

        pip install pyqpanda

详细介绍
>>>>>>>>
----

关于pyQPanda接口的详细介绍请参考

     - :ref:`pyQPanda-Core` 
     - :ref:`pyQPanda-ClassicalQuantumMachine` 
     - :ref:`pyQPanda-OtherQuantumMachine` 
     - :ref:`pyQPanda-Utilities` 
