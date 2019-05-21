系统配置和安装
=========================

.. _QPanda2: https://qpanda-2.readthedocs.io/zh_CN/latest/
为了兼容 \ **高效**\与\ **便捷**\，QPanda2提供了C++ 和 Python两个版本，本文中主要介绍python版本的使用。
如要了解和学习C++版本的使用请移步 QPanda2_。

我们通过pybind11工具，以一种直接和简明的方式，对QPanda2中的函数、类进行封装，并且提供了几乎完美的映射功能。
封装部分的代码在QPanda2编译时会生成为动态库，从而可以作为python的包引入。

系统配置
>>>>>>>>>>>>

pyqpanda是以C++为宿主语言，其对系统的环境要求如下：

.. list-table::

    * - software
      - version
    * - GCC
      - >= 5.4.0 
    * - Python
      - >= 3.6.0  


下载pyqpanda
>>>>>>>>>>>>>>>>>

如果你已经安装好了python环境和pip工具， 在终端或者控制台输入下面命令：

    .. code-block:: python

        pip install pyqpanda

.. note:: 在linux下若遇到权限问题需要加 ``sudo``

