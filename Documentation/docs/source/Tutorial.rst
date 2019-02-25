入门教程
==============

构建QPanda-2
------------------
| 为了方便用户使用，我们支持在Windows、Linux、MacOS下构建QPanda-2。用户可以通过CMake的方式来构建QPanda-2。

Windows
***************
| 在Windows构建QPanda-2。用户首先需要保证在当前主机下安装了CMake环境和C++编译环境，用户可以通过Visual Studio和MinGW方式编译QPanda-2。

1. 使用Visual Studio
***************************

| 使用Visual Studio编译QPanda-2,只需要安装Visual Studio，并需要在组件中安装CMake组件。安装完成之后，用Visual Studio打开QPanda-2文件夹，即可使用CMake编译QPanda-2。

2. 使用minGW
********************

| 使用minGW编译QPanda-2，需要自行搭建CMake和minGW环境，用户可自行在网上查询环境搭建教程。（注意： minGW需要安装64位版本）
| CMake+minGW的编译命令如下：

1. 在QPanda-2根目录下创建build文件夹
2. 进入build文件夹，打开cmd
3. 由于MinGW对CUDA的支持存在一些问题，所以在编译时需要禁掉CUDA，输入以下命令：

.. code-block:: c

    cmake -G"MinGW Makefiles" -DFIND_CUDA=OFF ..
    mingw32-make

Linux 和MacOS
******************

| 在Linux和MacOS下构建QPanda-2，命令是一样的。
| 编译步骤如下：

1. 进入QPanda-2根目录
2. 输入以下命令：

.. code-block:: c

    mkdir -p build
    cd build
    cmake ..
    make

cmake链接QPanda-2库的方法
******************************

1. 进入QPanda-2根目录
2. 在cmakeLists.txt下面输入以下内容：

.. code-block:: c

    add_executable(QvmTest test.cpp)
    target_link_libraries(QvmTest QPanda2.0)

.. note:: 
    - *add_executable* 添加可执行程序
    - *QvmTest*  可执行程序
    - *test.cpp*  测试cpp
    - *target_link_libraries* 链接库
    - *QPanda2.0*  QPanda2库


入门指南
--------------

| QPanda-2构建完成后，我们就可以开始量子编程之旅了，首先先实现一个小的示例程序。

.. code-block:: c

    #include "QPanda.h"
    USING_QPANDA

    int main(void)
    {
        init();
        auto c = cAlloc();
        auto qvec = qAllocMany(5);
        c.setValue(0);
        QProg while_prog;
        while_prog<<H(qvec[c])<<(c=c+1);
        auto qwhile = CreateWhileProg(c<5,&while_prog);
        QProg prog;
        prog<<qwhile;

        load(prog);
        run();
        auto result = getProbDict(qvec);

        for(auto & aiter : result)
        {
            std::cout << aiter.first << " : " << aiter.second << std::endl;
        }

        finalize();
        return 0;
    }

.. note::
    - *init* 初始化
    - *cAlloc* 申请一个量子表达式
    - *qAllocMany* 申请多个量子比特
    - *setValue* 设置量子表达式的值
    - *CreateWhileProg* 创建一个QWhileProg
    - *load* 加载量子程序
    - *run* 运行量子程序
    - *getProbDict* 概率测量的方式获取量子程序运行结果
    - *finalize* 释放资源
    - 上面的示例程序主要是对从量子虚拟机申请的5个量子比特做Hadamard门操作，然后通过概率测量的方式获取计算结果并输出。该示例程序体现了QPanda-2的部分功能特征，我们会在以下章节中详细介绍QPanda-2的使用。
