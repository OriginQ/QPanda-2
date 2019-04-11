编译、安装和使用
==============

编译
------------------
我们支持在Windows、Linux、MacOS下构建QPanda-2。用户可以通过CMake的方式来构建QPanda-2。

Windows
***************
在Windows构建QPanda-2。用户首先需要保证在当前主机下安装了CMake环境和C++编译环境，用户可以通过Visual Studio和MinGW方式编译QPanda-2。

1. 使用Visual Studio
***************************
使用Visual Studio编译QPanda-2,只需要安装Visual Studio，并需要在组件中安装CMake组件。安装完成之后，用Visual Studio打开QPanda-2文件夹，即可使用CMake编译QPanda-2。

.. image:: images/VS编译.png
    :align: center   

2. 使用minGW
********************

使用minGW编译QPanda-2，需要自行搭建CMake和minGW环境，用户可自行在网上查询环境搭建教程。（注意： minGW需要安装64位版本）

CMake+minGW的编译命令如下：

1. 在QPanda-2根目录下创建build文件夹
2. 进入build文件夹，打开cmd
3. 由于MinGW对CUDA的支持存在一些问题，所以在编译时需要禁掉CUDA，输入以下命令：

.. code-block:: c

    cmake -G"MinGW Makefiles" -DFIND_CUDA=OFF -DCMAKE_INSTALL_PREFIX=C:/cmake/install ..
    mingw32-make

Linux 和MacOS
******************

在Linux和MacOS下编译QPanda-2，命令是一样的。

编译步骤如下：

1. 进入QPanda-2根目录
2. 输入以下命令：

.. code-block:: c

    mkdir -p build
    cd build
    cmake ..
    make

如果有需求，用户通过命令修改QPanda-2的安装路径，配置方法如下所示：

.. code-block:: c

    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr ..
    make


安装
------------------

Windows
***************

1. 使用Visual Studio
***************************
在QPanda-2编译完成后，用户可以安装QPanda-2，Visual Studio的安装方式很简单，只需要在Cmake菜单中选择安装即可。

.. image:: images/VS安装.png
    :align: center   


QPanda-2会安装在用户在CMakeSettings.json中配置的安装目录下。安装成功后会在用户配置的的目录下生成install文件夹，里面安装生成include和lib文件。如果有需求，用户可以在Visual Studio的CMakeSettings.json配置文件修改QPanda-2的安装路径。生成CMakeSettings.json的方法如下图所示：

.. image:: images/VS修改CMake配置.png
    :align: center   

修改QPanda-2的安装路径如下图所示：


.. image:: images/VS修改安装路径.png
    :align: center   

参数修改完成后，cmake选项下执行安装，Qpanda-2的lib库文件和include头文件会安装到用户指定的安装位置。(注意：需先进行编译成功后才能进行安装)

2. 使用minGW
********************

在QPanda-2编译完成后，用户可以安装QPanda-2，安装命令如下：

.. code-block:: c

    mingw32-make install

Linux 和MacOS
******************

在Linux和MacOS下安装命令QPanda-2，命令是一样的，安装命令如下：

.. code-block:: c

    make install

使用
------------------
