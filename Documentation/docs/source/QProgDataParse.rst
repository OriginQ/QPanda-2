解析量子程序二进制文件
=====================

简介
--------------

将 :ref:`QProgStored` 方法存储的量子程序的二进制文件解析成量子程序

接口介绍
--------------

.. cpp:function:: bool binaryQProgFileParse(QProg &prog, const std::string &filename = DEF_QPROG_FILENAME)
    
    将存储的量子程序二进制文件解析成量子程序

    **参数**  
        - prog 量子程序
        - filename 文件名    

使用实例
---------

    .. code-block:: c
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            init();
            auto prog = CreateEmptyQProg();
            bool is_success = binaryQProgFileParse(prog);
            if (!is_success)
            {
                std::cout << "parse failure!" << std::endl;
            }
            else
            {
                std::cout << "parse success!" << std::endl;
            }

            finalize();
            return 0;
        }
