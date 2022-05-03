# QPanda编译后处理工具集



## stubgen工具

### 概述

该工具主要用于为使用pybind11封装的pyQPanda接口生成`.pyi`的stub文件，用于IDE的intelliSENCE（自动补全，参数提示等等）。

该工具基于[mypy](https://github.com/python/mypy)改造而来，主要修改其stubgenc.py文件中`generate_c_function_stub`和`generate_c_type_stub`函数，增强其功能。现已将关键流程中的函数独立出来，集中在stubgen.py和stubgencmod.py中。

主要修改：

1. 将pyQPanda包中的类，函数doc，添加到pyi文件中相应的注释位置
2. 将pyQPanda包的doc中保存的形参默认值填写到pyi文件中（通过python语法解析器工具[parso](https://github.com/davidhalter/parso)来将函数签名解析为语法树，从中提取形参的默认值字符串）

### 使用方法

主要的使用方法可参考mypy的stubgen工具，**使用`-h`参数查看工具的参数选项**。

安装依赖

```
pip install -r requirement.txt
```

假设待处理的python二进制包为test.pyd或test.so，放在当前目录下，二进制包中定义的模块名为test

```
python stubgen/stubgen.py -m test
```

默认会在当前目录的out文件夹下生成test.pyi。将test.pyi文件拷贝到test.pyd同一目录下，即可让IDE识别到python包接口提示。