## QPanda 2

![图片: ](./Documentation/img/1.png)

QPanda2 is an open source quantum computing framework developed by Origin Quantum, which can be used to build, run and optimize quantum algorithms.
QPanda2 is the basic library of a series of software developped by  Origin Quantum, which provides core components for QRunes, Qurator and quantum computing services.

| Linux                | Windows |
|-------------------------|------------------|
[![Build Status](https://travis-ci.org/OriginQ/QPanda-2.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-2)        |    [![Build Status](https://dev.azure.com/yekongxiaogang/QPanda2/_apis/build/status/OriginQ.QPanda-2?branchName=master)](https://dev.azure.com/yekongxiaogang/QPanda2/_build/latest?definitionId=4&branchName=master)   

| C++ Documents         | Python Documents |
|-------------------------|-----------------|
 | [![Documentation Status](https://readthedocs.org/projects/qpanda-tutorial/badge/?version=latest)](https://qpanda-tutorial.readthedocs.io/zh/latest/?badge=latest)      | [![Documentation Status](https://readthedocs.org/projects/pyqpanda-toturial/badge/?version=latest)](https://pyqpanda-toturial.readthedocs.io/zh/latest/?badge=latest)    


## Install for Python

### Python 3.7-3.9

Install using pip:

    pip install pyqpanda
    
### Other versions of Python and C++

If you want to use other versions of Python3 or use C++ API, Compiling from source is recommended. 
Reference to the [Documents for tutorials](https://pyqpanda-tutorial-en.readthedocs.io/en/latest/)

### Python sample code

The following example can be used to construct quantum entanglement in a quantum computer(|0000>+|1111>), measure all qubits and run 1000 times:

    from pyqpanda import *

    qvm = CPUQVM()
    qvm.init_qvm()
    prog = QProg()
    q = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)
    prog << H(q[0])\
        << CNOT(q[0:-1],q[1:])\
        << measure_all(q,c)
    result = qvm.run_with_configuration(prog, c, 1000)
    print(result)
    qvm.finalize()

 Results:
 
    {'0000': 518, '1111': 482}
    
See more [examples](https://github.com/OriginQ/QPanda-Example)。

## Other informations

 - [Official website of Origin Quantum](http://originqc.com.cn/)
 - [OriginQ Cloud](http://www.qubitonline.cn/)
 - [OriginQ Education](https://learn-quantum.com/EDU/index.html)
 - [ReadTheDocs(C++)](https://qpanda-tutorial.readthedocs.io/zh/latest/)
 - [ReadTheDocs(Python)](https://pyqpanda-tutorial-en.readthedocs.io/en/latest/)
 - [QRunes](https://qrunes-tutorial.readthedocs.io/en/latest/)
 - [Qurator-VSCode](https://qurator-vscode.readthedocs.io/zh_CN/latest/)


## About

QPanda is developed by Origin Quantum, which is committed to the development and application of quantum computers, 
It has launched 6-Qubit superconducting quantum chip (KF C6-130) and 2-Qubit semi-conducting quantum chip (XW B2-100).
The goal of the team is to produce more qubit chips in recent years, provide open cloud services, and realize quantum advantages and quantum applications.
The software team underpins the hardware,In addition to QPanda, it has also developed QRunes, Qurator, 
OriginQ Cloud service platform, OriginQ Education cloud and other products.


 ## License
 Apache License 2.0
