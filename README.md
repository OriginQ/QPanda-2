[![Build Status](https://travis-ci.org/OriginQ/QPanda-SDK.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-SDK)
## QPanda 2.0



![图片: ](https://images-cdn.shimo.im/GCEO3o2i9dcRrfV0/68747470733a2f2f696d616765732d63646e2e7368696d6f2e696d2f4274755036615665306f6f326a526c5a2f696d6167652e706e67.png)

Qpanda 2.0 is a quantum software development kit used to deal with quantum circuits and experiments on various quantum computers launched by the Origin Quantum Computing Company. It supports mainstream quantum logic gate operation and can be adapted to a variety of quantum chips, and the targeted optimization of quantum programs under different platforms.

QPanda 2.0 uses the C++ language as the host language and supports quantum languages ​​written in QRunes and QASM. Currently, QPanda 2.0 supports up to 32-qubits in the local simulation. It integrates a quantum virtual machine(QVM) and encapsulates the mainstream quantum algorithms.

Besides, quantum programs can make logical decisions where we have increased the idea of ​​control flow(q-if). Thus in line with the programming specifications of high-level languages.

There are three processes in QPanda 2.0: **initialization**, **compilation** and **running**.

**Initialization**:

The initialization allows users to design different quantum circuits to deal with the corresponding problems.

**Compilation**:

Compilation allows users to rewrite them to run on different backends (such as simulators, quantum chips, quantum chips of different companies, and such).

**Running**:

that is the process of collecting results(classical information), depending on the design of the problem to do the corresponding operation.

## The Design Ideas of QPanda


The design of **QPanda 2.0** is forward-looking, considering that quantum computing will flourish and be widely applied in the future. So QPanda did the following consideration when it was designed:

1.  Full series compatibility.

2.  Standard architecture.

3.  Standardized quantum machine model.

## The Project Includes：

![图片: ](https://images-cdn.shimo.im/j71VAaimgHkKWXEW/image.png)

-   **QPanda SDK**：

C++ is used as the host language for compiling quantum programs in QPanda 2.0 SDK. It enables users to connect and execute quantum programs conveniently.

-   **QRunes**：

QRunes is a set of quantum computing instructions developed by the Origin quantum team.

-   **QRunes(QASM) Generator**：

Qrunes Generator is a C + + library that supports generating Qrunes(QASM) directives in function calls.

-   **[QPanda Documentation](./QPanda-2.0.Documentation/README.md)**：
The QPanda 2.0 documentation is detailed documentation. It includes algorithm summary, corresponding quantum circuits, and QPanda 2.0 code. Aiming at guiding users to use QPanda correctly and quickly.

 ## License
 Apache License 2.0
