[![Build Status](https://travis-ci.org/OriginQ/QPanda-SDK.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-SDK)
## QPanda
Qpanda is a software used to deal with quantum circuits and experiments on various quantum computers.

There are three processes in QPanda: initialization, compilation, and running.

**Initialization**:The initialization allows users to design different quantum circuits to deal with the corresponding problems.

**compilation**:Compilation allows users to rewrite them to run on different backends (such as simulators, quantum chips, quantum chips of different companies, etc.).

**Running**: that is, the process of collecting results(classical information), depending on the design of the problem to do the corresponding operation. Some problems may depend on the results of a previous quantum program before they can be executed, and so on.

## The Design Ideas of QPanda


The design of **QPanda** is forward-looking, considering that quantum computing will flourish and be widely applied in the future. So QPanda did the following consideration when it was designed:

1.  Full series compatibility.

2.  Standard architecture.

3.  Standardized quantum machine model.

## The Project Includes：

-   **QPanda SDK**：

C++ is used as the host language to compiling quantum programs in QPanda SDK. It enables users to connect and execute quantum programs conveniently.

-   **QRunes**：

Qrunes is a set of quantum computing instructions developed by the Origin quantum team.

-   **QRunes Generator**：

Qrunes Generator is a C + + library that supports generating Qrunes directives in function calls.

-   **[QPanda Documentation](./QPanda-2.0.Documentation\README.md)**：

 Instructions for QPanda software. It includes algorithm summary, corresponding quantum circuits, corresponding QPanda code, etc., aiming at guiding users to use QPanda correctly and quickly.

 ## License
 Apache License 2.0
