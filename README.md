[![Build Status](https://travis-ci.org/OriginQ/QPanda-SDK.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-SDK)

Qpanda is a software used to deal with quantum circuits and experiments on various quantum computers.

There are three processes in QPanda: initialization, compilation, and running.

The initialization allows users to design different quantum circuits to deal with the corresponding problems.

Compilation allows users to rewrite them to run on different backends (such as simulators, quantum chips, quantum chips of different companies, etc.).

Running, that is, the process of collecting results, depending on the design of the problem to do the corresponding storage or transformation of the results.Some problems may depend on the results of a previous quantum program before they can be executed, and so on.

## The Design Ideas of QPanda


The design ideas of Qpanda are as follows:

1.  Full series compatibility.

2.  Standard architecture.

3.  Standardized quantum machine model.

## The QPanda Project Includes：

-   **QPanda SDK**：

C++ is used as the host language to compiling quantum programs in QPanda SDK. It enables users to connect and execute quantum programs conveniently.

-   **QRunes**：

Qrunes is a set of quantum computing instructions developed by the Origin quantum team.

-   **QRunes Generator**：

Qrunes Generator is a C + + library that supports generating Qrunes directives in function calls.

-   **QPanda Documentation**：

 Instructions for QPanda software. It includes algorithm summary, corresponding quantum circuits, corresponding QPanda code, etc., aiming at guiding users to use QPanda correctly and quickly.

 ## License
 Apache License 2.0
