[![Build Status](https://travis-ci.org/OriginQ/QPanda-2.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-2)
[![Documentation Status](https://readthedocs.org/projects/qpanda-2/badge/?version=latest)](https://qpanda-tutorial.readthedocs.io/zh/latest/)

## QPanda 2


QPanda (**Q**uantum **P**rogramming **A**rchitecture for **N**ISQ **D**evice **A**pplications) is a library for quantum computing which can be applied for realizing various quantum algorithms. QPanda is mainly written in C++, and can be extended to Python.

## Documentation
Documentation is hosted at https://qpanda-tutorial.readthedocs.io/zh/latest/

## Build your first quantum program with QPanda

### C++ Version

    #include "QPanda.h"
    #include <stdio.h>
    using namespace QPanda;

    init(QuantumMachine_type::CPU);
    QProg prog;
    auto q = qAllocMany(2);
    auto c = cAllocMany(2);
    prog << H(q[0])
         << CNOT(q[0],q[1])
         << MeasureAll(q, c);
    
    auto results = runWithConfiguration(prog, c, 1000);
    for (auto result : results){
        printf("%s : %d\n", result.first.c_str(), result.second);
    }
    finalize()

### Python Version
    from pyqpanda import *
    init(QuantumMachine_type.CPU)
    prog = QProg()
    q = qAlloc_many(2)
    c = cAlloc_many(2)
    prog.insert(H(q[0]))
    prog.insert(CNOT(q[0],q[1]))
    prog.insert(measure_all(q,c))
    result = run_with_configuration(prog, cbit_list = c, shots = 1000)
    print(result)
    finalize()

## The Design Ideas of QPanda 2

The design of **QPanda 2** is forward-looking, considering that quantum computing will flourish and be widely applied in the future. So QPanda 2 did the following consideration when it was designed:

- Full series compatibility.

- Standard architecture.

- Standardized quantum machine model.

## Installation and configuration

-   **[See Documentation！](https://qpanda-tutorial.readthedocs.io/zh/latest/)**

 ## License
 Apache License 2.0

 Copyright (c) 2017-2019 By Origin Quantum Computing. All Right Reserved.
