/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QAOATestFactory.h

Author: LiYe
Created in 2018-11-12


*/

#ifndef QAOATESTFACTORY_H
#define QAOATESTFACTORY_H

#include <memory>
#include <string>

namespace QPanda
{

    class AbstractQAOATest;
    class QAOATestFactory
    {
        public:
            QAOATestFactory() = delete;
            static std::unique_ptr<AbstractQAOATest>
                makeQAOATest(const std::string &testname);
    };

}

#endif // QAOATESTFACTORY_H
