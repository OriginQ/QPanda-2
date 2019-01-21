/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

VQETest.h

Author: LiYe
Created in 2018-11-14


*/

#ifndef VQETEST_H
#define VQETEST_H

#include "TestInterface/AbstractTest.h"

namespace QPanda
{

    class VQE;
    class VQETest : public AbstractTest
    {
        SINGLETON_DECLARE(VQETest)
    public:
        virtual bool exec(rapidjson::Document &doc);
    private:
        bool mpiTest(
                VQE &vqe,
                const std::string &filename,
                const rapidjson::Value &value) const;
        void setVQEPara(
                VQE &vqe,
                const rapidjson::Value &value,
                size_t rank = 0,
                size_t size = 0) const;
        QMoleculeGeometry getProblem(const rapidjson::Value &value) const;
        QAtomsPosGroup get2AtomPosGroup(const rapidjson::Value &value) const;
        vector_d get2AtomDistances(const rapidjson::Value &value) const;
        QPosition getPos(const rapidjson::Value &value) const;
        bool writeResultToFile(const std::string &filename,
                               const VQE &vqe,
                               const rapidjson::Value &value) const;
        std::pair<size_t, size_t> getIndexAndCount(
                size_t length,
                size_t rank,
                size_t size) const;
        VQETest();
    };

}

#endif // VQETEST_H
