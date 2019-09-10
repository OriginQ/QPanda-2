/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ChemiQTest.h

Author: LiYe
Created in 2018-11-14


*/

#ifndef CHEMIQTEST_H
#define CHEMIQTEST_H

#include "TestInterface/AbstractTest.h"

namespace QPanda
{

    class ChemiQ;
    class ChemiQTest : public AbstractTest
    {
        SINGLETON_DECLARE(ChemiQTest)
    public:
        virtual bool exec(rapidjson::Document &doc);
    private:
        void setOptimizerPara(
            ChemiQ &chemiq,
            const rapidjson::Value& value) const;
        void setChemiQPara(
                ChemiQ &Chemiq,
                const rapidjson::Value &doc) const;
        QMoleculeGeometry getProblem(const rapidjson::Value &value) const;
        QAtomsPosGroup get2AtomPosGroup(const rapidjson::Value &value) const;
        QAtomsPosGroup getNormalAtomPosGroup(const rapidjson::Value &value) const;
        vector_d get2AtomDistances(const rapidjson::Value &value) const;
        QPosition getPos(const rapidjson::Value &value) const;
        //bool writeResultToFile(const std::string &filename,
        //                       const ChemiQ &ChemiQ,
        //                       const rapidjson::Value &value) const;
        ChemiQTest();
    private:
        QMoleculeGeometry m_molecule_geometry;
    };

}

#endif // CHEMIQTEST_H
