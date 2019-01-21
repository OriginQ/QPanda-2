#ifndef _CLASSICAL_CONDITION_TEST_H
#define _CLASSICAL_CONDITION_TEST_H
#include "gtest/gtest.h"
#include "QPanda.h"
class ClassicalConditionTest : public ::testing::Test
{
public:

    QPanda::QuantumMachine * m_qvm;

    ClassicalConditionTest()
    {
        m_qvm = QPanda::initQuantumMachine();
    }

    virtual void SetUp()
    {

    }
};
#endif