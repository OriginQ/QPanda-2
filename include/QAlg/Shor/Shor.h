/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Shor.h

Author: LiuYan
Created in 2020-08-07

*/

#ifndef  SHOR_H
#define  SHOR_H

#include "Core/Core.h"
#include "QAlg/ArithmeticUnit/ArithmeticUnit.h"
#include "QAlg/Base_QCircuit/QFT.h"

QPANDA_BEGIN

/**
* @brief  Shor Algorthm
* @ingroup QAlg
* @ingroup Grover_Algorithm
* @param[in] target the large number to be decomposed
* @return 2 prime factors
* @note
*/
class ShorAlg
{
public:
    /**
    *@m_target_Num the input large number
    *@m_s1,m_s2  2 prime factors to be returned
    */
    int m_s1 = 1, m_s2 = 0, m_target_Num;

    /**
    *@initialize the large number and get the result.
    *@target the input large number
    */
    ShorAlg(int target);
    /**
    *@a convenient interface to do prime factorization for any number.
    *@p,q  2 variables to store result
    */
    void get_Shor_result(int target, int &p, int &q);
private:
    /**
    *@Greatest Common Divisor of a and b.
    */
    static int _gcd(int a, int b);
    /**
    *@find the smallest r such that base^r mod target = 1.
    */
    int _perid_finding(int base, int target);
    /**
    *@continuous fraction expansion(CFE) for result/2^([log_2^{target}]*2).
    */
    int _continuous_frac_expan(int target, int result);
    /**
    *@CFE for the max results and get the lowest common multiple(LCM) for them.
    */
    int _measure_result_parse(int target, vector<int> max_result);
};

QPANDA_END

#endif