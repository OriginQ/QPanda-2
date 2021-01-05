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
#include "QAlg/Base_QCircuit/base_circuit.h"

QPANDA_BEGIN

/**
* @brief  Shor Algorthm
* @ingroup QAlg
*/
class ShorAlg
{
public:
    /**
    *@param[in] target the number to initialize the large number
    */
    ShorAlg(int target);

    /**
    * @param[in] set the smallest base, default = 2
    */
    void set_decomposition_starter(int smallest_base);

    /**
    *@brief execute the prime factorization for target number
    *@return whether the process succeed
    */
    bool exec();
   
    /**
    * @brief get the decomposition result
    * @return get the decomposition result
    */
    std::pair<int, int> get_results();

private:
    /**
    *@param[in] m_factor_1,m_factor_2  2 prime factors to be returned
    *@param[in] starter the low bound of the period finding loop, default = 2
    *@param[in] m_target_Num the input large number
    */
    int m_factor_1 = 1;
    int m_factor_2 = 0;
    int starter = 2;
    int m_target_Num;

private:
    /**
    *@brief Greatest Common Divisor of a and b.
    */
    static int _gcd(int a, int b);

    /**
    *@param[in] base,target  the base number and the target number
    *@brief find the smallest r such that base^r module target = 1.
    */
    int _period_finding(int base, int target);

    /**
    * @return continuous fraction expansion(CFE) for result/2^([log_2^{target}]*2).
    */
    int _continuous_frac_expan(int target, int result);

    /**
    *@return CFE for the max results and get the lowest common multiple(LCM) for them.
    */
    int _measure_result_parse(int target, vector<int> max_result);
};

/**
*@brief  Shor Algorthm Interface Function
*@ingroup QAlg
*@param[in] target the input large number
*@return whether the process is sucessful; p,q the prime factorization results
*/
std::pair<bool, std::pair<int, int>> Shor_factorization(int target);


QPANDA_END

#endif