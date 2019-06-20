/*! \file Uinteger.h */
#ifndef UNSIGNEDINTEGER_H
#define UNSIGNEDINTEGER_H
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include "ThirdParty/uintwide/generic_template_uintwide_t.h"

using namespace std;
using uint128_t = wide_integer::generic_template::uint128_t;
using uint256_t = wide_integer::generic_template::uint256_t;
using uint512_t = wide_integer::generic_template::uint512_t;
/**
* @namespace QPanda
*/

/**
* @brief  Unsigned integer to binary string
* @ingroup Utilities
* @param[in]  const UnsignedIntegralType & number
* @param[in]  int binary string length
* @return     std::string  unsigned integer string
*/
template <typename UnsignedIntegralType>
std::string integerToBinary(const UnsignedIntegralType &number, int ret_len)
{
    static_assert(((std::numeric_limits<UnsignedIntegralType>::is_integer == true) 
                && (std::numeric_limits<UnsignedIntegralType>::is_signed  == false)),
                    "bad unsigned integral type");

    std::stringstream ss;
    for (int i = ret_len - 1; i > -1; i--)
    {
        ss << ((number >> i) & 1);
    }
    return ss.str();
}

/**
* @brief  Unsigned integer to binary string
* @ingroup  Utilities
* @param[in]  const UnsignedIntegralType & number
* @return     std::string   unsigned integer string
*/
template <typename UnsignedIntegralType>
std::string integerToString(const UnsignedIntegralType &number)
{
    static_assert(((std::numeric_limits<UnsignedIntegralType>::is_integer == true)
                && (std::numeric_limits<UnsignedIntegralType>::is_signed  == false)),
                   "bad unsigned integral type");

    std::stringstream ss;
    ss << number;
    return ss.str();
}


/**
* @brief  Get quantum state dec index in pmeasure
* @param[in]  const UnsignedIntegralType & num1
* @param[in]  const UnsignedIntegralType & num2
* @param[in]  std::vector<size_t> qvec
* @param[in]  size_t binary string length
* @return     Unsigned Integral Type
*/
template <typename UnsignedIntegralType>
UnsignedIntegralType getDecIndex(const UnsignedIntegralType &num1,
                                 const UnsignedIntegralType &num2,
                                 std::vector<size_t> qvec,
                                 size_t len)
{
    static_assert(((std::numeric_limits<UnsignedIntegralType>::is_integer == true)
                && (std::numeric_limits<UnsignedIntegralType>::is_signed  == false)),
                   "bad unsigned integral type");

    using uint_type = UnsignedIntegralType;

    uint_type index = 0;
    size_t pos1{ 0 }, pos2{ 0 };
    for (size_t i = 0; i < len; ++i)
    {
        if (qvec.end() == find(qvec.begin(), qvec.end(), i))
        {
            auto bit = (num2 & (1 << pos1)) >> pos1;
            index += ((uint_type)bit << i);
            pos1++;
        }
        else
        {
            auto bit = (num1 & (1 << pos2)) >> pos2;
            index += ((uint_type)bit << i);
            pos2++;
        }
    }

    return index;
}

#endif //!UNSIGNEDINTEGER_H