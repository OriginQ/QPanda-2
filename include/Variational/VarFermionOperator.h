/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Author: LiYe
Created in 2019-01-24

*/

#ifndef VARFERMIONOPERATOR_H
#define VARFERMIONOPERATOR_H

#include "Variational/var.h"
#include "Variational/utils.h"
#include "QAlg/Components/Operator/FermionOperator.h"

QPANDA_BEGIN

using VarFermionOperator = FermionOp<complex_var>;

template<>
VarFermionOperator::FermionOp(double value)
{
    Variational::var v(value);
    m_data.push_back(std::make_pair(FermionPair(), complex_var(v, 0)));
}

template<>
std::string VarFermionOperator::toString() const
{
    std::string str = "{";
    for (auto iter = m_data.begin(); iter != m_data.end(); iter++)
    {
        str += "\n";

        auto pair = iter->first;
        auto value = iter->second;

        str += "\"" + pair.second + "\" : [var type]";
    }

    if (!m_data.empty())
    {
        str += "\n";
    }

    str += "}";
    return str;
}

template<>
void VarFermionOperator::reduceDuplicates()
{
    std::map<std::string, complex_var> data_map;
    std::map<std::string, OrbitalActVec> term_map;

    for (auto iter = m_data.begin(); iter != m_data.end(); iter++)
    {
        auto pair = iter->first;
        complex_var value = iter->second;
        OrbitalActVec oa_vec = pair.first;
        std::string str = pair.second;

        auto result = data_map.find(str);
        if (result != data_map.end())
        {
            result->second.first = result->second.first + value.first;
            result->second.second = result->second.second + value.second;
        }
        else
        {
            data_map.insert(std::make_pair(str, value));
            term_map.insert(std::make_pair(str, oa_vec));
        }
    }

    FermionData fermion_data;
    for (auto iter = data_map.begin(); iter != data_map.end(); iter++)
    {
        FermionPair pair;
        pair.first = term_map[iter->first];
        pair.second = iter->first;

        fermion_data.push_back(std::make_pair(pair, iter->second));
    }

    m_data = std::move(fermion_data);
}

template<>
VarFermionOperator VarFermionOperator::normal_ordered_ladder_term(
        OrbitalActVec &term,
        std::string &term_str,
        complex_var &coefficient)
{
    VarFermionOperator op;
    for (size_t i = 1; i < term.size(); i++)
    {
        for (size_t j = i; j > 0; j--)
        {
            auto right_operator = term[j];
            auto left_operator = term[j - 1];

            // Swap operators if raising on right and lowering on left.
            if (right_operator.second && (!left_operator.second))
            {
                term[j - 1] = right_operator;
                term[j] = left_operator;
                term_str = OrbitalActVec2String(term);
                coefficient = coefficient * -1.0;

                // Replace a a^\dagger with 1 + parity*a^\dagger a
                // if indices are the same.
                if (right_operator.first == left_operator.first)
                {
                    OrbitalActVec a;
                    a.resize(j-1);
                    memcpy(a.data(), term.data(), (j-1)*sizeof(OrbitalAct));

                    OrbitalActVec b;
                    b.resize(term.size()-j-1);
                    memcpy(b.data(), term.data()+j+1,
                           b.size()*sizeof(OrbitalAct));

                    a.insert(a.end(), b.begin(), b.end());
                    std::string a_str = OrbitalActVec2String(a);
                    complex_var tmp_coef = coefficient*-1.0;
                    // Recursively add the processed new term.
                    op += normal_ordered_ladder_term(a, a_str, tmp_coef);
                }
            }
            // Handle case when operator type is the same.
            else if (right_operator.second == left_operator.second)
            {
                // If same two Fermionic operators are repeated,
                // evaluate to zero.
                if (right_operator.first == left_operator.first)
                {
                    return op;
                }
                // Swap if same ladder type but lower index on left.
                else if (right_operator.first > left_operator.first)
                {
                    term[j - 1] = right_operator;
                    term[j] = left_operator;
                    term_str = OrbitalActVec2String(term);
                    coefficient = coefficient * -1;
                }
            }
        }
    }

    FermionPair item{term, term_str};
    FermionData data;
    data.push_back(std::make_pair(item, coefficient));
    op += VarFermionOperator(data);

    return op;
}


QPANDA_END
#endif // VARFERMIONOPERATOR_H
