/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

FermionOperator.h

Author: LiYe
Created in 2019-01-18


*/
#ifndef FERMIONOPERATOR_H
#define FERMIONOPERATOR_H

#include <map>
#include <unordered_map>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QString.h"
#include "QAlg/DataStruct.h"

QPANDA_BEGIN

/*
 * # Denotation
 *
 * Annihilation and Creation Operators
 *
 * Annihilation: x denotes $a_x$
 * Creation: x+ denotes $a_x^\dagger$
 *
 * For example:
 *
 * "1+ 3 5+ 1"
 * represents
 * a_1^\dagger a_3 a_5^\dagger a_1
 *
 *
 * # Rearrange Rule
 *
 * Different Number
 *
 * "1 2" = -1 * "2 1"
 *
 * "1+ 2+" = -1 * "2+ 1+"
 *
 * "1+ 2" = -1 * "2 1+"
 *
 * Same number
 *
 * "1 1+" = 1 - "1+ 1"
 *
 * "1+ 1+" = 0
 *
 * "1 1" = 0
 *
 */

/*
 * first: orbital index; second: action, true create, false annihilation
 */
using OrbitalAct = std::pair<size_t, bool>;
using OrbitalActVec = std::vector<OrbitalAct>;
using FermionPair = std::pair<OrbitalActVec, std::string>;

template<class T>
class FermionOp
{
public:
    using FermionItem = std::pair<FermionPair, T>;
    using FermionData = std::vector<FermionItem>;

    using FermionMap = std::map<std::string, T>;
public:
    FermionOp(){}
    FermionOp(double value)
    {
        insertData("", T(value));
    }

    FermionOp(const T &value)
    {
        insertData("", value);
    }

    FermionOp(const std::string &key, const T &value)
    {
        insertData(key, value);
    }

    FermionOp(const FermionMap &map)
    {
        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            insertData(iter->first, iter->second);
        }

        reduceDuplicates();
    }

    FermionOp(FermionData &&fermion_data):
        m_data(std::move(fermion_data))
    {
        reduceDuplicates();
    }

    FermionOp(const FermionData &fermion_data):
        m_data(fermion_data)
    {
        reduceDuplicates();
    }

    FermionOp(FermionOp &&op):
        m_data(std::move(op.m_data))
    {
    }

    FermionOp(const FermionOp &op):
        m_data(op.m_data)
    {
    }

    FermionOp &operator = (const FermionOp &op)
    {
        m_data = op.m_data;
        return *this;
    }

    FermionOp &operator = (FermionOp &&op)
    {
        m_data = std::move(op.m_data);
        return *this;
    }

    /*
     * Compute and return the normal ordered form of a FermionOperator
     *
     * In our convention, normal ordering implies terms are ordered
     * from highest tensor factor (on left) to lowest (on right).In
     * addition:
     *
     * a^\dagger comes before a
     *
     */
    FermionOp normal_ordered()
    {
        auto data = FermionOp();

        auto tmp_data = m_data;
        for (auto & i: tmp_data)
        {
            data += normal_ordered_ladder_term(
                        i.first.first,
                        i.first.second,
                        i.second);
        }

        return data;
    }

    bool isEmpty() { return m_data.empty(); }
    std::string toString() const
    {
        std::string str = "{";
        for (auto iter = m_data.begin(); iter != m_data.end(); iter++)
        {
            str += "\n";

            auto pair = iter->first;
            auto value = iter->second;

            str += "\"" + pair.second + "\" : ";
            if (fabs(value.real()) < m_error_threshold)
            {
                str += std::to_string(value.imag()) + "i";
            }
            else if (fabs(value.imag()) < m_error_threshold)
            {
                str += std::to_string(value.real());
            }
            else
            {
                if (value.imag() < 0)
                {
                    str += "(" + std::to_string(value.real()) +
                        std::to_string(value.imag()) + "i)";
                }
                else
                {
                    str += "(" + std::to_string(value.real()) + "+" +
                        std::to_string(value.imag()) + "i)";
                }
            }
        }

        if (!m_data.empty())
        {
            str += "\n";
        }

        str += "}";
        return str;
    }

    void setAction(char create, char annihilation)
    {
        if (create != annihilation)
        {
            m_action = std::make_pair(create, annihilation);
        }
    }

    void setErrorThreshold(double threshold)
    {
        m_error_threshold = threshold;
    }

    double error_threshold() const { return m_error_threshold; }

    FermionData data()  { return m_data; }

    FermionOp  operator + (const FermionOp &rhs) const
    {
        FermionData m1 = m_data;
        FermionData m2 = rhs.m_data;
        m1.insert(m1.end(), m2.begin(), m2.end());

        FermionOp tmp(m1);
        return tmp;
    }

    FermionOp  operator - (const FermionOp &rhs) const
    {
        FermionOp tmp_fermion(rhs);
        tmp_fermion *= -1.0;

        return *this + tmp_fermion;
    }

    FermionOp  operator * (const FermionOp &rhs) const
    {
        FermionData tmp_data;
        for (auto i = m_data.begin(); i != m_data.end(); i++)
        {
            for (auto j = rhs.m_data.begin(); j != rhs.m_data.end(); j++)
            {   auto i_f = i->first;
                auto i_s = i->second;
                auto j_f = j->first;
                auto j_s = j->second;

                i_f.first.insert(i_f.first.end(),
                                 j_f.first.begin(), j_f.first.end());
                i_f.second += " " + j_f.second;

                tmp_data.push_back(std::make_pair(i_f, i_s*j_s));
            }
        }

        FermionOp tmp_fermion(std::move(tmp_data));
        tmp_fermion.reduceDuplicates();

        return tmp_fermion;
    }

    FermionOp &operator +=(const FermionOp &rhs)
    {
        auto &cdata = rhs.m_data;

        m_data.insert(m_data.end(), cdata.begin(), cdata.end());
        reduceDuplicates();

        return *this;
    }

    FermionOp &operator -=(const FermionOp &rhs)
    {
        FermionOp tmp(rhs);
        tmp *= -1.0;

        m_data.insert(m_data.end(), tmp.m_data.begin(), tmp.m_data.end());
        return *this;
    }

    FermionOp &operator *=(const FermionOp &rhs)
    {
        FermionOp tmp(std::move(m_data));
        auto result = tmp * rhs;

        result.reduceDuplicates();
        m_data = std::move(result.m_data);

        return *this;
    }

    friend FermionOp operator + (const complex_d &lhs,
                                        const FermionOp &rhs)
    {
        return rhs + lhs;
    }

    friend FermionOp operator - (const complex_d &lhs,
                                        const FermionOp &rhs)
    {
        return rhs*-1.0 + lhs;
    }

    friend FermionOp operator * (const complex_d &lhs,
                                        const FermionOp &rhs)
    {
        return rhs * lhs;
    }

    friend std::ostream  &operator <<(std::ostream &out,
                                        const FermionOp &rhs)
    {
        out << rhs.toString();
        return out;
    }
private:
    OrbitalAct getOrbitalAct(const QString &item)
    {
        OrbitalAct oa;
        if (m_action.first == "")
        {
            if (item.find(m_action.second) == std::string::npos)
            {
                bool ok = false;
                oa.first = static_cast<size_t>(item.toInt(&ok));
                if (!ok)
                {
                    std::string err = "Bad fermion string.";
                    QCERR(err);
                    throw std::invalid_argument(err);
                }
                oa.second = true;
            }
            else
            {
                bool ok = false;
                oa.first = static_cast<size_t>(item.mid(0, item.size()
                                    - m_action.second.size()).toInt(&ok));
                if (!ok)
                {
                    std::string err = "Bad fermion string.";
                    QCERR(err);
                    throw std::invalid_argument(err);
                }
                oa.second = false;
            }
        }
        else if (m_action.second == "")
        {
            if (item.find(m_action.first) == std::string::npos)
            {
                bool ok = false;
                oa.first = static_cast<size_t>(item.toInt(&ok));
                if (!ok)
                {
                    std::string err = std::string("Bad fermion string.")
                            + item.data();
                    QCERR(err);
                    throw std::invalid_argument(err);
                }
                oa.second = false;
            }
            else
            {
                bool ok = false;
                oa.first = static_cast<size_t>(item.mid(0, item.size()
                                    - m_action.second.size()).toInt(&ok));
                if (!ok)
                {
                    std::string err = std::string("Bad fermion string.")
                            + item.data();
                    QCERR(err);
                    throw std::invalid_argument(err);
                }
                oa.second = true;
            }
        }
        else
        {
            if (item.find(m_action.first) != std::string::npos)
            {
                bool ok = false;
                oa.first = static_cast<size_t>(item.mid(0, item.size()
                                    - m_action.second.size()).toInt(&ok));
                if (!ok)
                {
                    std::string err = std::string("Bad fermion string.")
                            + item.data();
                    QCERR(err);
                    throw std::invalid_argument(err);
                }
                oa.second = true;
            }
            else if (item.find(m_action.second) != std::string::npos)
            {
                bool ok = false;
                oa.first = static_cast<size_t>(item.toInt(&ok));
                if (!ok)
                {
                    std::string err = std::string("Bad fermion string.")
                            + item.data();
                    QCERR(err);
                    throw std::invalid_argument(err);
                }
                oa.second = false;
            }
            else
            {
                std::string err = std::string("Bad fermion string.")
                        + item.data();
                QCERR(err);
                throw std::invalid_argument(err);
            }
        }

        return oa;
    }

    void insertData(const std::string &str, const T &value)
    {
        QString key(str);

        auto str_vec = key.split(" ", QString::SkipEmptyParts);
        if (str_vec.empty())
        {
            m_data.emplace_back(std::make_pair(FermionPair(), value));
            return;
        }

        OrbitalActVec oa_vec;
        for (size_t i = 0; i < str_vec.size(); i++)
        {
            OrbitalAct oa = getOrbitalAct(str_vec[i]);
            oa_vec.emplace_back(oa);
        }

        FermionPair item = std::make_pair(oa_vec, str);
        m_data.emplace_back(std::make_pair(item, value));
    }

    void reduceDuplicates()
    {
        std::map<std::string, T> data_map;
        std::map<std::string, OrbitalActVec> term_map;

        for (auto iter = m_data.begin(); iter != m_data.end(); iter++)
        {
            auto pair = iter->first;
            T value = iter->second;
            OrbitalActVec oa_vec = pair.first;
            std::string str = pair.second;

            if (data_map.find(str) != data_map.end())
            {
                data_map[str] += value;
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
            auto err = std::abs(iter->second);
            if (fabs(err) > fabs(m_error_threshold))
            {
                FermionPair pair;
                pair.first = term_map[iter->first];
                pair.second = iter->first;

                fermion_data.push_back(std::make_pair(pair, iter->second));
            }
        }

        m_data = std::move(fermion_data);
    }

    FermionOp normal_ordered_ladder_term(
            OrbitalActVec &term,
            std::string &term_str,
            T &coefficient)
    {
        FermionOp op;
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
                    coefficient *= -1;

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
                        T tmp_coef = coefficient*-1.0;
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
                        coefficient *= -1;
                    }
                }
            }
        }

        FermionPair item{term, term_str};
        FermionData data;
        data.push_back(std::make_pair(item, coefficient));
        op += FermionOp(data);

        return op;
    }

    std::string OrbitalActVec2String(const OrbitalActVec &vec)
    {
        std::string str;
        for (auto i = 0u; i < vec.size(); i++)
        {
            auto &item = vec[i];
            str += std::to_string(item.first);
            str += item.second ? m_action.first : m_action.second;

            if (i != (vec.size() - 1))
            {
                str += " ";
            }
        }

        return str;
    }
private:
    /*
     * first create symbol, second annihilation symbol
     */
    std::pair<std::string, std::string> m_action{"+", ""};
    FermionData m_data;

    double m_error_threshold{1e-6};
};

using FermionOperator = FermionOp<complex_d>;

QPANDA_END
#endif // FERMIONOPERATOR_H
