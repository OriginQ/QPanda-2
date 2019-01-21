/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

PauliOperator.h

Author: LiYe
Created in 2018-09-20


*/

#ifndef PAULIOPERATOR_H
#define PAULIOPERATOR_H

#include <complex>
#include <ostream>
#include "QAlg/DataStruct.h"
#include "Core/Utilities/QString.h"
QPANDA_BEGIN
/*
Pauli Operator
*/
class PauliOperator
{
public:
    PauliOperator(){}
    PauliOperator(double value);
    PauliOperator(const complex_d &complex);
    PauliOperator(const QPauliMap &map);
    PauliOperator(const QHamiltonian &hamiltonian);
    PauliOperator(PauliOperator &&op);
    PauliOperator(const PauliOperator &op);
    PauliOperator(QPauli &&pauli);
    PauliOperator(const QPauli &pauli);
    PauliOperator &operator = (const PauliOperator &op);
    PauliOperator &operator = (PauliOperator &&op);

    PauliOperator dagger() const;

    PauliOperator remapQubitIndex(std::map<size_t, size_t> &index_map);

    size_t getMaxIndex();

    bool isEmpty() { return m_data.empty(); }
    bool isAllPauliZorI();

    void setErrorThreshold(double threshold)
    {
        m_error_threshold = threshold;
    }

    double error_threshold() const { return m_error_threshold; }

    QHamiltonian toHamiltonian(bool *ok = nullptr) const;
    std::string  toString() const;

    const QPauli &data() const { return m_data; }

    PauliOperator  operator + (const PauliOperator &rhs) const;
    PauliOperator  operator - (const PauliOperator &rhs) const;
    PauliOperator  operator * (const PauliOperator &rhs) const;
    PauliOperator &operator +=(const PauliOperator &rhs);
    PauliOperator &operator -=(const PauliOperator &rhs);
    PauliOperator &operator *=(const PauliOperator &rhs);

    friend PauliOperator operator + (const complex_d &lhs,
                                        const PauliOperator &rhs)
    {
        return rhs + lhs;
    }

    friend PauliOperator operator - (const complex_d &lhs,
                                        const PauliOperator &rhs)
    {
        return rhs*-1.0 + lhs;
    }

    friend PauliOperator operator * (const complex_d &lhs,
                                        const PauliOperator &rhs)
    {
        return rhs * lhs;
    }

    friend PauliOperator operator + (const QHamiltonian &lhs,
                                        const PauliOperator &rhs)
    {
        return rhs + lhs;
    }

    friend PauliOperator operator - (const QHamiltonian &lhs,
                                        const PauliOperator &rhs)
    {
        return rhs*-1.0 + lhs;
    }

    friend PauliOperator operator * (const QHamiltonian &lhs,
                                        const PauliOperator &rhs)
    {
        return rhs * lhs;
    }

    friend std::ostream  &operator <<(std::ostream &out,
                                        const PauliOperator &rhs)
    {
        out << rhs.toString();
        return out;
    }

    friend std::ostream  &operator <<(std::ostream &out,
                                        const QHamiltonian &rhs)
    {
        out << PauliOperator(rhs).toString();
        return out;
    }

    friend std::ostream  &operator <<(std::ostream &out,
                                        const QPauliMap &rhs)
    {
        out << PauliOperator(rhs).toString();
        return out;
    }

    friend std::ostream  &operator <<(std::ostream &out,
                                        const QPauli &rhs)
    {
        out << PauliOperator(rhs).toString();
        return out;
    }

private:
    QTermPair   genQTermPair(const QString &str) const;
    QPauliItem  genQPauliItem(const QTerm &map_i,
                                const QTerm &map_j,
                                const complex_d &value) const;
    std::string QTerm2StdString(const QTerm &map) const;

    void reduceDuplicates();
private:
    QPauli m_data;

    double m_error_threshold{1e-6};
};

QPANDA_END
#endif // PAULIOPERATOR_H
