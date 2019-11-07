/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MODULE_H
#define MODULE_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include <array>
QPANDA_BEGIN

/**
* @namespace QPanda
*/

/**
* @class ModuleContext
* @brief 
* @note
*/
class ModuleContext {
    static QuantumMachine* qm;
    template<size_t nqubit>
    friend class qubits;
    friend class qubit;
public:
    inline static QuantumMachine* getContext() {
        return qm;
    }
    inline static void setContext(QuantumMachine* qm_new){
        qm = qm_new;
    }
};

/**
* @class qubit
* @brief Apply for qubit
* @note
*/
class qubit {
    Qubit* q;
public:
    qubit() {
        q = ModuleContext::qm->allocateQubit();
    }
    qubit(Qubit* q_) {
        q = ModuleContext::qm->allocateQubitThroughPhyAddress
            (q_->getPhysicalQubitPtr()->getQubitAddr());
    }
    explicit qubit(const qubit& q_old) {
        q =  ModuleContext::qm->allocateQubitThroughPhyAddress
            (q_old.q->getPhysicalQubitPtr()->getQubitAddr());
    }

    qubit operator=(const qubit &q_old)
    {
        ModuleContext::qm->qFree(q);
        q = ModuleContext::qm->allocateQubitThroughPhyAddress
        (q_old.q->getPhysicalQubitPtr()->getQubitAddr());
    }
    Qubit* get() {
        return q;
    }
    operator Qubit*() {
        return get();
    }
    ~qubit() {
        ModuleContext::qm->Free_Qubit(q);
    }
};

/**
* @class qubit
* @brief Apply for qubits
* @note
*/
template<size_t nqubit>
class qubits {
    std::array<qubit, nqubit> m_qubits;
public:
    qubits() {
    }
    qubits(QVec qs) {
        if (qs.size() != nqubit) throw std::runtime_error("bad size.");
        for (size_t i = 0; i < qs.size(); ++i) {
            m_qubits[i] = qs[i];
        }
    }
    qubits(const qubits<nqubit>& qs_old):m_qubits(qs_old.m_qubits){

    }
    qubit operator[](size_t i) {
        if (i >= nqubit)
            throw std::runtime_error("bad index.");
        return m_qubits[i];
    }
    QVec get() {
        QVec qs;
        for (size_t i = 0; i < m_qubits.size(); ++i) {
            qs.push_back(m_qubits[i]);
        }
        return qs;
    }
    ~qubits() {
    }
};

/**
* @class qubit
* @brief Apply for qubit vector
* @note
*/
class qvec :public std::vector<qubit> {
    typedef std::vector<qubit> BaseClass;
public:
    qvec()
    {
    }
    qvec(size_t size) {
        resize(size);
    }

    qvec(QVec qs) {
        for (size_t i = 0; i < qs.size(); ++i) {
            push_back(qs[i]);
        }
    }


    qvec(const qvec& qs_old) {
        clear();
        for (size_t i = 0; i < qs_old.size(); ++i) {
            push_back(qs_old[i]);
        }
    }

    template<size_t nqubit>
    qvec(qubits<nqubit>& qs_old) {
        clear();
        auto temp_qs = qs_old.get();
        for (size_t i = 0; i < temp_qs.size(); ++i) {
            push_back(temp_qs[i]);
        }
    }

    QVec get() {
        QVec qs;
        for (size_t i = 0; i < size(); ++i) {
            qs.push_back(BaseClass::operator[](i));
        }
        return qs;
    }

    qvec operator +(qvec vec)
    {
        qvec new_vec(*this);

        for (auto aiter = vec.begin(); aiter != vec.end(); aiter++)
        {
            auto biter = begin();
            for (; biter != end(); biter++)
            {
                if (*aiter == *biter)
                {
                    break;
                }
            }

            if (biter == end())
            {
                new_vec.push_back(*aiter);
            }
        }

        return new_vec;
    }

    qvec operator -(qvec& vec)
    {
        qvec new_vec;

        for (auto aiter = begin(); aiter != end(); aiter++)
        {
            auto biter = vec.begin();
            for (; biter != vec.end(); biter++)
            {
                if (*aiter == *biter)
                {
                    break;
                }
            }

            if (biter == vec.end())
            {
                new_vec.push_back(*aiter);
            }
        }

        return new_vec;
    }
    ~qvec() {
    }
};


QPANDA_END
#endif