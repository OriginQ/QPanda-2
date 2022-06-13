/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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

#ifndef CPU_QUANTUM_GATE_H
#define CPU_QUANTUM_GATE_H

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <stdio.h>
#include <array>
#include <bitset>
#include <iostream>
#include "Core/Utilities/Tools/Utils.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "include/Core/VirtualQuantumProcessor/CPUSupportAvx2.h"
QPANDA_BEGIN

/**
* @brief QPU implementation by  CPU model
* @ingroup VirtualQuantumProcessor
*/
template <typename data_t = double>
class CPUImplQPU : public QPUImpl
{
public:
    CPUImplQPU();
    CPUImplQPU(size_t qubit_num);
    ~CPUImplQPU();

    template<const qcomplex_t& U00, const qcomplex_t& U01, const qcomplex_t& U10, const qcomplex_t& U11>
    QError single_gate(size_t qn, bool is_dagger, double error_rate)
    {
        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, matrix, is_dagger);

        return qErrorNone;
    }


    QError U1_GATE(size_t qn, double theta, bool is_dagger, double error_rate)
    {
        QStat matrix = { 1, 0, 0, qcomplex_t(cos(theta),sin(theta)) };
        _U1(qn, matrix, is_dagger);
        return qErrorNone;
    }

	QError P_GATE(size_t qn, double theta, bool is_dagger, double error_rate)
	{
		QStat matrix = { 1, 0, 0, qcomplex_t(cos(theta),sin(theta)) };
		_U1(qn, matrix, is_dagger);
		return qErrorNone;
	}


    template<const double& Nx, const double& Ny, const double& Nz>
    QError single_angle_gate(size_t qn, double theta, bool is_dagger, double error_rate)
    {
        qcomplex_t U00(cos(theta / 2), -sin(theta / 2)*Nz);
        qcomplex_t U01(-sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U10(sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U11(cos(theta / 2), sin(theta / 2)*Nz);

        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, matrix, is_dagger);

        return qErrorNone;
    }

    template<const double& Nx, const double& Ny, const double& Nz>
    QError control_single_angle_gate(size_t qn,
        double theta,
        Qnum vControlBit,
        bool is_dagger,
        double error_rate)
    {
        qcomplex_t U00(cos(theta / 2), -sin(theta / 2)*Nz);
        qcomplex_t U01(-sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U10(sin(theta / 2)*Ny, -sin(theta / 2)*Nx);
        qcomplex_t U11(cos(theta / 2), sin(theta / 2)*Nz);

        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, vControlBit, matrix, is_dagger);
        return qErrorNone;
    }

    template<const qcomplex_t& U00,
        const qcomplex_t& U01,
        const qcomplex_t& U10,
        const qcomplex_t& U11>
        QError control_single_gate(
            size_t qn,
            Qnum  vControlBit,
            bool is_dagger,
            double error_rate)
    {
        QStat matrix = { U00, U01, U10, U11 };
        _single_qubit_normal_unitary(qn, vControlBit, matrix, is_dagger);
        return qErrorNone;
    }

    std::vector<std::complex<data_t>> convert(const QStat& v) const;
	QError single_qubit_gate_fusion(size_t qn, QStat& matrix);
	QError double_qubit_gate_fusion(size_t qn_0, size_t qn_1, QStat &matrix);
    QError three_qubit_gate_fusion(size_t qn_0, size_t qn_1, QStat &matrix);
    //define unitary single/double quantum gate
    QError unitarySingleQubitGate(size_t qn,
        QStat& matrix, bool is_dagger,
        GateType);
    QError controlunitarySingleQubitGate(size_t qn, Qnum& controls,
        QStat& matrix, bool is_dagger,
        GateType type);
    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
        QStat& matrix, bool is_dagger,
        GateType);
    QError controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum& controls,
        QStat& matrix, bool is_dagger,
        GateType);
    QError DiagonalGate(Qnum& vQubit, QStat & matrix,
        bool isConjugate, double error_rate);
    QError controlDiagonalGate(Qnum& vQubit, QStat & matrix, Qnum& vControlBit,
        bool isConjugate, double error_rate);

    QError OracleGate(Qnum& qubits, QStat &matrix,
                      bool is_dagger);
    QError controlOracleGate(Qnum& qubits, const Qnum &controls,
                             QStat &matrix, bool is_dagger);
    
    virtual QError process_noise(Qnum& qnum, QStat& matrix);
    virtual QError debug(std::shared_ptr<QPanda::AbstractQDebugNode> debugger);

    QStat getQState();
    QError Reset(size_t qn);
    bool qubitMeasure(size_t qn);
    QError pMeasure(Qnum& qnum, prob_tuple &probs,
        int select_max = -1);
    QError pMeasure(Qnum& qnum, prob_vec &probs);
    QError initState(size_t head_rank, size_t rank_size, size_t qubit_num);
    QError initState(size_t qubit_num, const QStat &state = {});
	QError initMatrixState(size_t qubit_num, const QStat& state = {});


protected:
	
    QError _single_qubit_normal_unitary(size_t qn, QStat& matrix, bool is_dagger);
    QError _single_qubit_normal_unitary(size_t qn, Qnum& controls, QStat& matrix, bool is_dagger);

    QError _double_qubit_normal_unitary(size_t qn_0, size_t qn_1, QStat& matrix, bool is_dagger);
    QError _double_qubit_normal_unitary(size_t qn_0, size_t qn_1, Qnum& controls, QStat& matrix, bool is_dagger);
    QError _three_qubit_gate(Qnum &qubits, QStat& matrix, bool is_dagger, const Qnum& controls = {});
    QError _four_qubit_gate(Qnum &qubits, QStat& matrix, bool is_dagger, const Qnum& controls = {});
    QError _five_qubit_gate(Qnum &qubits, QStat& matrix, bool is_dagger, const Qnum& controls = {});
	QError _X(size_t qn);
    QError _Y(size_t qn);
    QError _Z(size_t qn);
    QError _S(size_t qn, bool is_dagger);
    QError _U1(size_t qn, QStat &matrix, bool is_dagger);
	QError _P(size_t qn, QStat &matrix, bool is_dagger);
    QError _RZ(size_t qn, QStat &matrix, bool is_dagger);
    QError _H(size_t qn, QStat &matrix);

    QError _CNOT(size_t qn_0, size_t qn_1);
    QError _CZ(size_t qn_0, size_t qn_1);
    QError _CR(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
	QError _CP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
    QError _SWAP(size_t qn_0, size_t qn_1);
    QError _iSWAP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
    QError _iSWAP_theta(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);
    QError _CU(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger);

    QError _X(size_t qn, Qnum &controls);
    QError _Y(size_t qn, Qnum &controls);
    QError _Z(size_t qn, Qnum &controls);
    QError _S(size_t qn, bool is_dagger, Qnum &controls);
    QError _U1(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls);
	QError _P(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _RZ(size_t qn, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _H(size_t qn, QStat &matrix, Qnum &controls);

    QError _CNOT(size_t qn_0, size_t qn_1, Qnum &controls);
    QError _CZ(size_t qn_0, size_t qn_1, Qnum &controls);
    QError _CR(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
	QError _CP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _SWAP(size_t qn_0, size_t qn_1, Qnum &controls);
    QError _iSWAP(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _iSWAP_theta(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    QError _CU(size_t qn_0, size_t qn_1, QStat &matrix, bool is_dagger, Qnum &controls);
    void set_parallel_threads_size(size_t size);


	inline int64_t _insert(int64_t value, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
	{
		int64_t mask1 = (1ll << n1) - 1;
		int64_t mask2 = (1ll << (n2 - 1)) - 1;
		int64_t mask3 = (1ll << (n3 - 2)) - 1;
		int64_t mask4 = (1ll << (n4 - 3)) - 1;
		int64_t mask5 = (1ll << (n5 - 4)) - 1;
		int64_t t = value & mask1;
		int64_t s = ~mask1 & value & mask2;//(value - w) & mask2; //~mask1 & value & mask2;
		int64_t w = ~mask2 & value & mask3;//(value - z) & mask3;					//~mask1 & value & mask2 & mask3;
		int64_t z = ~mask3 & value & mask4;//(value - z) & mask3;
		int64_t y = ~mask4 & value & mask5;//~mask3 & value;
		int64_t x = ~mask5 & value;
		return ((x << 5) | (y << 4) | (z << 3) | (w << 2) | (s << 1) | t);
	}

	inline int64_t _insert(int64_t value, size_t n1, size_t n2, size_t n3, size_t n4)
	{
		int64_t mask1 = (1ll << n1) - 1;
		int64_t mask2 = (1ll << (n2 - 1)) - 1;
		int64_t mask3 = (1ll << (n3 - 2)) - 1;
		int64_t mask4 = (1ll << (n4 - 3)) - 1;
		int64_t s = value & mask1;
		int64_t w = ~mask1 & value & mask2;//(value - w) & mask2; //~mask1 & value & mask2;
		int64_t z = ~mask2 & value & mask3;//(value - z) & mask3;					//~mask1 & value & mask2 & mask3;
		int64_t y = ~mask3 & value & mask4;//(value - z) & mask3;
		int64_t x = ~mask4 & value;//~mask3 & value;

		return ((x << 4) | (y << 3) | (z << 2) | (w << 1) | s);
	}


	inline int64_t _insert(int64_t value, size_t n1, size_t n2, size_t n3)
	{
		int64_t mask1 = (1ll << n1) - 1;
		int64_t mask2 = (1ll << (n2 - 1)) - 1;
		int64_t mask3 = (1ll << (n3 - 2)) - 1;
		int64_t w = value & mask1;
		int64_t z = ~mask1 & value & mask2;//(value - w) & mask2; //~mask1 & value & mask2;
		int64_t y = ~mask2 & value & mask3;//(value - z) & mask3;					//~mask1 & value & mask2 & mask3;
		int64_t x = ~mask3 & value;//~mask3 & value;

		return ((x << 3) | (y << 2) | (z << 1) | w);
	}
    inline int64_t _insert(int64_t value, size_t n1, size_t n2)
    {
		if (n1 > n2)
		{
			std::swap(n1, n2);
		}
        int64_t mask1 = (1ll << n1) - 1;
        int64_t mask2 = (1ll << (n2 - 1)) - 1;
        int64_t z = value & mask1;
        int64_t y = ~mask1 & value & mask2;
        int64_t x = ~mask2 & value;

        return ((x << 2) | (y << 1) | z);
    }

    inline int64_t _insert(int64_t value, size_t n)
    {
        int64_t number = 1ll << n;
        if (value < number)
        {
            return value;
        }

        int64_t mask = number - 1;
        int64_t x = mask & value;
        int64_t y = ~mask & value;
        return ((y << 1) | x);
    }

	int64_t _insert(int64_t value, Qnum &qns)
	{
		std::sort(qns.begin(), qns.end());
		std::vector<int> MASK(qns.size());
		for (int i = 0; i < qns.size(); i++)
		{
			MASK[i] = (1 << (qns[i] - i)) - 1;
		}
		std::vector<int> tmp(MASK.size() + 1);
		tmp[0] = MASK[0] & value;

		for (int i = 1; i < tmp.size() - 1; i++)
		{
			tmp[i] = ~MASK[i - 1] & value & MASK[i];
		}
		tmp[tmp.size() - 1] = ~MASK[MASK.size() - 1] & value;

		int offset = tmp[0];
		for (int i = 1; i < tmp.size(); i++)
		{
			offset += tmp[i] << i;
		}

		return offset;
	}

    int64_t _insert(Qnum &sorted_qubits, int num_qubits, const int64_t k)
    {
        int64_t lowbits, retval = k;
        for (size_t j = 0; j < num_qubits; j++) {
            lowbits = retval & ((1 << sorted_qubits[j]) - 1);
            retval >>= sorted_qubits[j];
            retval <<= sorted_qubits[j] + 1;
            retval |= lowbits;
        }
        return retval;
    }

    inline void load_index(int64_t index0, int num_qubits, int64_t* indexes,
        const size_t indexes_size, const Qnum& qregs)
    {
        for (size_t i = 0; i < indexes_size; ++i) {
            indexes[i] = index0;
        }

        for (size_t n = 0; n < num_qubits; ++n) {
            for (size_t i = 0; i < indexes_size; i += (1ull << (n + 1))) {
                for (size_t j = 0; j < (1ull << n); ++j) {
                    indexes[i + j + (1ull << n)] += (1ull << qregs[n]);

                }
            }
        }
    }

    void _verify_state(const QStat &state);
    inline int _omp_thread_num(size_t size);
private:
    bool m_is_init_state{false};
    /* 
      qubits state vetor of tensor product, 
      
      qubit (ax, bx are complex):
      q0 = a0|0> + b0|1>
      q1 = a1|0> + b1|1>
      ...
      qn = an|0> + bn|1>    

      qubits state vetor of tensor product is arraged as sequence:
      m_state = [an...a1a0, an...a1b0, an...b1a0, an...b1b0, ..., bn...b1b0]
    */
    std::vector<std::complex<data_t>> m_state;
    std::vector<std::complex<data_t>> m_init_state;
    size_t m_qubit_num;
    const int64_t m_threshold = 1ll << 9;
    int64_t m_max_threads_size = 0;
};

class CPUImplQPUWithOracle : public CPUImplQPU<double> {
public:
    QError controlOracularGate(std::vector<size_t> bits,
        std::vector<size_t> controlbits,
        bool is_dagger,
        std::string name);
};

QPANDA_END

#endif
