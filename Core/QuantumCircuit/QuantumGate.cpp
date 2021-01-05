/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGate.cpp
Author: Menghan.Dou
Created in 2018-6-30

Classes for QGate

Update@2018-8-30
Update by code specification
*/

#include "QuantumGate.h"
#include "QGlobalVariable.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include <float.h>

using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA


QuantumGate::QuantumGate()
{
    gate_type = -1;
}

U4::U4(U4 &toCopy)
{
    operation_num = toCopy.operation_num;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
    this->gate_matrix = toCopy.gate_matrix;
}

U4::U4()
{
    operation_num = 1;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
    gate_matrix.push_back(1);
    gate_matrix.push_back(0);
    gate_matrix.push_back(0);
    gate_matrix.push_back(1);
	gate_type = GateType::U4_GATE;
}

U4::U4(double _alpha, double _beta, double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    operation_num = 1;
    QStat matrix;
    gate_matrix.push_back(qcomplex_t(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2)));
    gate_matrix.push_back(qcomplex_t(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2)));
    gate_matrix.push_back(qcomplex_t(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2)));
    gate_matrix.push_back(qcomplex_t(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2)));
	gate_type = GateType::U4_GATE;
}
U4::U4(QStat & matrix)
{
    operation_num = 1;
    gate_matrix.resize(4);
    gate_matrix[0] = matrix[0];
    gate_matrix[1] = matrix[1];
    gate_matrix[2] = matrix[2];
    gate_matrix[3] = matrix[3];
	if (abs(gate_matrix[0]) >= 1.0)
	{
		gamma = 0.0;
	}
	else
	{
		gamma = 2 * acos(abs(gate_matrix[0]));
	}
    
    if ((abs(gate_matrix[0]) > DBL_EPSILON) && (abs(gate_matrix[2])> DBL_EPSILON))
    {
        beta = argc(gate_matrix[2] / gate_matrix[0]);
		
        delta = argc(gate_matrix[3] / gate_matrix[2]);
        alpha = beta / 2 + delta / 2 + argc(gate_matrix[0]);
    }
    else if (abs(gate_matrix[0]) > DBL_EPSILON)
    {
        beta = argc(gate_matrix[3] / gate_matrix[0]);
        delta = 0;
        alpha = beta / 2 + argc(gate_matrix[0]);
    }
    else
    {
        beta = argc(gate_matrix[2] / gate_matrix[1]) + PI;
        delta = 0;
        alpha = argc(gate_matrix[1]) + beta / 2 - PI;
    }
	gate_type = GateType::U4_GATE;
}


U4::U4(QuantumGate *qgate_old)
{
	if (nullptr == qgate_old)
	{
		QCERR("Parameter qgate_old error");
		throw invalid_argument("Parameter qgate_old error");
	}

	auto u4_ptr_old = static_cast<U4 *>(qgate_old);
	if (nullptr == u4_ptr_old)
	{
		QCERR("static cast fail");
		throw invalid_argument("static cast fail");
	}

	u4_ptr_old->getMatrix(gate_matrix);
	alpha = u4_ptr_old->alpha;
	beta  = u4_ptr_old->beta;
	delta = u4_ptr_old->delta;
	gamma = u4_ptr_old->gamma;

	gate_type = u4_ptr_old->gate_type;
	operation_num = u4_ptr_old->operation_num;
}

void U4::getMatrix(QStat & matrix) const
{
    if (gate_matrix.size() != 4)
    {
        QCERR("the size of gate_matrix is error");
        throw invalid_argument("the size of gate_matrix is error");
    }

    for (auto aIter : gate_matrix)
    {
        matrix.push_back(aIter);
    }
}

//I gate
I::I()
{
	operation_num = 1;
	alpha = 0;
	beta = 0;
	gamma = 0;
	delta = 0;
	gate_matrix[0] = 1;
	gate_matrix[1] = 0;
	gate_matrix[2] = 0;
	gate_matrix[3] = 1;
	gate_type = GateType::I_GATE;
}

//X_GATE gate
X::X()
{
	operation_num = 1;
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = PI;
    gate_matrix[0] = 0;
    gate_matrix[1] = 1;
    gate_matrix[2] = 1;
    gate_matrix[3] = 0;
	gate_type = GateType::PAULI_X_GATE;
}


//Y_GATE gate
Y::Y()
{
	operation_num = 1;
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = 0;
    gate_matrix[0] = 0;
    gate_matrix[1].imag(-1);
    gate_matrix[2].imag(1);
    gate_matrix[3] = 0;
	gate_type = GateType::PAULI_Y_GATE;
}


//PauliZ gate,[1 0;0 -1]
Z::Z()
{
	operation_num = 1;
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
    gate_matrix[3] = -1;
	gate_type = GateType::PAULI_Z_GATE;
}

//RX(pi/2) gate
X1::X1()
{
	operation_num = 1;
    alpha = PI;
    beta = 3.0 / 2 * PI;
    gamma = PI / 2;
    delta = PI / 2;
    gate_matrix[0] = 1 / SQRT2;
    gate_matrix[1] = qcomplex_t(0, -1 / SQRT2);
    gate_matrix[2] = qcomplex_t(0, -1 / SQRT2);
    gate_matrix[3] = 1 / SQRT2;
	gate_type = GateType::X_HALF_PI;
}


//RY(pi/2) gate
Y1::Y1()
{
	operation_num = 1;
    alpha = 0;
    beta = 0;
    gamma = PI / 2;
    delta = 0;
    gate_matrix[0] = 1 / SQRT2;
    gate_matrix[1] = -1 / SQRT2;
    gate_matrix[2] = 1 / SQRT2;
    gate_matrix[3] = 1 / SQRT2;
	gate_type = GateType::Y_HALF_PI;
}


//RZ(pi/2) gate
Z1::Z1()
{
	operation_num = 1;
    alpha = 0;
    beta = PI / 2;
    gamma = 0;
    delta = 0;
    gate_matrix[0] = qcomplex_t(1 / SQRT2, -1 / SQRT2);
    gate_matrix[3] = qcomplex_t(1 / SQRT2, 1 / SQRT2);
	gate_type = GateType::Z_HALF_PI;
}

H::H()
{
	operation_num = 1;
    alpha = PI / 2;
    beta = 0;
    gamma = PI / 2;
    delta = PI;
    gate_matrix[0] = 1 / SQRT2;
    gate_matrix[1] = 1 / SQRT2;
    gate_matrix[2] = 1 / SQRT2;
    gate_matrix[3] = -1 / SQRT2;
	gate_type = GateType::HADAMARD_GATE;
}

ECHO::ECHO()
{
	operation_num = 1;
	alpha = 0;
	beta = 0;
	gamma = 0;
	delta = 0;
	gate_matrix[0] = 1;
	gate_matrix[1] = 0;
	gate_matrix[2] = 0;
	gate_matrix[3] = 1;
	gate_type = GateType::ECHO_GATE;
}

BARRIER::BARRIER()
{
    operation_num = 1;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
    gate_matrix[0] = 1;
    gate_matrix[1] = 0;
    gate_matrix[2] = 0;
    gate_matrix[3] = 1;
    gate_type = GateType::BARRIER_GATE;
}

//S
S::S()
{
	operation_num = 1;
    alpha = PI / 4;
    beta = PI / 2;
    gamma = 0;
    delta = 0;
    gate_matrix[3].real(0);
    gate_matrix[3].imag(1);
	gate_type = GateType::S_GATE;
}

T::T()
{
	operation_num = 1;
    alpha = PI / 8;
    beta = PI / 4;
    gamma = 0;
    delta = 0;
    gate_matrix[3].real(1 / SQRT2);
    gate_matrix[3].imag(1 / SQRT2);
	gate_type = GateType::T_GATE;
}

RX::RX(double angle)
{
	operation_num = 1;
    alpha = PI;
    beta = 3.0 / 2 * PI;
    gamma = angle;
    delta = PI / 2;
    gate_matrix[0] = cos(angle / 2);
    gate_matrix[1].imag(-1 * sin(angle / 2));
    gate_matrix[2].imag(-1 * sin(angle / 2));
    gate_matrix[3] = cos(angle / 2);
	gate_type = GateType::RX_GATE;
}

RY::RY(double angle)
{
	operation_num = 1;
    alpha = 0;
    beta = 0;
    gamma = angle;
    delta = 0;
    gate_matrix[0] = cos(angle / 2);
    gate_matrix[1] = -sin(angle / 2);
    gate_matrix[2] = sin(angle / 2);
    gate_matrix[3] = cos(angle / 2);
	gate_type = GateType::RY_GATE;
}

RZ::RZ(double angle)
{
	operation_num = 1;
    alpha = 0;
    beta = angle;
    gamma = 0;
    delta = 0;
    gate_matrix[0].real(cos(angle / 2));
    gate_matrix[0].imag(-1 * sin(angle / 2));
    gate_matrix[3].real(cos(angle / 2));
    gate_matrix[3].imag(1 * sin(angle / 2));
	gate_type = GateType::RZ_GATE;
}

RPhi::RPhi(double angle, double phi)
{
	operation_num = 1;
	m_phi = phi;
	alpha = 0;
	beta = angle;
	gamma = 0;
	delta = 0;
	gate_matrix[0] = (cos(angle / 2));
	gate_matrix[1] = qcomplex_t(0, -1) * qcomplex_t(sin(angle / 2), 0) * (qcomplex_t(cos(phi), 0) - qcomplex_t(0, sin(phi)));
	gate_matrix[2] = qcomplex_t(0, -1) * qcomplex_t(sin(angle / 2), 0) * (qcomplex_t(cos(phi), 0) + qcomplex_t(0, sin(phi)));
	gate_matrix[3] = (cos(angle / 2));
	gate_type = GateType::RPHI_GATE;
}

//U1_GATE=[1 0;0 exp(i*angle)]
U1::U1(double angle)
{
	operation_num = 1;
    alpha = angle / 2;
    beta = angle;
    gamma = 0;
    delta = 0;
    gate_matrix[3].real(cos(angle));
    gate_matrix[3].imag(1 * sin(angle));
	gate_type = GateType::U1_GATE;
}

QDoubleGate::QDoubleGate(QuantumGate  * qgate_old)
{
	if (nullptr == qgate_old)
	{
		QCERR("Parameter qgate_old error");
		throw invalid_argument("Parameter qgate_old error");
	}

	auto double_gate_ptr_old = static_cast<QDoubleGate *>(qgate_old);
	if (nullptr == double_gate_ptr_old)
	{
		QCERR("Static cast fail");
		throw invalid_argument("Static cast fail");
	}

	gate_type = double_gate_ptr_old->gate_type;
	gate_matrix.assign(double_gate_ptr_old->gate_matrix.begin(),
		double_gate_ptr_old->gate_matrix.end());
	operation_num = double_gate_ptr_old->operation_num;
}

QDoubleGate::QDoubleGate()
{
    operation_num = 2;
	gate_type = GateType::TWO_QUBIT_GATE;
    gate_matrix.resize(16);
    gate_matrix[0] = 1;
    gate_matrix[5] = 1;
    gate_matrix[10] = 1;
    gate_matrix[15] = 1;
}

QDoubleGate::QDoubleGate(const QDoubleGate & oldDouble)
{
    this->operation_num = oldDouble.operation_num;
    this->gate_matrix = oldDouble.gate_matrix;
	gate_type = oldDouble.gate_type;
}
QDoubleGate::QDoubleGate(QStat & matrix)
{
    operation_num = 2;
    if (matrix.size() != 16)
    {
        QCERR("Given matrix is invalid.");
        throw invalid_argument("Given matrix is invalid.");
    }
    this->gate_matrix = matrix;
	gate_type = GateType::TWO_QUBIT_GATE;
}
void QDoubleGate::getMatrix(QStat & matrix) const
{
    if (gate_matrix.size() != 16)
    {
        QCERR("Given matrix is invalid.");
        throw invalid_argument("Given matrix is invalid.");
    }
    matrix = gate_matrix;
}

CU::CU(QuantumGate  * gate_old) :QDoubleGate(gate_old)
{
	auto cu_gate_ptr_old = static_cast<CU *>(gate_old);
	if (nullptr == cu_gate_ptr_old)
	{
		QCERR("Static cast fail");
		throw invalid_argument("Static cast fail");
	}
	alpha = cu_gate_ptr_old->alpha;
	beta = cu_gate_ptr_old->beta;
	gamma = cu_gate_ptr_old->gamma;
	delta = cu_gate_ptr_old->delta;

	gate_type = GateType::CU_GATE;
}

CU::CU()
{
	operation_num = 2;
	gate_matrix.resize(16, 0);
	gate_matrix[0] = 1;
	gate_matrix[5] = 1;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
	gate_type = GateType::CU_GATE;
}

CU::CU(const CU &toCopy)
{
    operation_num = toCopy.operation_num;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
    this->gate_matrix = toCopy.gate_matrix;
	gate_type = GateType::CU_GATE;
}

CU::CU(double _alpha, double _beta,
    double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    operation_num = 2;
    gate_matrix[10] = qcomplex_t(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2));
    gate_matrix[11] = qcomplex_t(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2));
    gate_matrix[14] = qcomplex_t(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2));
    gate_matrix[15] = qcomplex_t(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2));
	gate_type = GateType::CU_GATE;
}

CU::CU(QStat & matrix)
{
    operation_num = 2;
    gate_matrix.resize(16);
    gate_matrix[0] = 1;
    gate_matrix[5] = 1;
    gate_matrix[10] = matrix[0];
    gate_matrix[11] = matrix[1];
    gate_matrix[14] = matrix[2];
    gate_matrix[15] = matrix[3];

	if (abs(gate_matrix[10]) >= 1.0)
	{
		gamma = 0.0;
	}
	else
	{
		gamma = 2 * acos(abs(gate_matrix[10]));
	}
	if ((abs(gate_matrix[10]) > DBL_EPSILON) && (abs(gate_matrix[14]) > DBL_EPSILON))
    /*if (abs(gate_matrix[10] * gate_matrix[11]) > 1e-13)*/
    {
        beta = argc(gate_matrix[14] / gate_matrix[10]);
        delta = argc(gate_matrix[15] / gate_matrix[14]);
        alpha = beta / 2 + delta / 2 + argc(gate_matrix[10]);
    }
    else if (abs(gate_matrix[10]) > DBL_EPSILON)
    {
        beta = argc(gate_matrix[15] / gate_matrix[10]);
        delta = 0;
        alpha = beta / 2 + argc(gate_matrix[10]);
    }
    else
    {
        beta = argc(gate_matrix[14] / gate_matrix[11]) + PI;
        delta = 0;
        alpha = argc(gate_matrix[11]) + beta / 2 - PI;
    }

	gate_type = GateType::CU_GATE;
}

CNOT::CNOT()
{
	operation_num = 2;
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = PI;
    gate_matrix[10] = 0;
    gate_matrix[11] = 1;
    gate_matrix[14] = 1;
    gate_matrix[15] = 0;

	gate_type = GateType::CNOT_GATE;
}

CNOT::CNOT(const CNOT & toCopy)
{
    operation_num = toCopy.operation_num;
    this->gate_matrix = toCopy.gate_matrix;
	gate_type = GateType::CNOT_GATE;
}

CPHASE::CPHASE(double angle)
{
    operation_num = 2;
    alpha = angle / 2;
    beta = angle;
    gamma = 0;
    delta = 0;
    gate_matrix[15] = cos(angle);
    gate_matrix[15].imag(1 * sin(angle));
	gate_type = GateType::CPHASE_GATE;
}

CZ::CZ()
{
	operation_num = 2;
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
    gate_matrix[15] = -1;
	gate_type = GateType::CZ_GATE;
}

ISWAPTheta::ISWAPTheta(QuantumGate  * gate_old) :QDoubleGate(gate_old)
{
	if (gate_old->getGateType() != GateType::ISWAP_THETA_GATE)
	{
		QCERR("Parameter qgate_old error");
		throw std::invalid_argument("Parameter qgate_old error");
	}

	auto iswap_theta_gate_ptr_old = static_cast<ISWAPTheta *>(gate_old);
	if (nullptr == iswap_theta_gate_ptr_old)
	{
		QCERR("Static cast fail");
		throw invalid_argument("Static cast fail");
	}
	

	gate_type = gate_old->getGateType();
	theta = iswap_theta_gate_ptr_old->theta;
}

ISWAPTheta::ISWAPTheta(double angle)
{
    operation_num = 2;
    theta = angle;
    gate_matrix[5] = cos(angle);
    gate_matrix[6].imag(-1 * sin(angle));
    gate_matrix[9].imag(-1 * sin(angle));
    gate_matrix[10] = cos(angle);
	gate_type = GateType::ISWAP_THETA_GATE;
}

ISWAP::ISWAP()
{
	operation_num = 2;
    gate_matrix[5] = 0;
    gate_matrix[6].imag(-1);
    gate_matrix[9].imag(-1);
    gate_matrix[10] = 0;
	gate_type = GateType::ISWAP_GATE;
}

SQISWAP::SQISWAP()
{
	operation_num = 2;
    theta = PI / 4;
    gate_matrix[5] = 1 / SQRT2;
    gate_matrix[6].imag(-1 / SQRT2);
    gate_matrix[9].imag(-1 / SQRT2);
    gate_matrix[10] = 1 / SQRT2;
	gate_type = GateType::SQISWAP_GATE;
}

SWAP::SWAP()
{
	operation_num = 2;
    gate_matrix[5] = 0;
    gate_matrix[6] = 1;
    gate_matrix[9] = 1;
    gate_matrix[10] = 0;
	gate_type = GateType::SWAP_GATE;
}

OracularGate::OracularGate(QuantumGate * qgate_old)
{
	if (nullptr == qgate_old)
	{
		QCERR("Parameter qgate_old error");
		throw invalid_argument("Parameter qgate_old error");
	}

	if (qgate_old->getGateType() != GateType::ORACLE_GATE)
	{
		QCERR("Parameter qgate_old error");
		throw invalid_argument("Parameter qgate_old error");
	}

	auto oracular_ptr_old = static_cast<OracularGate *>(qgate_old);
	if (nullptr == oracular_ptr_old)
	{
		QCERR("static cast fail");
		throw invalid_argument("static cast fail");
	}

	oracle_name = oracular_ptr_old->oracle_name;
}

U2::U2(double phi, double lambda)
    :m_phi(phi), m_lambda(lambda)
{
    gate_type = GateType::U2_GATE;

    alpha = (phi + lambda) / 2;
    beta = phi;
    gamma = PI / 2;
    delta = lambda;
    
    auto coefficient = static_cast<qstate_type>(sqrt(2) / 2);
    gate_matrix[0] = 1 * coefficient;
    gate_matrix[1].real(static_cast<qstate_type>(-cos(lambda)) * coefficient);
    gate_matrix[1].imag(static_cast<qstate_type>(-sin(lambda)) * coefficient);

    gate_matrix[2].real(static_cast<qstate_type>(cos(phi)) * coefficient);
    gate_matrix[2].imag(static_cast<qstate_type>(sin(phi)) * coefficient);
    gate_matrix[3].real(static_cast<qstate_type>(cos(phi + lambda)) * coefficient);
    gate_matrix[3].imag(static_cast<qstate_type>(sin(phi + lambda)) * coefficient);
}

U3::U3(double theta, double phi, double lambda)
    :m_theta(theta), m_phi(phi), m_lambda(lambda)
{
    gate_type = GateType::U3_GATE;
    auto tmp_value1 = qcomplex_t(0.5, 0) * (qcomplex_t(1, 0) + exp(qcomplex_t(0, theta)));
    auto tmp_value2 = qcomplex_t(0.5, 0) * (qcomplex_t(1, 0) - exp(qcomplex_t(0, theta)));

    gate_matrix[0] = tmp_value1;
    gate_matrix[1] = qcomplex_t(0, -1) * exp(qcomplex_t(0, lambda)) * tmp_value2;
    gate_matrix[2] = qcomplex_t(0, 1) * exp(qcomplex_t(0, phi)) * tmp_value2;
    gate_matrix[3] = exp(qcomplex_t(0, phi + lambda)) * tmp_value1;

    gamma = 2 * acos(abs(gate_matrix[0]));
	if ((abs(gate_matrix[0]) > DBL_EPSILON) && (abs(gate_matrix[2]) > DBL_EPSILON))
    /*if (abs(gate_matrix[0] * gate_matrix[1]) > 1e-20)*/
    {
        beta = argc(gate_matrix[2] / gate_matrix[0]);
        delta = argc(gate_matrix[3] / gate_matrix[2]);
        alpha = beta / 2 + delta / 2 + argc(gate_matrix[0]);
    }
    else if (abs(gate_matrix[0]) > DBL_EPSILON)
    {
        beta = argc(gate_matrix[3] / gate_matrix[0]);
        delta = 0;
        alpha = beta / 2 + argc(gate_matrix[0]);
    }
    else
    {
        beta = argc(gate_matrix[2] / gate_matrix[1]) + PI;
        delta = 0;
        alpha = argc(gate_matrix[1]) + beta / 2 - PI;
    }
}
