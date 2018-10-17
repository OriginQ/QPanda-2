#ifndef _GPU_QUANTUM_GATE_H
#define _GPU_QUANTUM_GATE_H
#include "config.h"


#ifdef USE_CUDA

#include "QuantumGates.h"
#include "GPUStruct.h"

class GPUQuantumGates : public QuantumGates
{
	size_t miQbitNum;
	bool mbIsInitQState;
public:

	GPUQuantumGates() :mbIsInitQState(false) {}
	~GPUQuantumGates();

	GATEGPU::QState mvQuantumStat;
	GATEGPU::QState mvCPUQuantumStat;

	size_t getQStateSize();

	/*************************************************************************************************************
	Name:        getQState
	Description: get quantum state
	Argin:       pQuantumProParam      quantum program param.
	Argout:      sState                state string
	return:      quantum error
	*************************************************************************************************************/
	bool getQState(string & sState, QuantumGateParam *pQuantumProParam);

	/*************************************************************************************************************
	Name:        Hadamard
	Description: Hadamard gate,the matrix is:[1/sqrt(2),1/sqrt(2);1/sqrt(2),-1/sqrt(2)]
	Argin:       psi          reference of vector<complex<double>> which contains quantum state information.
	qn           qubit number that the Hadamard gate operates on.
	error_rate   the errorrate of the gate
	Argout:      psi          quantum state after the gate
	return:      quantum error
	*************************************************************************************************************/
	QError Hadamard(size_t qn, bool isConjugate,
		double error_rate);
	QError Hadamard(size_t qn, Qnum& vControlBit,
		bool isConjugate, double error_rate);
	QError X(size_t qn, bool isConjugate,
		double error_rate);
	QError X(size_t qn, Qnum& vControlBit,
		bool isConjugate, double error_rate);
	QError Y(size_t qn, bool isConjugate,
		double error_rate);
	QError Y(size_t qn, Qnum& vControlBit,
		bool isConjugate, double error_rate);
	QError Z(size_t qn, bool isConjugate,
		double error_rate);
	QError Z(size_t qn, Qnum& vControlBit,
		bool isConjugate, double error_rate);
	QError T(size_t qn, bool isConjugate,
		double error_rate);
	QError T(size_t qn, Qnum& vControlBit,
		bool isConjugate, double error_rate);

	QError S(size_t qn, bool isConjugate,
		double error_rate);
	QError S(size_t qn, Qnum& vControlBit,
		bool isConjugate, double error_rate);


	QError RX_GATE(size_t qn, double theta,
		bool isConjugate, double error_rate);
	QError RX_GATE(size_t qn, double theta,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);
	QError RY_GATE(size_t qn, double theta,
		bool isConjugate, double error_rate);
	QError RY_GATE(size_t qn, double theta,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);
	QError RZ_GATE(size_t qn, double theta,
		bool isConjugate, double error_rate);
	QError RZ_GATE(size_t qn, double theta,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);
	QError CNOT(size_t qn_0, size_t qn_1,
		bool isConjugate, double error_rate);
	QError CNOT(size_t qn_0, size_t qn_1,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);
	QError CR(size_t qn_0, size_t qn_1, double theta,
		bool isConjugate, double error_rate);
	QError CR(size_t qn_0, size_t qn_1,
		Qnum& vControlBit,
		double theta,
		bool isConjugate,
		double error_rate);
	QError CZ(size_t qn_0, size_t qn_1,
		bool isConjugate, double error_rate);
	QError CZ(size_t qn_0, size_t qn_1,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);

	QError iSWAP(size_t qn_0, size_t qn_1,
		double theta,
		bool isConjugate,
		double error_rate);
	QError iSWAP(size_t qn_0, size_t qn_1,
		Qnum& vControlBit,
		double theta,
		bool isConjugate,
		double error_rate);

	QError iSWAP(size_t qn_0, size_t qn_1,
		bool isConjugate,
		double error_rate);
	QError iSWAP(size_t qn_0, size_t qn_1,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);
	QError SqiSWAP(size_t qn_0, size_t qn_1,
		bool isConjugate,
		double error_rate);
	QError SqiSWAP(size_t qn_0, size_t qn_1,
		Qnum& vControlBit,
		bool isConjugate,
		double error_rate);




	QError Reset(size_t qn);
	bool qubitMeasure(size_t qn);
	QError pMeasure(Qnum& qnum, vector<pair<size_t, double>> &mResult,
		int select_max = -1);

	QError pMeasure(Qnum& qnum, vector<double> &mResult);

	QError initState(QuantumGateParam *);

	QError endGate(QuantumGateParam *pQuantumProParam,
		QuantumGates * pQGate);

	QError unitarySingleQubitGate(size_t qn, QStat& matrix,
		bool isConjugate,
		double error_rate);

	QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
		QStat& matrix,
		bool isConjugate,
		double error_rate);

	QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
		QStat& matrix,
		bool isConjugate,
		double error_rate);

	QError controlunitaryDoubleQubitGate(size_t qn_0,
		size_t qn_1,
		Qnum& qnum,
		QStat& matrix,
		bool isConjugate,
		double error_rate);

};


#endif // USE_CUDA

#endif

