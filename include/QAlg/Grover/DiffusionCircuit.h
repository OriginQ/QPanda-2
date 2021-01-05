#ifndef  _DIFFUSION_CIRCUIT_H
#define  _DIFFUSION_CIRCUIT_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>

QPANDA_BEGIN

class AbstractDiffusionOperator
{
public:
	virtual QCircuit build_diffusion_circuit(const QVec &qvec) = 0;

};

class DiffusionCirBuilder : public AbstractDiffusionOperator
{
public:
	DiffusionCirBuilder(){}
	~DiffusionCirBuilder() {}

	QCircuit build_diffusion_circuit(const QVec &qvec) override {
		vector<Qubit*> controller(qvec.begin(), --(qvec.end()));
		QCircuit c;
		c << apply_QGate(qvec, H);
		c << apply_QGate(qvec, X);
		c << Z(qvec.back()).control(controller);
		c << apply_QGate(qvec, X);
		c << apply_QGate(qvec, H);

		return c;
	}

private:

};

inline QCircuit build_diffusion_circuit(const QVec &qvec, AbstractDiffusionOperator &op) {
	return op.build_diffusion_circuit(qvec);
}

QPANDA_END

#endif