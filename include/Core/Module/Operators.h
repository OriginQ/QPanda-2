#include "Module.h"

QPANDA_BEGIN

/**
* @brief H quantum gate operation for each quantum bit
* @ingroup Module
* @param[in] qvec qubit vector
* @return QCircuit   quantum circuit
*/
QCircuit h(qvec qs) {
	QCircuit c;
	for (size_t i = 0; i < qs.size(); ++i) {
		c << H(qs[i]);
	}
	return c;
}

QPANDA_END