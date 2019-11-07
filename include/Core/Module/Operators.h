#include "Module.h"

QPANDA_BEGIN

QCircuit h(qvec qs) {
	QCircuit c;
	for (size_t i = 0; i < qs.size(); ++i) {
		c << H(qs[i]);
	}
	return c;
}

QPANDA_END