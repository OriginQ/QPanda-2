#ifndef __SZEGEDY_WALK_H__
#define __SZEGEDY_WALK_H__

#include "QPanda.h"
#include "Core/Utilities/Tools/QMatrixDef.h"

QPANDA_BEGIN

class SzegedyWalk {
public:
	SzegedyWalk() {}
	virtual ~SzegedyWalk() {}

    QMatrixXcd expm(QMatrixXcd H);
    QMatrixXcd expm(QMatrixXd H);

	QMatrixXcd expm_i(QMatrixXcd H);
	QMatrixXcd expm_i(QMatrixXd H);

private:

};

QPANDA_END
#endif // !__SZEGEDY_WALK_H__
