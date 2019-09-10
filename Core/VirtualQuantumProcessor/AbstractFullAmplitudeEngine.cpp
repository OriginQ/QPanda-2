#include "Core/VirtualQuantumProcessor/AbstractFullAmplitudeEngine.h"
USING_QPANDA
QError DistributedFullAmplitudeEngine::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw qvm_attributes_error("_PQGates is null");
    }

    try
    {
        _PQGates->initState(head_rank, rank_size, qubit_num);
        return QError::qErrorNone;
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw qalloc_fail(e.what());
    }
}

QStat DistributedFullAmplitudeEngine::getQState()
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    return _PQGates->getQState();
}

bool DistributedFullAmplitudeEngine::qubitMeasure(size_t qn)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    _PQGates->measureQubitOperation(qn);
    return QError::qErrorNone;
}



QError DistributedFullAmplitudeEngine::pMeasure(Qnum& qnum, prob_vec &mResult)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    else
    {
        _PQGates->PMeasureQubitOperation(qnum, mResult);
    }
    return QError::qErrorNone;
}


QError DistributedFullAmplitudeEngine::unitarySingleQubitGate(size_t qn, QStat& matrix,
    bool isConjugate,GateType)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    _PQGates->singleQubitOperation(qn, matrix, isConjugate);
    return QError::qErrorNone;
}

QError DistributedFullAmplitudeEngine::controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
    QStat& matrix, bool isConjugate,GateType)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    _PQGates->controlsingleQubitOperation(qn, qnum, matrix, isConjugate);
    return QError::qErrorNone;
}

QError DistributedFullAmplitudeEngine::unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
    QStat& matrix, bool isConjugate,GateType)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    _PQGates->doubleQubitOperation(qn_0, qn_1, matrix, isConjugate);
    return QError::qErrorNone;
}

QError DistributedFullAmplitudeEngine::controlunitaryDoubleQubitGate(size_t qn_0,
    size_t qn_1, Qnum& qnum, QStat& matrix, bool isConjugate,GateType)
{
    if (nullptr == _PQGates)
    {
        QCERR("_PQGates is null");
        throw result_get_fail("_PQGates is null");
    }
    _PQGates->controldoubleQubitOperation(qn_0, qn_1, qnum, matrix, isConjugate);

    return QError::qErrorNone;
}
