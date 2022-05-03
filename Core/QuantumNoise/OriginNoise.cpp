#include "Core/QuantumNoise/OriginNoise.h"

USING_QPANDA

OriginNoise::OriginNoise(QVec vec, QStat matrix)
    : m_vec(vec),
      m_op(matrix)
{
}

QVec OriginNoise::get_qvec()
{
    return m_vec;
}

QStat OriginNoise::get_ops()
{
    return m_op;
}