#include "Core/QuantumNoise/QNoise.h"

USING_QPANDA

QNoise::QNoise(std::shared_ptr<AbstractQNoiseNode> node)
    : m_noise_node(node)
{
}

std::shared_ptr<AbstractQNoiseNode> QNoise::getImplementationPtr()
{
  return m_noise_node;
}

QVec QNoise::get_qvec()
{
  return m_noise_node->get_qvec();
}

QStat QNoise::get_ops()
{
  return m_noise_node->get_ops();
}
