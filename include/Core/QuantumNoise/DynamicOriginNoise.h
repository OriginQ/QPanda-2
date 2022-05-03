#pragma once

#include <type_traits>
#include "OriginNoise.h"

QPANDA_BEGIN

class AbstractOpsGenerator
{
public:
    AbstractOpsGenerator() = default;
    virtual ~AbstractOpsGenerator() = default;
    virtual QStat generate_op() = 0;
};

/**
 * @brief Dynamicly generate noise matrix when call get_ops()
 *
 * @tparam Generator
 * Generator type should derived from AbstractOpsGenerator
 */
template <typename Generator>
class DynamicOriginNoise : public OriginNoise
{
    static_assert(std::is_base_of<AbstractOpsGenerator, Generator>::value, "Generator type should derive from AbstractOpsGenerator");

public:
    DynamicOriginNoise(QVec vec, const Generator &gen)
        : OriginNoise(vec),
          m_generator(gen)
    {
    }

    virtual ~DynamicOriginNoise() = default;

    virtual QStat get_ops()
    {
        if (m_op.empty())
        {
            m_op = m_generator.generate_op();
        }
        return m_op;
    }

private:
    Generator m_generator;
};

QPANDA_END