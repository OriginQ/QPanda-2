#include "Core/VirtualQuantumProcessor/Stabilizer/PauliGroup.h"

USING_QPANDA

BinaryChunk::BinaryChunk() : m_length(0), m_data(0)
{};

BinaryChunk::BinaryChunk(uint64_t length) : m_length(length), m_data((length - 1) / MAX_BLOCK_SIZE + 1, m_val_0)
{};

BinaryChunk::BinaryChunk(std::vector<uint64_t> data) : m_length(data.size()), m_data(data)
{};

bool BinaryChunk::operator[](const uint64_t index) const
{
    auto q = index / MAX_BLOCK_SIZE;
    auto r = index % MAX_BLOCK_SIZE;

    return (m_data[q] & (m_val_1 << r)) != 0;
}

BinaryChunk &BinaryChunk::operator+=(const BinaryChunk &rhs)
{
    const auto size = m_data.size();
    for (size_t i = 0; i < size; i++)
        m_data[i] ^= rhs.m_data[i];

    return (*this);
}

void BinaryChunk::reset(bool reset_zero)
{ 
    if (reset_zero)
        m_data.assign((m_length - 1) / MAX_BLOCK_SIZE + 1, m_val_0); 
    else
        m_data.assign((m_length - 1) / MAX_BLOCK_SIZE + 1, m_val_1);

    return;
}

bool BinaryChunk::compare(const BinaryChunk& other) const
{
    if (m_length != other.m_length)
        return false;

    for (size_t q = 0; q < m_data.size(); q++)
    {
        if (m_data[q] != other.m_data[q])
            return false;
    }

    return true;
}

void BinaryChunk::set_val(bool value, uint64_t index)
{
    auto q = index / MAX_BLOCK_SIZE;
    auto r = index % MAX_BLOCK_SIZE;

    if (value)
        m_data[q] |= (m_val_1 << r);
    else
        m_data[q] &= ~(m_val_1 << r);

    return;
}
 
void BinaryChunk::flip(const uint64_t index) 
{
    auto q = index / MAX_BLOCK_SIZE;
    auto r = index % MAX_BLOCK_SIZE;

    m_data[q] ^= (m_val_1 << r);

    return;
}

void BinaryChunk::swap(BinaryChunk &rhs)
{
    uint64_t tmp;
    tmp = rhs.m_length;
    rhs.m_length = m_length;
    m_length = tmp;

    m_data.swap(rhs.m_data);

    return;
}

int PauliGroup::phase_exponent(const PauliGroup& lhs, const PauliGroup& rhs) 
{
    int exponent = 0;
    for (size_t q = 0; q < lhs.X.get_length(); q++) 
    {
        exponent += rhs.X[q] * lhs.Z[q] * (1 + 2 * rhs.Z[q] + 2 * lhs.X[q]);
        exponent -= lhs.X[q] * rhs.Z[q] * (1 + 2 * lhs.Z[q] + 2 * rhs.X[q]);
        exponent %= 4;
    }

    if (exponent < 0)
        exponent += 4;

    return exponent;
}
