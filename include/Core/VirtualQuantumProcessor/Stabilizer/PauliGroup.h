#pragma once

#include <iostream>
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

const uint64_t m_val_0 = 0ull;
const uint64_t m_val_1 = 1ull;

const size_t MAX_BLOCK_SIZE = 64;

class BinaryChunk 
{
public:

    BinaryChunk();
    BinaryChunk(uint64_t length);
    BinaryChunk(std::vector<uint64_t> data);

    void set_val(bool value, uint64_t index);
    void set_val_0(uint64_t index) { set_val(m_val_0, index); };
    void set_val_1(uint64_t index) { set_val(m_val_1, index); };

    void flip(uint64_t index);
    void swap(BinaryChunk &rhs);
    void reset(bool reset_zero = true);

    uint64_t get_length() const { return m_length; };
    std::vector<uint64_t> get_data() const { return m_data; };

    bool compare(const BinaryChunk& other) const;
    bool operator[](const uint64_t index) const;
    BinaryChunk &operator+=(const BinaryChunk &rhs);

private:

    uint64_t m_length;
    std::vector<uint64_t> m_data;
};

inline bool operator==(const BinaryChunk &lhs, const BinaryChunk &rhs) { return lhs.compare(rhs); };
inline bool operator!=(const BinaryChunk &lhs, const BinaryChunk &rhs) { return !lhs.compare(rhs); };

class PauliGroup
{
public:

    BinaryChunk X;
    BinaryChunk Z;

    PauliGroup() : X(0), Z(0) {};
    PauliGroup(uint64_t length) : X(length), Z(length) {};

    // exponent g of i such that P(x1,z1) P(x2,z2) = i^g P(x1+x2,z1+z2)
    static int phase_exponent(const PauliGroup& lhs, const PauliGroup& rhs);

};



QPANDA_END
