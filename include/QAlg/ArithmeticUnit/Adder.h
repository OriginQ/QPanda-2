#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"

#include <unordered_map>

QPANDA_BEGIN

enum ADDER_MODE
{
    CIN = 0b00000001,    /* c_in is taken, c_out is ignored  */
    COUT = 0b00000010,   /* c_in is ignored, c_out is taken */
    CINOUT = 0b00000011, /* both c_in and c_out is taken */
    CNULL = 0b00000000,  /* both c_in and c_out is ignored */
    OF = 0b00000100      /* both c_in and c_out is ignored, overflow tag is taken */
};
ADDER_MODE operator|(ADDER_MODE lhs, ADDER_MODE rhs);
ADDER_MODE operator&(ADDER_MODE lhs, ADDER_MODE rhs);

class AbstractAdder
{
public:
    AbstractAdder() = default;
    virtual ~AbstractAdder() = default;
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT) = 0;
    virtual size_t auxBitSize(size_t s)=0;
};

/**
 * @brief CDKMRippleAdder
 * CDKM short for authers' family name in ref[1]
 * we provide a way to control c_in bit and c_out/overflow bit separately
 *                 ┌──────┐                                          ┌──────┐
 *            c_0: ┤c_0   ├──────────────────────────────────────────┤      ├ c_0
 *                 │      │                                          │      │
 *            b_0: ┤ MAJ  ├──────────────────────────────────────────┤ UMA  ├ s_0
 *                 │      │┌──────┐                          ┌──────┐│      │
 *            a_0: ┤a     ├┤c_1   ├──────────────────────────┤      ├┤      ├ a_0
 *                 └──────┘│      │                          │      │└──────┘
 *            b_1: ────────┤ MAJ  ├──────────────────────────┤ UMA  ├──────── s_1
 *                         │      │     ┌──────┐     ┌──────┐│      │
 *            a_1: ────────┤      ├──■──┤c_2   ├─────┤      ├┤      ├──────── a_1
 *                         └──────┘  │  │      │     │      │└──────┘
 *            b_2: ──────────────────┼──┤ MAJ  ├─────┤ UMA  ├──────────────── s_2
 *                                   │  │      │     │      │
 *            a_2: ──────────────────┼──┤   c_3├──■──┤      ├──────────────── a_2
 *                                 ┌─┴─┐└──────┘┌─┴─┐└──────┘
 *       overflow: ────────────────┤ X ├────────┤ X ├──────────────────────── overflow
 *                                 └───┘        └───┘
 *
 * [1] Cuccaro et al., A new quantum ripple-carry addition circuit, 2004.
 * arXiv:quant-ph/0410184 <https://arxiv.org/pdf/quant-ph/0410184.pdf>
 */
class CDKMRippleAdder : public AbstractAdder
{
public:
    CDKMRippleAdder() = default;
    virtual ~CDKMRippleAdder() = default;
    /**
     * @param aux aux[0] is c_in, aux.back() is c_out/overflow
     * @note resulte saved at b, a not changed
     *       if addition overflow, aux.back() be flip, else not changed
     *       input aux should be |0> except in ADDER_MODE::CIN mode,
     *       ADDER_MODE::CIN only allow change aux[0] as carry in
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT) override;
    virtual size_t auxBitSize(size_t s) override;

    static QCircuit MAJ(Qubit *c, Qubit *b, Qubit *a);
    static QCircuit UMA(Qubit *c, Qubit *b, Qubit *a);

private:
    static QCircuit addWithCin(QVec a, QVec b, QVec aux, ADDER_MODE mode);
};

/**
 * @brief DraperQFTAdder
 *       c_in:   ─────────■──────■───────■───────────────────────────────────────────────────────── c_in
 *                        │      │       │
 *        a_0:   ─────────┼──────┼───────┼────────■──────■───────■───────────────────────────────── a_0
 *                        │      │       │        │      │       │
 *        a_1:   ─────────┼──────┼───────┼────────┼──────┼───────┼───────■──────■────────────────── a_1
 *               ┌──────┐ │P(π)  │       │        │P(π)  │       │       │      │         ┌───────┐
 *        b_0:   ┤0     ├─■──────┼───────┼────────■──────┼───────┼───────┼──────┼─────────┤0      ├ s_0
 *               │      │        │P(π/2) │               │P(π/2) │       │P(π)  │         │       │
 *        b_1:   ┤1 qft ├────────■───────┼───────────────■───────┼───────■──────┼─────────┤1 iqft ├ s_1
 *               │      │                │P(π/4)                 │P(π/4)        │P(π/2)   │       │
 *       c_out : ┤2     ├────────────────■───────────────────────■──────────────■─────────┤2      ├ c_out
 *               └──────┘                                                                 └───────┘
 * [1] T. G. Draper, Addition on a Quantum Computer, 2000.
 * arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>
 *
 * [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
 * arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>
 */
class DraperQFTAdder : public AbstractAdder
{
public:
    DraperQFTAdder() = default;
    virtual ~DraperQFTAdder() = default;
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param aux
     * @note result is save at b, a  not changed
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT) override;
    virtual size_t auxBitSize(size_t s) override;

    // private:
    static QCircuit AQFT(QVec qvec, int aqft_threshold);
    static QCircuit inverseAQFT(QVec qvec, int aqft_threshold);
    static QCircuit carryOverflow(QVec a, QVec b, Qubit *overflow, int aqft_threshold);
    static QCircuit inverseOverflow(QVec a, QVec b, Qubit *overflow, int aqft_threshold);
};

/**
 * @brief VBERippleAdder
 *                 ┌────────┐                                  ┌─────────┐
 *            c_0: ┤        ├──────────────────────────────────┤         ├ c_0
 *                 │        │                                  │         │
 *            a_0: ┤        ├──────────────────────────────────┤         ├ a_0
 *                 │  Carry │                                  │  Carry* │
 *            b_0: ┤        ├──────────────────────────────────┤         ├ s_0
 *                 │        │┌────────┐     ┌──────┐┌─────────┐│         │
 *            c_1: ┤        ├┤        ├─────┤      ├┤         ├┤         ├ c_1
 *                 └────────┘│        │     │      ││         │└─────────┘
 *            a_1: ──────────┤        ├──■──┤  Sum ├┤         ├─────────── a_1
 *                           │  Carry │┌─┴─┐│      ││  Carry* │
 *            b_1: ──────────┤        ├┤ X ├┤      ├┤         ├─────────── s_1
 *                           │        │└───┘└──────┘│         │
 *            c_2: ──────────┤        ├─────────────┤         ├─────────── c_2
 *                           └────────┘             └─────────┘
 *
 * [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
 * arXiv:quant-ph/0406142v1 <https://arxiv.org/pdf/quant-ph/0406142v1.pdf>
 */
class VBERippleAdder : public AbstractAdder
{
public:
    VBERippleAdder() = default;
    virtual ~VBERippleAdder() = default;
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param aux
     * @note result saved at b, a not changed
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT) override;
    virtual size_t auxBitSize(size_t s) override;

    // private:
    static QCircuit carryModule(Qubit *c_in, Qubit *a, Qubit *b, Qubit *c_out);
    static QCircuit sumModule(Qubit *c_in, Qubit *a, Qubit *b);
};
//-----------------------------------------------------------

/**
 * @brief Draper Quantum Carry Look Ahead Adder(in-place mode)
 *
 * CLA adder circuit depth is O(log n)
 * Circuit is complex, we do not draw here
 *
 * [1] Thomas G. Draper et al., A Logarithmic-Depth Quantum Carry-Lookahead Adder, 2008.
 * arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>
 */
class DraperQCLAAdder : public AbstractAdder
{
public:
    DraperQCLAAdder() = default;
    virtual ~DraperQCLAAdder() = default;
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param aux
     * @note
     * in out-of-place mode, reuslt saved at aux[0, b.size()-1], with a, b not changed
     * in in-place mode, resulte saved at b, with a not changed
     * we implemented in-place mode
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT) override;
    virtual size_t auxBitSize(size_t s) override;

    /**
     * @brief helper class for DraperQCLAAdder calculate propagate
     */
    class Propagate
    {
    public:
        Propagate() = default;
        ~Propagate() = default;
        Propagate(QVec b, QVec aux);

        /**
         * @brief get qubit of p[i,j]
         */
        Qubit *operator()(int i, int j);

    private:
        QVec m_aux;
        int m_valide_aux_id;
        std::unordered_map<int, std::unordered_map<int, Qubit *>> m_propagate;
    };

    // private:
    static QCircuit perasGate(Qubit *a, Qubit *b, Qubit *aux);
    static QCircuit propagateModule(int n, QVec generate, Propagate &propagate, bool skip_last_bit);
};

QPANDA_END