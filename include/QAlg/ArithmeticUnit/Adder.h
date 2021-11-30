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
    CNULL = 0b00000000,  /* both c_in and c_out are ignored */
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
};

/**
 * @brief CDKMRippleAdder
 * CDKM short for authers' family name in ref[1] 
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
 *  [1] Cuccaro et al., A new quantum ripple-carry addition circuit, 2004.
 * arXiv:quant-ph/0410184 <https://arxiv.org/pdf/quant-ph/0410184.pdf>
 */
class CDKMRippleAdder : public AbstractAdder
{
public:
    CDKMRippleAdder() = default;
    virtual ~CDKMRippleAdder() = default;
    /**
     * @param aux aux[0] is c_in
     * @note resulte saved at b, a not changed
     *       if addition overflow aux[0] or aux[n-1] will be |1>, else not changed 
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT);

private:
    QCircuit MAJ(Qubit *c, Qubit *b, Qubit *a);
    QCircuit UMA(Qubit *c, Qubit *b, Qubit *a);
};

/**
 * @brief DraperQFTAdder
 *        a_0:   ─────────■──────■────────────────────────■────────────────
 *                        │      │                        │
 *        a_1:   ─────────┼──────┼────────■──────■────────┼────────────────
 *               ┌──────┐ │P(π)  │        │      │        │       ┌───────┐
 *        b_0:   ┤0     ├─■──────┼────────┼──────┼────────┼───────┤0      ├
 *               │      │        │P(π/2)  │P(π)  │        │       │       │
 *        b_1:   ┤1 qft ├────────■────────■──────┼────────┼───────┤1 iqft ├
 *               │      │                        │P(π/2)  │P(π/4) │       │
 *       cout_0: ┤2     ├────────────────────────■────────■───────┤2      ├
 *               └──────┘                                         └───────┘
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
     * @note result is save at b, a and aux not changed 
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT);

private:
    QCircuit AQFT(QVec qvec, int aqft_threshold);
    QCircuit inverseAQFT(QVec qvec, int aqft_threshold);
};

/**
 * @brief VBERippleAdder
 *                 ┌────────┐                       ┌───────────┐┌──────┐
 *          cin_0: ┤0       ├───────────────────────┤0          ├┤0     ├ cin_0
 *                 │        │                       │           ││      │
 *            a_0: ┤1       ├───────────────────────┤1          ├┤1     ├ a_0
 *                 │        │┌────────┐     ┌──────┐│           ││  Sum │
 *            a_1: ┤        ├┤1       ├──■──┤1     ├┤           ├┤      ├ a_1
 *                 │        ││        │  │  │      ││           ││      │
 *            b_0: ┤2 Carry ├┤        ├──┼──┤      ├┤2 Carry_dg ├┤2     ├ s_0
 *                 │        ││        │┌─┴─┐│      ││           │└──────┘
 *            b_1: ┤        ├┤2 Carry ├┤ X ├┤2 Sum ├┤           ├──────── s_1
 *                 │        ││        │└───┘│      ││           │
 *         cout_0: ┤        ├┤3       ├─────┤      ├┤           ├──────── cout_0
 *                 │        ││        │     │      ││           │
 *       helper_0: ┤3       ├┤0       ├─────┤0     ├┤3          ├──────── helper_0
 *                 └────────┘└────────┘     └──────┘└───────────┘
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
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT);

private:
    QCircuit carryModule(Qubit *c_in, Qubit *a, Qubit *b, Qubit *c_out);
    QCircuit carryDaggerModule(Qubit *c_in, Qubit *a, Qubit *b, Qubit *c_out);
    QCircuit sumModule(Qubit *c_in, Qubit *a, Qubit *b);
};
//-----------------------------------------------------------

/**
 * @brief Draper Quantum Carry Look Ahead Adder
 * 
 * CLA adder circuit depth is O(log n) 
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
     * in in-place mode, resulte saved at b, with a, aux not changed
     */
    virtual QCircuit QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode = ADDER_MODE::CINOUT);

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

private:
    QCircuit perasGate(Qubit *a, Qubit *b, Qubit *aux);
};

QPANDA_END