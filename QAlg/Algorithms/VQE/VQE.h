/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

VQE.h

Author: LiYe
Created in 2018-09-28


*/

#ifndef VQE_H
#define VQE_H

#include <map>
#include <memory>
#include "Core/QPanda.h"
#include "QAlg/DataStruct.h"

namespace QPanda
{
    /*

    Variational Quantum Eigensolver
    
    */
    class AbstractOptimizer;
    class PauliOperator;
    class VQE
    {
    public:
        VQE(OptimizerType optimizer = OptimizerType::POWELL);
        VQE(const std::string &optimizer);
        VQE(VQE &) = delete;
        VQE& operator =(VQE &) = delete;
        ~VQE();

        void setMoleculeGeometry(const QMoleculeGeometry &geometry)
        {
            m_geometry = geometry;
        }

        QMoleculeGeometry getMoleculeGeometry() const
        {
            return m_geometry;
        }

        void setAtomsPosGroup(const QAtomsPosGroup &pos)
        {
            m_atoms_pos_group = pos;
        }

        QAtomsPosGroup getAtomsPosGroup() const
        {
            return m_atoms_pos_group;
        }

        void setBasis(const std::string &basis)
        {
            m_basis = basis;
        }

        /*
         * Change the multiplicity (defined as 2S + 1).
         */
        void setMultiplicity(const int &multiplicity)
        {
            m_multiplicity = multiplicity;
        }

        /*
         * Change the overall molecular charge.
         */
        void setCharge(const int &charge)
        {
            m_charge = charge;
        }

        void setShots(size_t shots)
        {
            m_shots = shots;
        }

        void setPsi4Path(const std::string &path)
        {
            m_psi4_path = path;
        }


        AbstractOptimizer* getOptimizer()
        {
            return m_optimizer.get();
        }

        bool exec();

        auto getEnergies() const
        {
            return m_energies;
        }
    private:
        void QInit();
        void QFinalize();

        bool initVQE(std::string &err_msg);
        bool checkPara(std::string &err_msg);

        QMoleculeGeometry genMoleculeGeometry(const size_t &index);
        PauliOperator genPauliOperator(const std::string &s, bool *ok = nullptr);
        complex_d genComplex(const std::string &s);
        size_t getElectronNum(const QMoleculeGeometry &geometry);

        /*
        
        Coupled cluster single model.
        e.g. 4 qubits, 2 electrons
        then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3

        */
        size_t getCCS_N_Trem(const size_t &qn, const size_t &en);
        
        /*

        Coupled cluster single and double model.
        e.g. 4 qubits, 2 electrons
        then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23

        */
        size_t getCCSD_N_Trem(const size_t &qn, const size_t &en);

        /*

        Coupled cluster single model.
        J-W transform on CCS, get paulioperator
        
        */
        PauliOperator getCCS(
            const size_t &qn, 
            const size_t &en, 
            const vector_d &para_vec);

        /*

        Coupled cluster single and double model.
        J-W transform on CCSD, get paulioperator
        
        */
        PauliOperator getCCSD(
            const size_t &qn,
            const size_t &en,
            const vector_d &para_vec);
        
        /*

        fermion_op = ('a',1) or ('c',1)
        'a' is for annihilation
        'c' is for creation

        */
        PauliOperator getFermionJordanWigner(
            const char &fermion_type,
            const size_t &op_qubit);

        /*

        Generate Hamiltonian form of unitary coupled cluster based on coupled 
        cluster,H=1j*(T-dagger(T)), then exp(-jHt)=exp(T-dagger(T))

        */
        PauliOperator transCC2UCC(const PauliOperator &cc);

        /*

        Choose measurement basis,
        it means rotate all axis to z-axis

        */
        QCircuit transformBase(const QTerm &base);

        /*

        Get expectation of one paulioperator.
        
        */
        double getExpectation(
            const QHamiltonian &unitary_cc,
            const QHamiltonianItem &component);

        /*

        Get the select_max (default -1 for all distribution) largest
        component from the probability distribution with qubits (qlist)

        */
        QProbMap getProbabilites(int select_max = -1);
        /*

        String has element '0' or '1',paulis is a map like '1:X 2:Y 3:Z'.
        parity check of partial element of string, number of paulis are
        invloved position, to be repaired
        
        */
        bool PairtyCheck(const std::string &str, const QTerm &paulis);

        bool getDataFromPsi4(
                const QMoleculeGeometry &geometry,
                PauliOperator &pauli);

        std::string QMoleculeGeometry2String(const QMoleculeGeometry &geometry);

        bool psi4DataToPauli(
                const std::string &filename,
                PauliOperator &pauli);

        QResultPair callVQE(
                const vector_d &para,
                const QHamiltonian &hamiltonian);
    private:
        vector_d m_optimized_para;
        vector_d m_energies;
        QAtomsPosGroup m_atoms_pos_group;

        std::string m_psi4_path;
        std::string m_basis{"sto-3g"};

        int m_multiplicity{1};
        int m_charge{0};

        size_t m_shots{1000};
        size_t m_qn{0};
        size_t m_electron_num{0};

        QMoleculeGeometry m_geometry;

        QVec m_qubit_vec;
        std::vector<ClassicalCondition> m_cbit_vec;
        QProg m_prog;

        std::unique_ptr<AbstractOptimizer> m_optimizer;
    };

}

#endif // VQE_H
