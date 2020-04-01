#ifndef CHEMIQ_H
#define CHEMIQ_H

#include "Components/ChemiQ/Psi4Wrapper.h"
#include "Components/ChemiQ/ChemiqUtil.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/Variational/Optimizer.h"

QPANDA_BEGIN

class QuantumMachine;

/**
* @brief ChemiQ Algorithm class
* @ingroup QAlgChemiQ
*/
class DLLEXPORT ChemiQ
{
public:
    /**
    * @brief  Constructor of ChemiQ
    */
    ChemiQ();
    ~ChemiQ();

    /**
    * @brief  Initialize the quantum chemistry calculation
    * @param[in] std::string The dir of the psi4 chemistry calculation package 
    */
    void initialize(const std::string& dir);

    /**
    * @brief  Finalize the quantum chemistry calculation
    */
    void finalize();
    /**
    * @brief  Set the molecular model to calculate 
    * @param[in]  std::string  molecule model
    */
    void setMolecule(const std::string &molecule)
    {
        m_molecules.clear();
        m_molecules.push_back(molecule);
    }
    /**
    * @brief  Set the molecular model to calculate
    * @param[in]  vector_s  molecule model
    * @see vector_s
    */
    void setMolecules(const vector_s& molecules)
    {
        m_molecules = molecules;
    }
    /**
    * @brief  Set the multiplicity of the molecular model
    * @param[in]  int  multiplicity
    */
    void setMultiplicity(int multiplicity)
    {
        m_psi4_wapper.setMultiplicity(multiplicity);
    }
    /**
    * @brief  Set the charge of the molecular model
    * @param[in]  int  charge
    */
    void setCharge(int charge)
    {
        m_psi4_wapper.setCharge(charge);
    }
    /**
    * @brief  Set the calculation basis
    * @param[in]  std::string basis
    */
    void setBasis(const std::string &basis)
    {
        m_psi4_wapper.setBasis(basis);
    }
    /**
    * @brief  Set the transform type from Fermion operator to Pauli operator
    * @param[in]  TransFormType transform type
    * @see TransFormType
    */
    void setTransformType(TransFormType type)
    {
        m_transform_type = type;
    }
    /**
    * @brief  Set the ucc type to contruct the Fermion operator
    * @param[in]  UccType ucc type
    * @see UccType
    */
    void setUccType(UccType ucc_type)
    {
        m_ucc_type = ucc_type;
    }
    /**
    * @brief  Set the optimizer type
    * @param[in]  OptimizerType optimizer type
    * @see OptimizerType
    */
    void setOptimizerType(OptimizerType optimizer_type)
    {
        m_optimizer_type = optimizer_type;
    }
    /**
    * @brief  Set the optimizer iteration number
    * @param[in]  size_t iteration number
    */
    void setOptimizerIterNum(size_t iter_num)
    {
        m_optimizer_iter_num = iter_num;
    }
    /**
    * @brief  Set the optimizer function callback number
    * @param[in]  size_t function callback number
    */
    void setOptimizerFuncCallNum(size_t num)
    {
        m_optimizer_func_call_num = num;
    }
    /**
    * @brief  Set the optimizer xatol.It is the Absolute error in xopt between 
    *         iterations that is acceptable for convergence.
    * @param[in]  double absolute error between iterations
    */
    void setOptimizerXatol(double value)
    {
        m_xatol = value;
    }
    /**
    * @brief  Set the optimizer fatol.It is the Absolute error in func(xopt) 
    *         between iterations that is acceptable for convergence.
    * @param[in]  double absolute error between func(xopt)
    */
    void setOptimizerFatol(double value)
    {
        m_fatol = value;
    }
    /**
    * @brief  Whether to print the optimized log to the terminal.
    * @param[in]  bool enable
    */
    void setOptimizerDisp(bool enable)
    {
        m_disp = enable;
    }
    /**
    * @brief  Set the learing rate when using Gradient optimizer
    * @param[in]  double learing rate
    */
    void setLearningRate(double learning_rate)
    {
        m_learning_rate = learning_rate;
    }
    /**
    * @brief  Set the evolution time when doing hamiltonian simulation
    * @param[in]  double evolution time
    */
    void setEvolutionTime(double t)
    {
        m_evolutionTime = t;
    }
    /**
    * @brief  Set the hamiltonian simulation slices
    *         (e^iAt/n*e^iBt/n)^n, n is the slices
    * @param[in]  double hamiltonian simulation slices
    */
    void setHamiltonianSimulationSlices(size_t slices)
    {
        m_hamiltonian_simulation_slices = slices;
    }
    /**
    * @brief  Set the directory to save the calculated data.
    *         If it's a not exist dir data will not be saved.
    * @param[in]  std::string dir
    */
    void setSaveDataDir(const std::string dir)
    {
        m_save_data_dir = dir;
    }
    /**
    * @brief  Set the quantum machine type
    * @param[in]  QMachineType quantum machine type
    * @see QMachineType
    */
    void setQuantumMachineType(QMachineType type)
    {
        m_quantum_machine_type = type;
    }
    /**
    * @brief  Set random default optimizer paramter 
    * @param[in]  bool enable
    */
    void setRandomPara(bool enable)
    {
        m_random_para = enable;
    }
    /**
    * @brief  Set the default optimizer paramter by the given paramter
    * @param[in]  vecotr_d default paramter
	* @see vector_d
    */
    void setDefaultOptimizedPara(const vector_d &para)
    {
        m_default_optimized_para = para;
    }
    /**
    * @brief  Set to get hamiltonian from file
    * @param[in]  bool enable
    */
    void setToGetHamiltonianFromFile(bool enable)
    {
        m_hamiltonian_in_file = enable;
    }
    /**
    * @brief  Set hamiltonian generation only
    * @param[in]  bool enable
    */
    void setHamiltonianGenerationOnly(bool enable)
    {
        m_hamiltonian_gen_only = enable;
    }
    /**
    * @brief  get qubits num with the above config.
    * @return  int -1:means failed.
    */
    int getQubitsNum();
    /**
    * @brief  exec molecule calculate.
    * @return  bool true:success; false:failed
    */
    bool exec();
    /**
    * @brief  get last error.
    * @return  std::string last error
    */
    std::string getLastError() const
    {
        return m_last_err;
    }
    /**
    * @brief  get calculated energies of the molecules.
    * @return  vector_d energies
    */
    vector_d getEnergies() const
    {
        return m_energies;
    }
private:
    void initOptimizedPara(size_t size);
    size_t getMoleculerElectronNum(const std::string &moleculer) const;
    QCircuit prepareInitialState(QVec &qlist, size_t en);
    QOptimizationResult optimizeByGradient();
    QOptimizationResult optimizeByNoGradient();

    /*

    state is quantum state,paulis is a map like '1:X 2:Y 3:Z'.
    parity check of partial element of state, number of paulis are
    invloved position

    */
    bool ParityCheck(size_t state, const QTerm &paulis) const;

    std::shared_ptr<Variational::Optimizer> 
        genGradientOptimizer(std::vector<var> &para);
    QResultPair callVQE(
            const vector_d &para,
            const QHamiltonian &hamiltonian);
    /*

    Get expectation of one PauliOpComplex.

    */
    double getExpectation(
        const QHamiltonian &unitary_cc,
        const QHamiltonianItem &component);

    bool testTermination(
        const vector_d& p1,
        const vector_d& p2,
        double e1,
        double e2) const;
    bool saveResult() const;
    bool saveMoleculeOptimizedResult(
        size_t index,
        const std::string &molecule,
        const std::string &pauli,
        const QOptimizationResult& result) const;
    bool writeBaseData() const;
    bool writeExecLog(bool exec_flag) const;
    bool writeProgress(size_t iter_num) const;
    bool updateBaseData();
    bool getLastIthMoleculeResult(size_t index);
    bool getLastIthMoleculeOptimizedPara(const std::string& filename);
    bool getDataFromPsi4(size_t index);
    bool saveGradientOptimizerCacheFile(
        const vector_d& best_paras,
        const vector_d& cur_paras,
        const vector_d& last_paras,
        double b_value,
        double c_value,
        double l_value,
        size_t cur_iter
    ) const;
    bool restoreGradientOptimizerParaFromCache(
        vector_d& best_paras,
        vector_d& cur_paras,
        vector_d& last_paras,
        double &b_value,
        double &c_value,
        double &l_value,
        size_t &cur_iter
    );

    vector_d getVectorFromString(const std::string& para) const;
private:
    QMachineType m_quantum_machine_type{CPU};

    Psi4Wrapper m_psi4_wapper;
    std::string m_chemiq_dir;
    std::string m_last_err;

    vector_s m_molecules;

    TransFormType m_transform_type{TransFormType::Jordan_Wigner};
    UccType m_ucc_type{UccType::UCCS};
    OptimizerType m_optimizer_type{OptimizerType::NELDER_MEAD};
    size_t m_optimizer_iter_num{1000};
    size_t m_optimizer_func_call_num{1000};
    double m_xatol{1e-4};
    double m_fatol{1e-4};
    double m_learning_rate{ 0.2 };
    double m_evolutionTime{ 1.0 };
    size_t m_hamiltonian_simulation_slices{ 3 };
    std::string m_save_data_dir;
    vector_d m_energies;
	std::vector<Eigen::MatrixXi> m_BK;

    std::unique_ptr<QuantumMachine> m_machine;
    QVec m_qubit_vec;
    size_t m_qn{0};
    size_t m_electron_num{0};
    size_t m_para_num{0};

    PauliOperator m_pauli;

    size_t m_func_calls{ 0 };
    int m_process_i{ 0 };
    OriginCollection m_optimizer_data_db;

    bool m_disp{ false };
    bool m_random_para{ false };
    vector_d m_default_optimized_para;
    vector_d m_optimized_para;
    bool m_break_restoration{false};
    size_t m_last_iters{0};
    size_t m_last_fcalls{0};
    int m_last_process_i{ -1 };
    bool m_hamiltonian_in_file{ false };
    bool m_hamiltonian_gen_only{ false };

};

QPANDA_END
#endif // CHEMIQ_H
