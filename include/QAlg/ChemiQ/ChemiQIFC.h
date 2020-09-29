#pragma once

#include "QAlg/ChemiQ/ChemiQ.h"

using namespace QPanda;

extern "C" {
	/**
	* @brief Initialize the quantum chemistry calculation
	* @param[in] std::string The dir of the psi4 chemistry calculation package
	* @return ChemiQ* a ChemiQ object ptr
	*/
    DLLEXPORT ChemiQ* initialize(char* dir);

	/**
	* @brief  Finalize the quantum chemistry calculation
	* @param[in] ChemiQ* the ChemiQ object ptr will be finalized
	*/
    DLLEXPORT void finalize(ChemiQ* chemiq);

	/**
	* @brief Set the molecular model to calculate 
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in]  std::string  molecule model
	*/
    DLLEXPORT void setMolecule(ChemiQ* chemiq, char* molecule);

	/**
	* @brief Set the molecular model to calculate
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] char* the molecules ptr
	*/
    DLLEXPORT void setMolecules(ChemiQ* chemiq, char* molecules);

	/**
	* @brief Set the multiplicity of the molecular model
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in]  int  multiplicity
	*/
    DLLEXPORT void setMultiplicity(ChemiQ* chemiq, int multiplicity);

	/**
	* @brief Set the charge of the molecular model
	* @param[in] ChemiQ* the target ChemiQ object ptr
	* @param[in] int charge
	*/
    DLLEXPORT void setCharge(ChemiQ* chemiq, int charge);

	/**
	* @brief Set the calculation basis
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] std::string basis
	*/
    DLLEXPORT void setBasis(ChemiQ* chemiq, char* basis);

	/**
	* @brief Set Eq Tolerance
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] double EqTolerance value
	*/
	DLLEXPORT void setEqTolerance(ChemiQ* chemiq, double val);

	/**
	* @brief Set the transform type from Fermion operator to Pauli operator
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] int transform type
	* @see QPanda::TransFormType
	*/
    DLLEXPORT void setTransformType(ChemiQ* chemiq, int type);

	/**
	* @brief Set the ucc type to contruct the Fermion operator
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] int ucc type
	* @see QPanda::UccType
	*/
    DLLEXPORT void setUccType(ChemiQ* chemiq, int type);

	/**
	* @brief Set the optimizer type
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] int optimizer type
	* @see QPanda::OptimizerType
	*/
    DLLEXPORT void setOptimizerType(ChemiQ* chemiq, int type);

	/**
	* @brief Set the optimizer iteration number
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] size_t iteration number
	*/
    DLLEXPORT void setOptimizerIterNum(ChemiQ* chemiq, int value);

	/**
	* @brief  Set the optimizer function callback number
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in]  size_t function callback number
	*/
    DLLEXPORT void setOptimizerFuncCallNum(ChemiQ* chemiq, int value);

	/**
	* @brief Set the optimizer iteration number
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] size_t iteration number
	*/
    DLLEXPORT void setOptimizerXatol(ChemiQ* chemiq, double value);

	/**
	* @brief Set the optimizer fatol.It is the Absolute error in func(xopt) 
    *         between iterations that is acceptable for convergence.
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] double absolute error between func(xopt)
	*/
    DLLEXPORT void setOptimizerFatol(ChemiQ* chemiq, double value);

	/**
	* @brief Set the learing rate when using Gradient optimizer
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] double learing rate
	*/
    DLLEXPORT void setLearningRate(ChemiQ* chemiq, double value);

	/**
	* @brief Set the evolution time when doing hamiltonian simulation
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] double evolution time
	*/
    DLLEXPORT void setEvolutionTime(ChemiQ* chemiq, double value);

	/**
	* @brief Set the hamiltonian simulation slices
    *         (e^iAt/n*e^iBt/n)^n, n is the slices
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] double hamiltonian simulation slices
	*/
    DLLEXPORT void setHamiltonianSimulationSlices(ChemiQ* chemiq, int value);

	/**
	* @brief Set the directory to save the calculated data.
    *         If it's a not exist dir data will not be saved.
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @param[in] std::string dir
	*/
    DLLEXPORT void setSaveDataDir(ChemiQ* chemiq, char* dir);

    /**
    * @brief  get qubits num with the above config.
    * @param[in]  ChemiQ* the target ChemiQ object ptr
    * @return  int -1:means failed.
    */
    DLLEXPORT int getQubitsNum(ChemiQ* chemiq);

	/**
	* @brief exec molecule calculate.
	* @param[in]  ChemiQ* the target ChemiQ object ptr
	* @return  bool true:success; false:failed
	*/
    DLLEXPORT bool exec(ChemiQ* chemiq);

    /**
    * @brief  get last error.
    * @param[in]  ChemiQ* the target ChemiQ object ptr
    * @return  char* last error
    */
    DLLEXPORT const char* getLastError(ChemiQ* chemiq);
}
