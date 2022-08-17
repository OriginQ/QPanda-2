/*
 * This preconditioner is coded by ZhuY!
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/Core>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include "BasicFunctionByOrigin.h"

	std::pair< Eigen::MatrixXd, Eigen::VectorXd >  DynamicSparseApproximateInverse(Eigen::MatrixXd& Ae, Eigen::VectorXd& b, double epsilon, int SparseIndex, Eigen::MatrixXd& M);

	/*  \brief Get the diagonal matrix
	 *  \param[in] Matrix to be preconditioned.
	 * \return pair of digonal matrix and preconditioned matrix.
	 */
	void DiagonalScalingNew(SparseQMatrix<double>& m, vector<vector<double>>& Diag);

	/*  \brief Get the Sparse Approximate Inverse
	 *  \param[in] Matrix A, sparsity.
	 * \return matrix M.
	 */
	SparseQMatrix<double>  StaticSparseApproximateInverse(SparseQMatrix<double>& Ae, int SparseIndex);

	/*  \brief Get the Dynamic Sparse Approximate Inverse
	 *  \param[in] Matrix A, epsilon, sparsity.
	 * \return matrix M and M.A.
	 */
	SparseQMatrix<double> DynamicSparseApproximateInverse(SparseQMatrix<double>& Ae, SparseQMatrix<double>& M, double epsilon, int SparseIndex);
	
	//int DynamicSparseApproximateInverse2(Eigen::MatrixXd& Ae, Eigen::VectorXd& b, double epsilon, int SparseIndex) { return 0; }
	/*  \brief Get the Power Sparse Approximate Inverse
	 *  \param[in] Matrix A, epsilon, sparsity.
	 * \return matrix M and A.M.
	 */
	 //pair<SparseQMatrix<double>, SparseQMatrix<double>> PowerSparseApproximateInverseRight(SparseQMatrix<double> Ae, double epsilon, int SparseIndex);

	 /*  \brief Get the Dynamic Sparse Approximate Inverse
	  *  \param[in] Matrix A, epsilon, sparsity.
	  * \return matrix M and M.A.
	  */
	SparseQMatrix<double> PowerSparseApproximateInverseLeft(SparseQMatrix<double>& Ae, SparseQMatrix<double>& M, double epsilon, int SparseIndex);


	/*  \brief Get the jacobi precondition
	 *  \param[in] Matrix A, VectorXd b.
	 * \return matrix M and M.A.
	 */
	SparseQMatrix<double> JacobiPrecondition(SparseQMatrix<double>& Ae, VectorXd& b);

	/*  \brief Get the jacobi precondition
	 *  \param[in] Matrix A, vector<double> b.
	 * \return matrix M and M.A.
	 */
	SparseQMatrix<double> JacobiPrecondition(SparseQMatrix<double>& Ae, vector<double>& b);

	enum solvers {
		HHL = 0,
		SUB_HHL = 1,
		VQLS = 2,
		RESTART_VQLS = 3
	};

	/*  \brief calculate the preconditioner
	 *  \param[in] SparseQMatrix M, to be preconditioned.
	 *  \param[in] SparseQMatrix A, the new preconditioned sparse matrix.
	 *  \param[in] vector b.
	 *  \param[in] kind linear system precondtioner
	 *  \param[in] sparse index for SPAI
	 *  \param[in] epsilon for dynamic SPAI
	 * \return matrix M and M.A.
	 */
	void precondition(SparseQMatrix<double>& M, SparseQMatrix<double>& A, vector<double>& b,
		int KindLinPrec, int SparseIndex, double DySPepsilon);
