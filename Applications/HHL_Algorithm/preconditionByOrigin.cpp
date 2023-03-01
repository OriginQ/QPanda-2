/*
 * This preconditioner is coded by ZhuY!
 */

#include "preconditionByOrigin.h"
#include "QPandaConfig.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif // USE_OPENMP

using namespace Eigen;
using namespace std;

void DiagonalScalingNew(SparseQMatrix<double>& m, vector<vector<double>>& Diag)
{
	double ep1 = 0.001;
	double ep2 = 0.001;

	double r1 = 1000.0;
	double r2 = 1000.0;
	int size = m.size;

	vector<double> dR(size);
	vector<double> dC(size);
	vector<double> d1(size, 1.0);
	vector<double> d2(size, 1.0);

	while (abs(1 - r1) > ep1 && abs(1 - r2) > ep2) {
		get_dR_dC(dR, dC, m);
		update_d1_d2(dR, dC, d1, d2, m);
		vector<double> Bi(size);
		vector<double> Bj(size);
		get_Bi_Bj(Bi, Bj, m);
		std::vector<double>::iterator biggestBi = std::max_element(std::begin(Bi), std::end(Bi));
		std::vector<double>::iterator biggestBj = std::max_element(std::begin(Bj), std::end(Bj));
		r1 = *biggestBi;
		r2 = *biggestBj;
	}
	Diag.push_back(d1);
	Diag.push_back(d2);
}

SparseQMatrix<double> StaticSparseApproximateInverse(SparseQMatrix<double>& Ae, int SparseIndex) {
	int size = Ae.size;
	if (SparseIndex > size)
	{
		cout << "-- Error Exit --: Matrix dimension is " << size << "-----" << endl;
		cout << "-- Error Exit --: SparseIndex is bigger than matrix dimension-----" << endl;
		exit(100);
	}
	if (SparseIndex % 2 == 0)
	{
		cout << "-- Error Exit --: SparseIndex should be odd for Static_SPAI-----" << endl;
		exit(100);
	}

	int NoneZeroNum = 0;
	int rowFront = (SparseIndex - 1) / 2;

	SparseQMatrix<double> M;

	//JacobMatrix = Ae;

	vector<vector<pair<int, double>>> sparse_data(size);

#pragma omp parallel for num_threads(6) 
	for (int i = 0; i < size; i++)
	{
		vector<int> tempSp;
		if (i < rowFront) {
			NoneZeroNum = SparseIndex - rowFront + i;
			for (int j = 0; j < NoneZeroNum; j++) {
				tempSp.push_back(j);
			}
		}
		else if (i > size - 1 - rowFront) {
			NoneZeroNum = SparseIndex - rowFront - 1 + abs(i - size);
			for (int j = 0; j < NoneZeroNum; j++) {
				tempSp.push_back(size - 1 - j);
			}
		}
		else {
			NoneZeroNum = SparseIndex;
			for (int j = -rowFront; j < rowFront + 1; j++) {
				tempSp.push_back(i + j);
			}
		}

		sort(tempSp.begin(), tempSp.end());
		auto temp = getSubMatrix(tempSp, Ae);

		MatrixXd subA = temp.first;
		vector<int> No0Ind = temp.second;
		MatrixXd subAt = subA.transpose();
		VectorXd b = getb(i, No0Ind);
		VectorXd x = subAt.colPivHouseholderQr().solve(b);

		if (i < rowFront) {
			for (int j = 0; j < NoneZeroNum; j++)
				sparse_data[i].emplace_back(make_pair(j, x(j)));
				//M.insert(i, j) = x(j);
		}
		else if (i > size - 1 - rowFront) {
			for (int j = size - NoneZeroNum; j < size; j++)
				sparse_data[i].emplace_back(make_pair(j, x(j - size + NoneZeroNum)));
				//M.insert(i, j) = x(j - size + NoneZeroNum);
		}
		else {
			for (int j = i - rowFront; j < i - rowFront + NoneZeroNum; j++)
				sparse_data[i].emplace_back(make_pair(j, x(j - i + rowFront)));
				//M.insert(i, j) = x(j - i + rowFront);
		}
		tempSp.clear();
	}
	
	FastTransfer2SparseQMat(sparse_data.data(), size, M);

	return M;
}

SparseQMatrix<double> DynamicSparseApproximateInverse(SparseQMatrix<double>& Ae, SparseQMatrix<double>& M, double epsilon, int SparseIndex) {
    auto start0 = clock();
    sparI = read_sparsity();
    auto end0 = clock();
    double endtime = (double)(end0 - start0) / CLOCKS_PER_SEC;
    cout << "Total read_sparsity time:" << endtime << endl;

    int size = Ae.size;
    if (SparseIndex > size)
    {
        cout << "-- Error Exit --: Matrix dimension is " << size << "-----" << endl;
        cout << "-- Error Exit --: SparseIndex is bigger than matrix dimension-----" << endl;
        exit(100);
    }

    SparseQMatrix<double> MA;
    vector<vector<pair<int, double>>> sparse_dataM(size);
    vector<vector<pair<int, double>>> sparse_dataMA(size);

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < size; i++)
    {
        auto tempSp = sparI[i];
        double rnorm = 10.0;
        int count = 0;
        int jnew = 0;
        VectorXd x;
        VectorXd Amk;
        MatrixXd subA;
        auto No0Ind = getNZeroIndex(tempSp.data(), tempSp.size());

        while (rnorm > epsilon) {
            if (count != 0) {
                tempSp.push_back(jnew);
                AddNZeroIndex(No0Ind, jnew);
            }
            sort(tempSp.begin(), tempSp.end());

            int size_Sp = tempSp.size();
            int size_NNZ = No0Ind.size();

            subA = FastGetSubMatrix(No0Ind.data(), tempSp.data(), size_NNZ, size_Sp, Ae);

            VectorXd b = getb(i, No0Ind);
            x = subA.colPivHouseholderQr().solve(b);

            Amk = subA * x;
            VectorXd r = Amk - b;
            rnorm = r.norm();

            if (rnorm < epsilon || tempSp.size() >= SparseIndex)
            {
                sparI[i] = tempSp;
                break;
            }
            jnew = getRhoJ(r, rnorm, No0Ind, tempSp, Ae);
            count++;
        }

        for (int k = 0; k < tempSp.size(); k++) {
            sparse_dataM[i].emplace_back(make_pair(tempSp[k], x(k)));
        }
        for (int k = 0; k < No0Ind.size(); k++) {
            sparse_dataMA[i].emplace_back(make_pair(No0Ind[k], Amk(k)));
        }
    }
    FastTransfer2SparseQMat(sparse_dataM.data(), size, M);
    FastTransfer2SparseQMat(sparse_dataMA.data(), size, MA);
    write_sparsity();
    sparI.clear();
    return MA;
}


SparseQMatrix<double> PowerSparseApproximateInverseLeft(SparseQMatrix<double>& Ae, SparseQMatrix<double>& M, double epsilon, int SparseIndex) {
	
	sparI = read_sparsity();
	int size = Ae.size;
	if (SparseIndex > size)
	{
		cout << "-- Error Exit --: Matrix dimension is " << size << "-----" << endl;
		cout << "-- Error Exit --: SparseIndex is bigger than matrix dimension-----" << endl;
		exit(100);
	}
	double A1 = Ae.getMaxValue();//Get1_Norm(Ae);    ///Here is a problem;

	SparseQMatrix<double> MA;
	vector<vector<pair<int, double>>> sparse_dataM(size);
	vector<vector<pair<int, double>>> sparse_dataMA(size);

#pragma omp parallel for 
	for (int i = 0; i < size; i++)
	{
		auto tempSp = sparI[i];
		double rnorm = 10.0;
		int count = 0;
		int jnew = 0;
		vector<double> mk;
		VectorXd Amk;
		MatrixXd subA;
		VectorXd a0 = Ae.getSparseEigenRowVector(i);
		VectorXd x;
		auto No0Ind = getNZeroIndex(tempSp.data(), tempSp.size());

		while (rnorm > epsilon || tempSp.size() > SparseIndex) {
			if (count != 0) {
				a0 = MproductbDen(Ae, a0);
				auto NewIndex = AddNNZIndex(a0, tempSp);
				if (NewIndex.size() == 0) {
					mk.resize(x.size());
					for (int l = 0; l < x.size(); l++) mk[l] = x(l);
					break;
				}
				else {
					tempSp.insert(tempSp.end(), NewIndex.begin(), NewIndex.end());
					sort(tempSp.begin(), tempSp.end());
				}
				No0Ind = getNZeroIndex(tempSp.data(), tempSp.size());
			}
			subA = FastGetSubMatrix(No0Ind.data(), tempSp.data(), No0Ind.size(), tempSp.size(), Ae);
			VectorXd b = getb(i, No0Ind);
			//x = subA.colPivHouseholderQr().solve(b);
			x = subA.householderQr().solve(b);
			Amk = subA * x;
			VectorXd r = Amk - b;
			rnorm = r.norm();

			if (rnorm < epsilon || tempSp.size()> SparseIndex) {
				int nnz = Getnnz(x);
				double tolk = epsilon / A1 / nnz;
				vector<int> ZeroIndex;
				for (int k = 0; k < x.size(); k++) {
					if (x(k) <= tolk) {
						x(k) = 0;
						ZeroIndex.push_back(k);
					}
					if (x(k) != 0) mk.push_back(x(k));
				}
				if (ZeroIndex.size() != 0) {
					auto size_befor_update = tempSp.size();
					tempSp = GetFreshIndex(tempSp, ZeroIndex);
					auto size_after_update = tempSp.size();
					if (size_befor_update != size_after_update) {
						No0Ind = getNZeroIndex(tempSp.data(), tempSp.size());
						Amk = FastcalculateAMkLeft(tempSp, No0Ind, subA, mk, Ae);
					}
				}
				sparI[i] = tempSp;
				break;
			}
			count++;
		}

		for (int k = 0; k < tempSp.size(); k++) {
			sparse_dataM[i].emplace_back(make_pair(tempSp[k], mk[k]));
		}

		for (int l = 0; l < No0Ind.size(); l++) {
			sparse_dataMA[i].emplace_back(make_pair(No0Ind[l], Amk(l)));
		}
	}
	FastTransfer2SparseQMat(sparse_dataM.data(), size, M);
	FastTransfer2SparseQMat(sparse_dataMA.data(), size, MA);
	write_sparsity();
	sparI.clear();
	return MA;
}

SparseQMatrix<double> JacobiPrecondition(SparseQMatrix<double> &Ae, VectorXd &b) {

	auto diagonal = GetJacobiPrecondition(Ae);

	SparseQMatrix<double> MA = JacobiMatrixMatrixProduct(diagonal.data(), Ae);
	
#pragma omp parallel for
	for (int i = 0; i < b.size(); i++)
		b(i) = diagonal[i] * b(i);

	return MA;
}

SparseQMatrix<double> JacobiPrecondition(SparseQMatrix<double>& Ae, vector<double>& b) {

	auto diagonal = GetJacobiPrecondition(Ae);

	SparseQMatrix<double> MA = JacobiMatrixMatrixProduct(diagonal.data(), Ae);

#pragma omp parallel for
	for (int i = 0; i < b.size(); i++)
		b[i] = diagonal[i] * b[i];

	return MA;
}


void precondition(SparseQMatrix<double>& M, SparseQMatrix<double>& A, vector<double>& b, 
	              int KindLinPrec, int SparseIndex, double DySPepsilon) {
	auto dim = M.size;
	vector<vector<double>> diagD;
	vector<double> d1;
	//vector<double> d2;
	SparseQMatrix<double> Ms;

	switch (KindLinPrec) {
	case NO_PRE:
		A = M;
		break;
	case DIAGNOAL_SCALING:
		DiagonalScalingNew(M, diagD);
		cout << "DiagonalScaling is over" << endl;
		d1 = diagD[0];
		dright = diagD[1];
		A = M;
		for (int i = 0; i < dim; i++) {
			b[i] = b[i] * d1[i];
		}
		break;
	case STATIC_SPAI:
		Ms = StaticSparseApproximateInverse(M, SparseIndex);
		cout << "Static Sparse Approximate Inverse is over" << endl;
		A = MprodcutA(Ms, M, SparseIndex);
		b = MproductbDen(Ms, b);
		break;
	case DYNAMIC_SPAI:
		A = DynamicSparseApproximateInverse(M, Ms, DySPepsilon, SparseIndex);
		cout << "Dynamic Sparse Approximate Inverse is over" << endl;
		b = MproductbDen(Ms, b);
		break;
	case POWER_SAI_LEFT:
		A = PowerSparseApproximateInverseLeft(M, Ms, DySPepsilon, SparseIndex);
		cout << "Power Sparse Approximate Inverse is over" << endl;
		b = MproductbDen(Ms, b);
		break;
	case JACOBI_P:
		A = JacobiPrecondition(M, b);
		break;
	}
}

std::pair< Eigen::MatrixXd, Eigen::VectorXd > DynamicSparseApproximateInverse(Eigen::MatrixXd& Ae, Eigen::VectorXd& b, double epsilon, int SparseIndex, Eigen::MatrixXd& M) {
	//sparI = read_sparsity();
	//No0Index = read_N0index();
	initialSparsity(Ae.rows());
	initialNo0Index(Ae);
	int size = Ae.rows();
	if (SparseIndex > size)
	{
		cout << "-- Error Exit --: Matrix dimension is " << size << "-----" << endl;
		cout << "-- Error Exit --: SparseIndex is bigger than matrix dimension-----" << endl;
		exit(100);
	}
	cout << sparI.size() << endl;
	SparseQMatrix<double> MA;
	MatrixXd sparse_dataM = MatrixXd::Zero(size, size);
	MatrixXd sparse_dataMA = MatrixXd::Zero(size, size);

	for (int i = 0; i < size; i++)
	{
		auto tempSp = sparI[i];
		double rnorm = 10.0;
		int count = 0;
		int jnew = 0;
		VectorXd x;
		VectorXd Amk;
		MatrixXd subA;	
		auto No0Ind = getNZeroIndex(tempSp.data(), tempSp.size());
		while (rnorm > epsilon) {
			if (count != 0) {
				tempSp.push_back(jnew);
				AddNZeroIndex(No0Ind, jnew);
			}
			sort(tempSp.begin(), tempSp.end());

			int size_Sp = tempSp.size();
			int size_NNZ = No0Ind.size();

			subA = FastGetSubMatrix(No0Ind.data(), tempSp.data(), size_NNZ, size_Sp, Ae);

			VectorXd b_i = getb(i, No0Ind);
			x = subA.colPivHouseholderQr().solve(b_i);
			Amk = subA * x;
			VectorXd r = Amk - b_i;
			rnorm = r.norm();
			if (rnorm < epsilon || tempSp.size() >= SparseIndex)
			{
				sparI[i] = tempSp;
				break;
			}
			jnew = getRhoJ(r, rnorm, No0Ind, tempSp, Ae);
			count++;
		}

		for (int k = 0; k < tempSp.size(); k++) {
			auto s = x.size();
			sparse_dataM(i, tempSp[k]) = x(k);
			/*sparse_dataM[i].emplace_back(make_pair(tempSp[k], x(k)));*/
		}
		for (int k = 0; k < No0Ind.size(); k++) {
			sparse_dataMA(i, No0Ind[k]) = Amk(k);
			/*sparse_dataMA[i].emplace_back(make_pair(No0Ind[k], Amk(k)));*/
		}
	}
	b = sparse_dataM * b;
	M = sparse_dataM;
	sparI.clear();
	No0Index.clear();
	return make_pair(sparse_dataMA, b);
}
