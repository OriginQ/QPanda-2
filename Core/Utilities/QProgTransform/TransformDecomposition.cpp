/******************************************************************
Filename     : TransformDecomposition.cpp
Creator      : Menghan Dou��cheng xue
Create time  : 2018-07-04
Description  : Quantum program adaptation metadata instruction set
*******************************************************************/
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/QProgInfo/QGateCompare.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/QProgFlattening.h"

#define iunit qcomplex_t(0,1)
USING_QPANDA
using namespace std;

double transformMatrixToAxis(QStat &QMatrix, axis &Axis)
{

    double dRotateAngle;

	qcomplex_t a0 = (QMatrix[0] + QMatrix[3]) / qcomplex_t(2.0,0);
	qcomplex_t a1 = (QMatrix[1] + QMatrix[2]) / qcomplex_t(2.0, 0);
	qcomplex_t a2 =(QMatrix[1] - QMatrix[2])/ qcomplex_t(2.0, 0);
	qcomplex_t a3 = (QMatrix[0] - QMatrix[3]) / qcomplex_t(2.0, 0);

	double gphase;
	if (abs(a0) > ZeroJudgement)
		gphase = argc(a0);
	else if (abs(a1) > ZeroJudgement)
		gphase = argc(a1) - PI * 1.5;
	else if(abs(a2) > ZeroJudgement)
		gphase = argc(a2) - PI;
	else if (abs(a3) > ZeroJudgement)
		gphase = argc(a3) - PI * 1.5;
    
	qcomplex_t gtemp = qcomplex_t(cos(gphase), sin(gphase));
	dRotateAngle = acos((a0/gtemp).real());

    /*
    if QMatrix is unit matrix,set axis is Z axis,rotate angle is 0
    */
    if (abs(abs(a0) - 1) < ZeroJudgement)
    {
        Axis.nx = 0;
        Axis.ny = 0;
        Axis.nz = 1;
    }
    else
    {

        Axis.nx = -(a1/gtemp).imag()/sin(dRotateAngle);
        Axis.ny = -(a2 / gtemp).real() / sin(dRotateAngle);
        Axis.nz = -(a3 / gtemp).imag() / sin(dRotateAngle);

        double dSum = Axis.nx*Axis.nx + Axis.ny*Axis.ny + Axis.nz*Axis.nz;
        dSum = sqrt(dSum);

        Axis.nx = Axis.nx / dSum;
        Axis.ny = Axis.ny / dSum;
        Axis.nz = Axis.nz / dSum;
    }

    return 2 * dRotateAngle;

}

class CheckMultipleControlQGate : public TraverseByNodeIter
{
public:
	CheckMultipleControlQGate()
		:m_b_exist_multiple_gate(false)
	{}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		if (m_b_exist_multiple_gate)
		{
			return;
		}

		if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
		{
			return;
		}

		QVec control_qubits;
		cur_node->getControlVector(control_qubits);

		if ((control_qubits.size() > 0) || (cir_param.m_control_qubits.size() > 0))
		{
			m_b_exist_multiple_gate = true;
		}
	}

	void execute(std::shared_ptr<AbstractControlFlowNode>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		if (m_b_exist_multiple_gate)
		{
			return;
		}

		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		if (m_b_exist_multiple_gate)
		{
			return;
		}

		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		if (m_b_exist_multiple_gate)
		{
			return;
		}

		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	bool exist_multiple_gate(QProg prog) {
		traverse_qprog(prog);

		return m_b_exist_multiple_gate;
	}

private:
	bool m_b_exist_multiple_gate;
};

void DecomposeUnitarySingleQGateIntoMetadataSingleQGate::rotateAxis(QStat & QMatrix, axis & OriginAxis, axis& NewAxis)
{
    double dTheta;
    double dAlpha;

    if (abs(OriginAxis.nz - 1) < ZeroJudgement)
    {
        dAlpha = 0;
    }
    else
    {
        if (OriginAxis.ny > 0)
        {
            dAlpha = acos((qstate_type)(OriginAxis.nx / sqrt(OriginAxis.nx*OriginAxis.nx + OriginAxis.ny * OriginAxis.ny)));
        }
        else if(OriginAxis.ny < 0)
        {
            dAlpha = -acos((qstate_type)(OriginAxis.nx / sqrt(OriginAxis.nx*OriginAxis.nx + OriginAxis.ny * OriginAxis.ny)));
        }
        else if(OriginAxis.nx == 0 && OriginAxis.ny == 0)
        {
            dAlpha = acos(0);
        }
        else
        {
            dAlpha = acos((qstate_type)(OriginAxis.nx / sqrt(OriginAxis.nx*OriginAxis.nx + OriginAxis.ny * OriginAxis.ny)));
        }
    }

    dTheta = acos(OriginAxis.nz);

    qcomplex_t cTemp1 = QMatrix[0] * (qstate_type)cos(dTheta / 2) +
        QMatrix[1] * (qstate_type)sin(dTheta / 2)*qcomplex_t(cos(dAlpha), sin(dAlpha));
    qcomplex_t cTemp2 = QMatrix[2] * (qstate_type)cos(dTheta / 2) +
        QMatrix[3] * (qstate_type)sin(dTheta / 2)*qcomplex_t(cos(dAlpha), sin(dAlpha));
	double dTheta1 = 0;
	double dAlpha1 = 0;

    if (abs(abs(cTemp1) - 1) < ZeroJudgement)
    {
        dTheta1 = 0;
        dAlpha1 = 0;
    }
    else if (abs(abs(cTemp2) - 1) < ZeroJudgement)
    {
        dTheta1 = PI;
        dAlpha1 = 0;
    }
    else
    {
		double gphase = argc(cTemp1);
		qcomplex_t qtemp = qcomplex_t(cos(gphase), sin(gphase));
		double temp1 = (cTemp1 / qtemp).real();
		double temp2 = argc(cTemp2 / qtemp);
        dTheta1 = 2 * acos((cTemp1/ qtemp).real());
        dAlpha1 = argc(cTemp2) - argc(cTemp1);
    }

    NewAxis.nx = sin(dTheta1)*cos(dAlpha1);
    NewAxis.ny = sin(dTheta1)*sin(dAlpha1);
    NewAxis.nz = cos(dTheta1);

    return;
}
void DecomposeDoubleQGate::matrixMultiplicationOfDoubleQGate(QStat & LeftMatrix, QStat & RightMatrix)
{
    QStat vMatrix(DoubleGateMatrixSize, 0);

    int iMatrixDimension = SingleGateMatrixSize;

    for (size_t i = 0; i < iMatrixDimension; i++)
    {
        for (size_t j = 0; j < iMatrixDimension; j++)
        {
            for (size_t k = 0; k < iMatrixDimension; k++)
            {
                vMatrix[iMatrixDimension * i + j] +=
                    RightMatrix[iMatrixDimension * i + k] * LeftMatrix[iMatrixDimension * k + j];
            }
        }
    }

    LeftMatrix = vMatrix;
}

void DecomposeDoubleQGate::generateMatrixOfTwoLevelSystem(QStat & NewMatrix, QStat & OldMatrix, size_t Row, size_t Column)
{
    for (auto iter = NewMatrix.begin(); iter != NewMatrix.end(); iter++)
    {
        *iter = 0;
    }

    int iMatrixDimension = SingleGateMatrixSize;

    NewMatrix[iMatrixDimension * Row + Row] = OldMatrix[0];
    NewMatrix[iMatrixDimension * Row + Column] = OldMatrix[1];
    NewMatrix[iMatrixDimension * Column + Row] = OldMatrix[2];
    NewMatrix[iMatrixDimension * Column + Column] = OldMatrix[3];

    for (size_t i = 0; i < iMatrixDimension; i++)
    {
        if (i != Row && i != Column)
        {
            NewMatrix[iMatrixDimension * i + i] = 1;
        }
    }
}

void DecomposeDoubleQGate::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
    if (!cur_node)
    {
        QCERR("pnode is null");
        throw invalid_argument("pnode is null");
    }
    
    QuantumGate* qGate;
    qGate = cur_node->getQGate();

    if (cur_node->getTargetQubitNum() == 1)
    {
        return;
    }

	auto & type = TransformQGateType::getInstance();

	QVec ctrl_qubits;
	cur_node->getControlVector(ctrl_qubits);

	for (auto aiter : m_valid_qgate_matrix[1])
	{
		if ((cur_node->getQGate()->getGateType() == type[aiter]) && (ctrl_qubits.size() == 0 ))
		{
			return;
		}
	}

    QVec vQubit;

    if (cur_node->getQuBitVector(vQubit) <= 0)
    {
        QCERR("the num of qubit vector error ");
        throw runtime_error("the num of qubit vector error");
    }

    QStat vMatrix;
    qGate->getMatrix(vMatrix);

    QStat vMatrix1(DoubleGateMatrixSize, 0);
    QStat vMatrix2(SingleGateMatrixSize, 0);
    QStat vMatrix3(SingleGateMatrixSize, 0);

    double dSum;

    QCircuit qCircuit = CreateEmptyCircuit();

    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = i + 1; j < SingleGateMatrixSize; j++)
        {
            if (abs(vMatrix[SingleGateMatrixSize * j + i]) > ZeroJudgement)
            {
                dSum = sqrt(abs(vMatrix[5 * i])*abs(vMatrix[5 * i]) +
                    abs(vMatrix[SingleGateMatrixSize * j + i])*abs(vMatrix[SingleGateMatrixSize * j + i]));

                vMatrix2[0] = qcomplex_t(vMatrix[5 * i].real(), -vMatrix[5 * i].imag()) / (qstate_type)dSum;
                vMatrix2[1] = qcomplex_t(vMatrix[SingleGateMatrixSize * j + i].real(),
                    -vMatrix[SingleGateMatrixSize * j + i].imag()) / (qstate_type)dSum;
                vMatrix2[2] = vMatrix[SingleGateMatrixSize * j + i] / (qstate_type)dSum;
                vMatrix2[3] = -vMatrix[5 * i] / (qstate_type)dSum;
                vMatrix3[0] = vMatrix2[3];
                vMatrix3[1] = vMatrix2[2];
                vMatrix3[2] = vMatrix2[1];
                vMatrix3[3] = vMatrix2[0];

                generateMatrixOfTwoLevelSystem(vMatrix1, vMatrix2, i, j);

                size_t stMatrixIndex = SingleGateMatrixSize * j + i;

                switch (stMatrixIndex)
                {
                case 4:
                    qCircuit << X(vQubit[0]) << CU(vMatrix2, vQubit[0], vQubit[1])
                        << X(vQubit[0]);
                    break;
                case 8:
                    qCircuit << X(vQubit[1]) << CU(vMatrix2, vQubit[1], vQubit[0])
                        << X(vQubit[1]);
                    break;
                case 12:
                    qCircuit << X(vQubit[0]) << CNOT(vQubit[0], vQubit[1])
                        << X(vQubit[0]) << CU(vMatrix2, vQubit[1], vQubit[0])
                        << X(vQubit[0]) << CNOT(vQubit[0], vQubit[1])
                        << X(vQubit[0]);
                    break;
                case 9:
                    qCircuit << CNOT(vQubit[1], vQubit[0])
                        << CU(vMatrix3, vQubit[0], vQubit[1])
                        << CNOT(vQubit[1], vQubit[0]);
                    break;
                case 13:
                    qCircuit << CU(vMatrix2, vQubit[1], vQubit[0]);
                    break;
                case 14:
                    qCircuit << CU(vMatrix2, vQubit[0], vQubit[1]);
                    break;
                }
                matrixMultiplicationOfDoubleQGate(vMatrix, vMatrix1);
            }
        }
    }

    vMatrix2[0] = 1;
    vMatrix2[1] = 0;
    vMatrix2[2] = 0;
    vMatrix2[3] = qcomplex_t(vMatrix[15].real(), -vMatrix[15].imag());
    qCircuit << CU(vMatrix2, vQubit[0], vQubit[1]);
    auto qCircuitDagger = qCircuit.dagger();
    auto count3 = getQGateNumber(qCircuit);
    if(cur_node->isDagger())
    {
        qCircuitDagger.setDagger(qCircuitDagger.isDagger()^true);
    }

	if (ctrl_qubits.size() > 0)
	{
		qCircuitDagger.setControl(ctrl_qubits);
	}
	
    replace_qcircuit(cur_node.get(), qCircuitDagger, parent_node.get());
}

void DecomposeMultipleControlQGate::QGateExponentArithmetic(AbstractQGateNode * pNode, double Exponent, QStat & QMatrix)
{
    QuantumGate *gat = pNode->getQGate();

    QStat pNodeMatrix;
    gat->getMatrix(pNodeMatrix);

    axis axi;

    double dTheta = transformMatrixToAxis(pNodeMatrix, axi);
    transformAxisToMatrix(axi, dTheta*Exponent, QMatrix);

    double dAlpha;

    if (abs(pNodeMatrix[0]) > ZeroJudgement)
    {
        dAlpha = argc(pNodeMatrix[0] / (QMatrix[0] * QMatrix[0] + QMatrix[1] * QMatrix[2]));
    }
    else
    {
        dAlpha = argc(pNodeMatrix[1] / (QMatrix[0] * QMatrix[1] + QMatrix[1] * QMatrix[3]));
    }

    for (auto i = 0; i < SingleGateMatrixSize; i++)
    {
        QMatrix[i] *= qcomplex_t(cos(dAlpha*Exponent), sin(dAlpha*Exponent));
    }

}

void DecomposeMultipleControlQGate::transformAxisToMatrix(axis &Axis, double Angle, QStat &QMatrix)
{
    QMatrix.resize(SingleGateMatrixSize);

    QMatrix[0] = qcomplex_t(cos(Angle / 2), -sin(Angle / 2)*Axis.nz);
    QMatrix[1] = qcomplex_t(-sin(Angle / 2)*Axis.ny, -sin(Angle / 2)*Axis.nx);
    QMatrix[2] = qcomplex_t(sin(Angle / 2)*Axis.ny, -sin(Angle / 2)*Axis.nx);
    QMatrix[3] = qcomplex_t(cos(Angle / 2), sin(Angle / 2)*Axis.nz);
}

QCircuit DecomposeMultipleControlQGate::
firstStepOfMultipleControlQGateDecomposition(AbstractQGateNode* pNode, Qubit * AncillaQubit)
{
    QVec vTargetQubit;

    if (pNode->getQuBitVector(vTargetQubit) <= 0)
    {
        QCERR("the num of qubit vector error ");
        throw runtime_error("the num of qubit vector error");
    }

    QVec vControlQubit;

    if (pNode->getControlVector(vControlQubit) <= 0)
    {
        QCERR("the num of control qubit vector error ");
        throw runtime_error("the num of control qubit vector error");
    }

    QuantumGate* qGate = pNode->getQGate();
    auto qCircuit = CreateEmptyCircuit();

    if (vControlQubit.size() > 3 && vTargetQubit.size() == 1)
    {
        vector<Qubit*> vUpQubits;
        vector<Qubit*> vDownQubits;
        vector<Qubit*> vTempQubits;

        if (vControlQubit.size() % 2 == 0)                /* number of control qubit is even */
        {
            vUpQubits.insert(vUpQubits.begin(), vControlQubit.begin(),
                vControlQubit.begin() + vControlQubit.size() / 2 + 1);
            vDownQubits.insert(vDownQubits.begin(), vControlQubit.begin() + vControlQubit.size() / 2 + 1, vControlQubit.end());

            auto qNode1 = X(AncillaQubit);
            qNode1.setControl(vUpQubits);

            auto qCircuit1 = secondStepOfMultipleControlQGateDecomposition(&qNode1, vDownQubits);
            auto qCircuit3(qCircuit1);

            vDownQubits.push_back(AncillaQubit);
            
 
            auto temp_gate = copy_qgate(qGate,{vTargetQubit[0]});
            temp_gate.setControl(vDownQubits);

            if (vDownQubits.size() >= 3)
            {
                vTempQubits.insert(vTempQubits.begin(), vUpQubits.begin(),
                    vUpQubits.begin() + vDownQubits.size() - 2);
            }

            auto qCircuit2 = secondStepOfMultipleControlQGateDecomposition(&temp_gate, vTempQubits);

            auto qCircuit4(qCircuit2);

            vDownQubits.pop_back();

            qCircuit << qCircuit1 << qCircuit2 << qCircuit3 << qCircuit4;
        }
        else                                 /* number of control qubit is odd */
        {
            vUpQubits.insert(vUpQubits.begin(), vControlQubit.begin(),
                vControlQubit.begin() + (vControlQubit.size() + 3) / 2);
            vDownQubits.insert(vDownQubits.begin(), vControlQubit.begin() + (vControlQubit.size() + 3) / 2, vControlQubit.end());
            vDownQubits.push_back(vTargetQubit[0]);

            auto qNode3 = X(AncillaQubit);
            qNode3.setControl(vUpQubits);

            auto qCircuit1 = secondStepOfMultipleControlQGateDecomposition(&qNode3, vDownQubits);
            auto qCircuit3(qCircuit1);

            vDownQubits.pop_back();
			
            vDownQubits.push_back(AncillaQubit);

            if (vDownQubits.size() >= 3)
            {
                vTempQubits.insert(vTempQubits.begin(), vUpQubits.begin(),
                    vUpQubits.begin() + vDownQubits.size() - 2);
            }

            auto qNode4 = copy_qgate(qGate,{vTargetQubit[0]});
            qNode4.setControl(vDownQubits);
            auto qCircuit2 = secondStepOfMultipleControlQGateDecomposition(&qNode4, vTempQubits);
            auto qCircuit4(qCircuit2);
            vDownQubits.pop_back();

            qCircuit << qCircuit1 << qCircuit2 << qCircuit3 << qCircuit4;
        }
    }
    else if (vControlQubit.size() == 3)
    {
        qCircuit << secondStepOfMultipleControlQGateDecomposition(pNode, { AncillaQubit });
    }
    else if (vControlQubit.size() == 2)
    {
        qCircuit << decomposeTwoControlSingleQGate(pNode);
    }

    return qCircuit;
}

QCircuit DecomposeMultipleControlQGate::
secondStepOfMultipleControlQGateDecomposition(AbstractQGateNode *pNode, vector<Qubit*> AncillaQubitVector)
{
    QVec vTargetQubit;

    if (pNode->getQuBitVector(vTargetQubit) <= 0)
    {
        QCERR("the num of qubit vector error ");
        throw runtime_error("the num of qubit vector error");
    }

    QVec vControlQubit;

    if (pNode->getControlVector(vControlQubit) <= 0)
    {
        QCERR("the num of control qubit vector error ");
        throw runtime_error("the num of control qubit vector error");
    }

    auto qCircuit = CreateEmptyCircuit();
    /*
    * n control qubits,n-2 ancilla qubits,1 target qubit
    */

    vector<Qubit*> vqtemp(2);

    QStat qMatrix;

    QuantumGate* qgate = pNode->getQGate();
    auto new_gate =  copy_qgate(qgate,{vTargetQubit[0]});

    if (vControlQubit.size() >2 && (vControlQubit.size() - AncillaQubitVector.size() == 2) && vTargetQubit.size() == 1)
    {
        vqtemp[0] = vControlQubit[vControlQubit.size() - 1];
        vqtemp[1] = AncillaQubitVector[AncillaQubitVector.size() - 1];
        new_gate.setControl(vqtemp);
        qCircuit << decomposeTwoControlSingleQGate(&new_gate);
        qCircuit << tempStepOfMultipleControlQGateDecomposition(vControlQubit, AncillaQubitVector);
        qCircuit << decomposeTwoControlSingleQGate(&new_gate);
        qCircuit << tempStepOfMultipleControlQGateDecomposition(vControlQubit, AncillaQubitVector);
    }
    else if (vControlQubit.size() == 2)
    {
        vqtemp[0] = vControlQubit[0];
        vqtemp[1] = vControlQubit[1];

        new_gate.setControl(vqtemp);
        qCircuit << decomposeTwoControlSingleQGate(&new_gate);
    }
    else
    {
        QCERR("unknow error ");
        throw runtime_error("unknow error");
    }
    return qCircuit;
}

QCircuit DecomposeMultipleControlQGate::decomposeToffoliQGate(Qubit * TargetQubit, vector<Qubit*> ControlQubits)
{
    auto qCircuit = CreateEmptyCircuit();

    QStat vMatrix;

    auto tempQGate = X(TargetQubit);

    QGateExponentArithmetic(&tempQGate, 0.5, vMatrix);

    qCircuit << CU(vMatrix, ControlQubits[1], TargetQubit) << CNOT(ControlQubits[0], ControlQubits[1]);

    auto qGate = CU(vMatrix, ControlQubits[1], TargetQubit);
    qGate.setDagger(1);

    qCircuit << qGate << CNOT(ControlQubits[0], ControlQubits[1])
        << CU(vMatrix, ControlQubits[0], TargetQubit);

    return qCircuit;
}
QCircuit DecomposeMultipleControlQGate::decomposeTwoControlSingleQGate(AbstractQGateNode * pNode)
{
    QVec vTargetQubit;

    auto qCircuit = CreateEmptyCircuit();

    if (pNode->getQuBitVector(vTargetQubit) <= 0)
    {
        QCERR("the num of qubit vector error ");
        throw runtime_error("the num of qubit vector error");
    }

    QVec vControlQubit;

    if (pNode->getControlVector(vControlQubit) <= 0)
    {
        QCERR("the num of control qubit vector error ");
        throw runtime_error("the num of control qubit vector error");
    }

    if (vTargetQubit.size() != 1 || vControlQubit.size() != 2)
    {
        QCERR("the size of qubit vector error ");
        throw runtime_error("the size of qubit vector error ");
    }

    QStat vMatrix;

    QGateExponentArithmetic(pNode, 0.5, vMatrix);

    auto qGate = CU(vMatrix, vControlQubit[1], vTargetQubit[0]);
    qGate.setDagger(1);

    qCircuit << CU(vMatrix, vControlQubit[1], vTargetQubit[0]) << CNOT(vControlQubit[0], vControlQubit[1])
        << qGate << CNOT(vControlQubit[0], vControlQubit[1])
        << CU(vMatrix, vControlQubit[0], vTargetQubit[0]);

    return qCircuit;
}
QCircuit DecomposeMultipleControlQGate::tempStepOfMultipleControlQGateDecomposition(vector<Qubit*> ControlQubits, vector<Qubit*> AncillaQubits)
{
    auto qCircuit = CreateEmptyCircuit();

    vector<Qubit*> vTempQubit(2);

    if (ControlQubits.size() == 3)
    {
        vTempQubit[0] = ControlQubits[0];
        vTempQubit[1] = ControlQubits[1];
        qCircuit << decomposeToffoliQGate(AncillaQubits[0], vTempQubit);
    }
    else if (ControlQubits.size()>3)
    {
        for (auto i = ControlQubits.size() - 2; i >= 2; i--)
        {
            vTempQubit[0] = ControlQubits[i];
            vTempQubit[1] = AncillaQubits[i - 2];
            qCircuit << decomposeToffoliQGate(AncillaQubits[i - 1], vTempQubit);
        }

        vTempQubit[0] = ControlQubits[0];
        vTempQubit[1] = ControlQubits[1];

        qCircuit << decomposeToffoliQGate(AncillaQubits[0], vTempQubit);

        for (auto i = 2; i <= ControlQubits.size() - 2; i++)
        {
            vTempQubit[0] = ControlQubits[i];
            vTempQubit[1] = AncillaQubits[i - 2];

            qCircuit << decomposeToffoliQGate(AncillaQubits[i - 1], vTempQubit);
        }
    }
    else
    {
        QCERR("unknow error");
        throw runtime_error("unknow error ");
    }
    return qCircuit;
}

QCircuit DecomposeMultipleControlQGate::decompose_multiple_control_qgate(AbstractQGateNode * cur_node)
{
	QVec vTargetQubit;
	QVec vControlQubit;
	cur_node->getQuBitVector(vTargetQubit);
	cur_node->getControlVector(vControlQubit);


	QuantumGate* qgate = cur_node->getQGate();
	QStat qMatrix;
	qgate->getMatrix(qMatrix);

	QStat vMatrix;
	QStat matrixdagger;

	QGateExponentArithmetic(cur_node, 0.5, vMatrix);

	auto qCircuit = CreateEmptyCircuit();


	auto qGate0 = CU(vMatrix, vControlQubit[vControlQubit.size() - 1], vTargetQubit[0]);
	qGate0.setDagger(1);

	if (cur_node->getControlQubitNum() == 1)
	{
		qCircuit << CU(qMatrix, vControlQubit[0], vTargetQubit[0]);
	}
	else if (cur_node->getControlQubitNum() == 2)
	{
		//pNode->setControl(vControlQubit);
		qCircuit << decomposeTwoControlSingleQGate(cur_node);
	}
	else if (cur_node->getControlQubitNum() == 3)
	{
		vector<Qubit*> vTempQubit;

		vTempQubit.push_back(vControlQubit[0]);
		vTempQubit.push_back(vControlQubit[1]);
		auto new_gate = U4(vMatrix, vTargetQubit[0]);
		new_gate.setControl(vTempQubit);

		qCircuit << CU(vMatrix, vControlQubit[2], vTargetQubit[0])
			<< decomposeToffoliQGate(vControlQubit[2], { vControlQubit[0],vControlQubit[1] })
			<< qGate0 << decomposeToffoliQGate(vControlQubit[2], { vControlQubit[0],vControlQubit[1] })
			<< decomposeTwoControlSingleQGate(&new_gate);
	}
	else if (cur_node->getControlQubitNum() > 3)
	{
		Qubit* temp = vControlQubit[vControlQubit.size() - 1];
		string class_name = "U4";
		auto new_gate = QGATE_SPACE::create_quantum_gate(class_name,
			qMatrix);
		auto temp_u4 = dynamic_cast<QGATE_SPACE::U4 *>(new_gate);
		double alpha = temp_u4->getAlpha();
		double beta = temp_u4->getBeta();
		double delta = temp_u4->getDelta();
		double gamma = temp_u4->getGamma();

		QCircuit A;
		A <<RY(vTargetQubit[0],gamma/2).control(temp) << RZ(vTargetQubit[0],beta).control(temp);
		QCircuit B;
		B << RZ(vTargetQubit[0], -(beta + delta) / 2).control(temp) << RY(vTargetQubit[0], -gamma / 2).control(temp);
		auto C = RZ(vTargetQubit[0], (delta - beta) / 2).control(temp);

		auto qGate1 = X(vTargetQubit[0]);
		vControlQubit.pop_back();
		qGate1.setControl(vControlQubit);
		auto u1 = U1(temp, alpha).control(vControlQubit);
		auto qCircuit1 = firstStepOfMultipleControlQGateDecomposition(&qGate1,temp);
		qCircuit << C << qCircuit1 << B << qCircuit1 << A << decompose_multiple_control_qgate(&u1);
	}

	if (cur_node->isDagger())
	{
		qCircuit.setDagger(qCircuit.isDagger() ^ true);
	}

	return qCircuit;
}

void DecomposeMultipleControlQGate::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
	if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
	{
		return;
	}
	
    QVec vTargetQubit;

    if (cur_node->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QVec vControlQubit;

    if (CIRCUIT_NODE == parent_node->getNodeType())
    {
        AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(parent_node.get());
        pQcir->getControlVector(vControlQubit);
    }
	cur_node->setControl(vControlQubit);
    if (cur_node->getControlQubitNum()<= 0)
    {
        return;
    }

	vControlQubit.clear();
	cur_node->getControlVector(vControlQubit);
	auto qCircuit = decompose_multiple_control_qgate(cur_node.get());
    replace_qcircuit(cur_node.get(), qCircuit, parent_node.get());
}

void DecomposeMultipleControlQGate::execute(std::shared_ptr<AbstractQuantumCircuit>  cur_node, std::shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node)
    {
        QCERR("node is nullptr");
        throw invalid_argument("node is nullptr");
    }

    if (nullptr == parent_node)
    {
        QCERR("parent node is nullptr");
        throw invalid_argument("parent node is nullptr");
    }

    if (CIRCUIT_NODE == parent_node->getNodeType())
    {
        auto parent_qcircuit = dynamic_pointer_cast<AbstractQuantumCircuit>(parent_node);
        QVec vControlQubit;
        parent_qcircuit->getControlVector(vControlQubit);
        cur_node->setControl(vControlQubit);
    }

    Traversal::traversal(cur_node,false,*this);
}

/******************************************************************
Name        : decomposeControlUnitarySingleQGate
Description : CU conversion to single gate
argin       : pNode              Target gate pointer
              pParentNode        Target gate's parent node
traversalAlgorithm traversalAlgorithm pointer
argout      :
Return      :
******************************************************************/
void DecomposeControlUnitarySingleQGate::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
	if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
	{
		return;
	}

	auto & type = TransformQGateType::getInstance();

	for (auto aiter : m_valid_qgate_matrix[1])
	{
		if (cur_node->getQGate()->getGateType() == type[aiter])
		{
			return;
		}
	}

    if (cur_node->getTargetQubitNum() == 1)
    {
        return;
    }

    auto target_qubit = cur_node->popBackQuBit();
    auto control_qubit = cur_node->popBackQuBit();

    cur_node->PushBackQuBit(target_qubit);

    vector<Qubit *> vControlQubit = { control_qubit };

    cur_node->setControl(vControlQubit);

    auto qgate = cur_node->getQGate();

    if (nullptr == qgate)
    {
        QCERR("qgate is null");
        throw runtime_error("qgate is null");
    }

    QVec qubitVector;

    if (cur_node->getQuBitVector(qubitVector) <= 0)
    {
        QCERR("the size of qubit vector is error");
        throw runtime_error("the size of qubit vector is error");
    }

    string class_name = "U4";
	QStat src_matrix;
	qgate->getMatrix(src_matrix);
	QStat target_matrix(4);
	target_matrix[0] = src_matrix[10];
	target_matrix[1] = src_matrix[11];
	target_matrix[2] = src_matrix[14];
	target_matrix[3] = src_matrix[15];

    auto new_gate =QGATE_SPACE::create_quantum_gate(class_name, 
		target_matrix);

    cur_node->setQGate(new_gate);
}

/******************************************************************
Name        : decomposeControlSingleQGateIntoMetadataDoubleQGate
Description : Control single gate conversion to metadata double
quantum gate
argin       : pNode              Target gate pointer
              pParentNode        Target gate's parent node
traversalAlgorithm traversalAlgorithm pointer
argout      :
Return      :
******************************************************************/
void DecomposeControlSingleQGateIntoMetadataDoubleQGate ::
     execute(std::shared_ptr<AbstractQGateNode>  cur_node,
         std::shared_ptr<QNode> parent_node)
{
	if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
	{
		return;
	}

	if (m_valid_qgate_matrix[1].size() == 0)
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: no valid double gate in meatdata.");
	}

    string sGateName = m_valid_qgate_matrix[1][0];

    if (sGateName.size() <= 0)
    {
		QCERR_AND_THROW_ERRSTR(runtime_error, "the size of sGateName is error");
    }

    QVec vTargetQubit;
    if (cur_node->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QVec vControlQubit;
    if (cur_node->getControlVector(vControlQubit) != 1)
    {
        return;
    }



    if (CIRCUIT_NODE == parent_node->getNodeType())
    {
        AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(parent_node.get());
        pQcir->getControlVector(vControlQubit);
    }
    QuantumGate* qgate = cur_node->getQGate();

    auto angle = dynamic_cast<AbstractAngleParameter *>(qgate);

    double dAlpha = angle->getAlpha();
    double dBeta = angle->getBeta();
    double dDelta = angle->getDelta();
    double dGamma = angle->getGamma();

    auto qCircuit = CreateEmptyCircuit();

    QStat QMatrix(SingleGateMatrixSize, 0);

    QMatrix[0] = 1;
    QMatrix[3] = qcomplex_t(cos(dAlpha), sin(dAlpha));

    auto gat = U4(QMatrix, vControlQubit[0]);

    auto qSwap = CreateEmptyCircuit();
    auto qSwapDagger = CreateEmptyCircuit();

    if (sGateName == "CNOT")
    {
        qCircuit << qSwap
            << RZ(vTargetQubit[0], (dDelta - dBeta) / 2) << CNOT(vControlQubit[0], vTargetQubit[0])
            << RZ(vTargetQubit[0], -(dDelta + dBeta) / 2) << RY(vTargetQubit[0], -dGamma / 2)
            << CNOT(vControlQubit[0], vTargetQubit[0]) << RY(vTargetQubit[0], dGamma / 2)
            << RZ(vTargetQubit[0], dBeta) << gat
            << qSwapDagger;
    }
    else if (sGateName == "CZ")
    {
        qCircuit << qSwap
            << RZ(vTargetQubit[0], (dDelta - dBeta) / 2)
            << H(vTargetQubit[0]) << CZ(vControlQubit[0], vTargetQubit[0]) << H(vTargetQubit[0])
            << RZ(vTargetQubit[0], -(dDelta + dBeta) / 2) << RY(vTargetQubit[0], -dGamma / 2)
            << H(vTargetQubit[0]) << CZ(vControlQubit[0], vTargetQubit[0]) << H(vTargetQubit[0])
            << RY(vTargetQubit[0], dGamma / 2)
            << RZ(vTargetQubit[0], dBeta) << gat
            << qSwapDagger;
    }
    else if (sGateName == "ISWAP")
    {
        auto qGate1 = iSWAP(vControlQubit[0], vTargetQubit[0]);
        qGate1.setDagger(1);

        qCircuit << qSwap
            << RZ(vTargetQubit[0], (dDelta - dBeta) / 2)
            << RZ(vControlQubit[0], -PI / 2) << RX(vTargetQubit[0], PI / 2) << RZ(vTargetQubit[0], PI / 2)
            << qGate1 << RX(vControlQubit[0], PI / 2)
            << qGate1 << RZ(vTargetQubit[0], PI / 2)             //CNOT
            << RZ(vTargetQubit[0], -(dDelta + dBeta) / 2) << RY(vTargetQubit[0], -dGamma / 2)
            << RZ(vControlQubit[0], -PI / 2) << RX(vTargetQubit[0], PI / 2) << RZ(vTargetQubit[0], PI / 2)
            << qGate1 << RX(vControlQubit[0], PI / 2)
            << qGate1 << RZ(vTargetQubit[0], PI / 2)             //CNOT
            << RY(vTargetQubit[0], dGamma / 2)
            << RZ(vTargetQubit[0], dBeta) << gat
            << qSwapDagger;
    }
    else
    {
        QCERR("unknow error");
        throw runtime_error("unknow error");
    }

    if(cur_node->isDagger())
    {
        qCircuit.setDagger(qCircuit.isDagger()^true);
    }
    replace_qcircuit(cur_node.get(), qCircuit, parent_node.get());
}


QCircuit DecomposeControlSingleQGateIntoMetadataDoubleQGate::
         swapQGate(vector<int> shortest_way, string metadata_qgate)
{
    auto qcircuit = CreateEmptyCircuit();

    Qubit *qTemp1 = nullptr;
    Qubit *qTemp2 = nullptr;

    if (metadata_qgate == "CNOT")
    {
        for (auto iter = shortest_way.begin(); iter != shortest_way.end() - 2; iter++)
        {
            if (m_quantum_machine->allocateQubitThroughVirAddress(*iter) != nullptr &&
                m_quantum_machine->allocateQubitThroughVirAddress(*(iter + 1)) != nullptr)
            {
                qTemp1 = m_quantum_machine->allocateQubitThroughVirAddress(*iter);
                qTemp2 = m_quantum_machine->allocateQubitThroughVirAddress(*(iter + 1));

            }
            else
            {
                QCERR("Unknown internal error");
                throw runtime_error("Unknown internal error");
            }
            qcircuit << CNOT(qTemp1, qTemp2) << CNOT(qTemp2, qTemp1) << CNOT(qTemp1, qTemp2);
        }
    }
    else if (metadata_qgate == "CZ")
    {
        for (auto iter = shortest_way.begin(); iter != shortest_way.end() - 2; iter++)
        {
            if (m_quantum_machine->allocateQubitThroughVirAddress(*iter) != nullptr &&
                m_quantum_machine->allocateQubitThroughVirAddress(*(iter + 1)) != nullptr)
            {
                qTemp1 = m_quantum_machine->allocateQubitThroughVirAddress(*iter);
                qTemp2 = m_quantum_machine->allocateQubitThroughVirAddress(*(iter + 1));

            }
            else
            {
                QCERR("Unknown internal error");
                throw runtime_error("Unknown internal error");
            }
            qcircuit << H(qTemp1) << CZ(qTemp1, qTemp2) << H(qTemp1)
                << H(qTemp2) << CZ(qTemp2, qTemp1) << H(qTemp2)
                << H(qTemp1) << CZ(qTemp1, qTemp2) << H(qTemp1);
        }
    }
    else if (metadata_qgate == "ISWAP")
    {
        for (auto iter = shortest_way.begin(); iter != shortest_way.end() - 2; iter++)
        {
            if (m_quantum_machine->allocateQubitThroughVirAddress(*iter) != nullptr &&
                m_quantum_machine->allocateQubitThroughVirAddress(*(iter + 1)) != nullptr)
            {
                qTemp1 = m_quantum_machine->allocateQubitThroughVirAddress(*iter);
                qTemp2 = m_quantum_machine->allocateQubitThroughVirAddress(*(iter + 1));

            }
            auto qgate = iSWAP(qTemp1, qTemp2);
            qgate.setDagger(true);
            qcircuit << RZ(qTemp2, PI / 2) << qgate << RX(qTemp1, -PI / 2) << qgate << RZ(qTemp1, -PI / 2)
                << RZ(qTemp2, PI / 2) << RX(qTemp2, PI / 2)
                << RZ(qTemp1, PI / 2) << qgate << RX(qTemp2, -PI / 2) << qgate << RZ(qTemp2, -PI / 2)
                << RZ(qTemp1, PI / 2) << RX(qTemp1, PI / 2)
                << RZ(qTemp2, PI / 2) << qgate << RX(qTemp1, -PI / 2) << qgate << RZ(qTemp1, -PI / 2)
                << RZ(qTemp2, PI / 2) << RX(qTemp2, PI / 2);
        }
    }
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return qcircuit;
}

DecomposeUnitarySingleQGateIntoMetadataSingleQGate::
DecomposeUnitarySingleQGateIntoMetadataSingleQGate(
    vector<vector<string>> qgate_matrix,
    vector<vector<string>> &valid_qgate_matrix)
{
    m_qgate_matrix = qgate_matrix;
    m_valid_qgate_matrix = valid_qgate_matrix;
    base.n2 = { 0,0,0 };
    /*
    * Initialize member variable
    */
    if (valid_qgate_matrix[0][0] == "RX")
    {
        base.n1 = { 1,0,0 };
        if (valid_qgate_matrix[0][1] == "RY")
        {
            base.n2 = { 0,1,0 };
        }
        else if (valid_qgate_matrix[0][1] == "RZ" || valid_qgate_matrix[0][1] == "U1")
        {
            base.n2 = { 0,0,1 };
        }
        else if (valid_qgate_matrix[0][1] == "H")
        {
            QStat QMatrix = { 1 / SQRT2,1 / SQRT2 ,1 / SQRT2 ,-1 / SQRT2 };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "Y1")
        {
            QStat QMatrix = { (qstate_type)cos(PI / 4),-(qstate_type)sin(PI / 4),(qstate_type)sin(PI / 4),(qstate_type)cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "Z1")
        {
            QStat QMatrix = { qcomplex_t(cos(PI / 4),-sin(PI / 4)),0,0,qcomplex_t(cos(PI / 4),+sin(PI / 4)) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "S")
        {
            QStat QMatrix = { 1,0,0,iunit };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "T")
        {
            QStat QMatrix = { 1,0,0,1 / SQRT2 + iunit / SQRT2 };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else
        {
            QCERR("unknow error");
            throw runtime_error("unknow error");
        }
    }
    else if (valid_qgate_matrix[0][0] == "RY")
    {
        base.n1 = { 0,1,0 };
        if (valid_qgate_matrix[0][1] == "RX")
        {
            base.n2 = { 1,0,0 };
        }
        else if (valid_qgate_matrix[0][1] == "RZ" || valid_qgate_matrix[0][1] == "U1")
        {
            base.n2 = { 0,0,1 };
        }
        else if (valid_qgate_matrix[0][1] == "X1")
        {
            QStat QMatrix = { (qstate_type)cos(PI / 4),-iunit *(qstate_type) sin(PI / 4),-iunit * (qstate_type)sin(PI / 4) ,(qstate_type)cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "Z1")
        {
            QStat QMatrix = { qcomplex_t(cos(PI / 4),-sin(PI / 4)),0,0,qcomplex_t(cos(PI / 4),sin(PI / 4)) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "S")
        {
            QStat QMatrix = { 1,0,0,iunit };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "T")
        {
            QStat QMatrix = { 1,0,0,1 / SQRT2 + iunit / SQRT2 };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else
        {
            QCERR("unknow error");
            throw runtime_error("unknow error");
        }
    }
    else if (valid_qgate_matrix[0][0] == "RZ" || valid_qgate_matrix[0][0] == "U1")
    {
        base.n1 = { 0,0,1 };
        if (valid_qgate_matrix[0][1] == "RX")
        {
            base.n2 = { 1,0,0 };
        }
        else if (valid_qgate_matrix[0][1] == "RY")
        {
            base.n2 = { 0,1,0 };
        }
        else if (valid_qgate_matrix[0][1] == "H")
        {
            QStat QMatrix = { 1 / SQRT2,1 / SQRT2 ,1 / SQRT2 ,-1 / SQRT2 };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "Y1")
        {
            QStat QMatrix = { (qstate_type)cos(PI / 4),-(qstate_type)sin(PI / 4),(qstate_type)sin(PI / 4),(qstate_type)cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "X1")
        {
            QStat QMatrix = { (qstate_type)cos(PI / 4),-iunit * (qstate_type)sin(PI / 4),-iunit * (qstate_type)sin(PI / 4) ,(qstate_type)cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "S")
        {
            QStat QMatrix = { 1,0,0,iunit };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else
        {
            QCERR("unknow error");
            throw runtime_error("unknow error");
        }
    }
	else if ((valid_qgate_matrix[0][0] == "U3") || (valid_qgate_matrix[0][0] == "U4"))
	{
	    return;
	}
    else
    {
        QCERR("unknow error");
        throw runtime_error("unknow error");
    }
}

void DecomposeUnitarySingleQGateIntoMetadataSingleQGate::
     getDecompositionAngle(QStat & qmatrix, vector<double> & vdAngle)
{
    double dTheta;
    double dAlpha;

    if (abs(abs(base.n1.nz) - 1) < ZeroJudgement)
    {
        dAlpha = 0;
    }
    else
    {
        if (base.n1.ny > 0)
        {
            dAlpha = acos((qstate_type)(base.n1.nx / sqrt(base.n1.nx*base.n1.nx + base.n1.ny * base.n1.ny)));
        }
        else if(base.n1.ny < 0)
        {
            dAlpha = -acos((qstate_type)(base.n1.nx / sqrt(base.n1.nx*base.n1.nx + base.n1.ny * base.n1.ny)));
        }
        else if(base.n1.nx == 0 && base.n1.ny == 0)
        {
            dAlpha = acos(0);
        }
        else
        {
            dAlpha = acos((qstate_type)(base.n1.nx / sqrt(base.n1.nx*base.n1.nx + base.n1.ny * base.n1.ny)));
        }
    }

    dTheta = acos(base.n1.nz);


    QStat RY_theta ={0,0,0,0};

    RY_theta[0] = cos(-dTheta / 2);
    RY_theta[1] = -sin(-dTheta / 2);
    RY_theta[2] = sin(-dTheta / 2);
    RY_theta[3] = cos(-dTheta / 2);

    QStat RZ_dalpha={0,0,0,0};
    RZ_dalpha[0].real(cos(-dAlpha / 2));
    RZ_dalpha[0].imag(-1 * sin(-dAlpha / 2));
    RZ_dalpha[3].real(cos(-dAlpha / 2));
    RZ_dalpha[3].imag(1 * sin(-dAlpha / 2));

    auto UnitaryMatrix = RY_theta*RZ_dalpha;
    axis TargetAxis;

    double dBeta = transformMatrixToAxis(qmatrix, TargetAxis);

	//dBeta, TargetAxis right
    double dBeta1;
    double dBeta2;
    double dBeta3;

    axis NewBaseAxis;
    axis NewTargetAxis;

    rotateAxis(UnitaryMatrix, base.n2, NewBaseAxis);
    rotateAxis(UnitaryMatrix, TargetAxis, NewTargetAxis);
	//NewTargetAxis right

    QStat NewMatrix(SingleGateMatrixSize);

    NewMatrix[0] = qcomplex_t(cos(dBeta / 2), -sin(dBeta / 2)*NewTargetAxis.nz);
    NewMatrix[1] = qcomplex_t(-sin(dBeta / 2)*NewTargetAxis.ny, -sin(dBeta / 2)*NewTargetAxis.nx);
    NewMatrix[2] = qcomplex_t(sin(dBeta / 2)*NewTargetAxis.ny, -sin(dBeta / 2)*NewTargetAxis.nx);
    NewMatrix[3] = qcomplex_t(cos(dBeta / 2), sin(dBeta / 2)*NewTargetAxis.nz);


    qcomplex_t cTemp = NewMatrix[0] * NewMatrix[3];

    double dTemp = (1 - cTemp.real()) / (1 - NewBaseAxis.nz*NewBaseAxis.nz);

	if (abs(dTemp - 1) < ZeroJudgement)
	{
		dBeta2 = PI;
	}
	else if (abs(dTemp + 1) < ZeroJudgement)
	{
		dBeta2 = -PI;
	}
	else
	{
		dBeta2 = acos(1-2*dTemp);
	}
    qcomplex_t cTemp1(cos(dBeta2 / 2), -sin(dBeta2 / 2)*NewBaseAxis.nz);
    qcomplex_t cTemp2(-sin(dBeta2 / 2)*NewBaseAxis.ny, -sin(dBeta2 / 2)*NewBaseAxis.nx);

    if (abs(abs(cTemp) - 1) < ZeroJudgement)
    {
        dBeta3 = 0;
        dBeta1 = -2 * argc(NewMatrix[0] / cTemp1);
    }
    else if (abs(cTemp) < ZeroJudgement)
    {
        dBeta3 = 0;
        dBeta1 = -2 * argc(NewMatrix[1] / cTemp2);
    }
    else
    {
        cTemp1 = NewMatrix[0] / cTemp1;
        cTemp2 = NewMatrix[1] / cTemp2;
        dBeta1 = -argc(cTemp1) - argc(cTemp2);
        dBeta3 = -argc(cTemp1) + argc(cTemp2);
    }

    vdAngle.push_back(dBeta1);
    vdAngle.push_back(dBeta2);
    vdAngle.push_back(dBeta3);
}

/******************************************************************
Name        : decomposeUnitarySingleQGateIntoMetadataSingleQGate
Description : single gate conversion to metadata single
quantum gate
argin       : pNode              Target gate pointer
pParentNode        Target gate's parent node
traversalAlgorithm traversalAlgorithm pointer
argout      :
Return      :
******************************************************************/
void DecomposeUnitarySingleQGateIntoMetadataSingleQGate::execute(std::shared_ptr<AbstractQGateNode>  cur_node, 
    std::shared_ptr<QNode> parent_node)
{
	if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
	{
		return;
	}

    /*
    * Check if the quantum gate is supported
    */
	auto & type = TransformQGateType::getInstance();

	for (auto aiter : m_qgate_matrix[0])
	{
		if (cur_node->getQGate()->getGateType() == type[aiter])
		{
			return;
		}
	}

    QVec vTargetQubit;
    if (cur_node->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QuantumGate * qgate = cur_node->getQGate();
    QStat qmatrix;
    qgate->getMatrix(qmatrix);

    vector<double> vdAngle;

    getDecompositionAngle(qmatrix, vdAngle);

    auto qcircuit = CreateEmptyCircuit();

    if (m_valid_qgate_matrix[0][0] == "RX")
    {
        if (m_valid_qgate_matrix[0][1] == "RY")
        {
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << RY(vTargetQubit[0], vdAngle[1])
                << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "RZ")
        {
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << RZ(vTargetQubit[0], vdAngle[1])
                << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "U1")
        {
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << U1(vTargetQubit[0], vdAngle[1]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "H")
        {
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << RX(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "Y1")
        {
            auto ygate = Y1(vTargetQubit[0]);
            ygate.setDagger(1);
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << ygate
                << RX(vTargetQubit[0], vdAngle[1]) << Y1(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "Z1")
        {
            auto zgate = Z1(vTargetQubit[0]);
            zgate.setDagger(1);
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << zgate
                << RX(vTargetQubit[0], vdAngle[1]) << Z1(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "S")
        {
            auto S_dagger = S(vTargetQubit[0]);
            S_dagger.setDagger(1);
            qcircuit << RX(vTargetQubit[0], vdAngle[2]) << S_dagger
                << RX(vTargetQubit[0], vdAngle[1]) << S(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
    }
    else if (m_valid_qgate_matrix[0][0] == "RY")
    {
        if (m_valid_qgate_matrix[0][1] == "RX")
        {
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << RX(vTargetQubit[0], vdAngle[1]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "RZ")
        {
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << RZ(vTargetQubit[0], vdAngle[1]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "U1")
        {
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << U1(vTargetQubit[0], vdAngle[1]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "H")
        {
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << RY(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "X1")
        {
            auto xgate = X1(vTargetQubit[0]);
            xgate.setDagger(1);
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << xgate
                << RY(vTargetQubit[0], vdAngle[1]) << X1(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "Z1")
        {
            auto zgate = Z1(vTargetQubit[0]);
            zgate.setDagger(1);
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << zgate
                << RY(vTargetQubit[0], vdAngle[1]) << Z1(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "S")
        {
            auto S_dagger = S(vTargetQubit[0]);
            S_dagger.setDagger(1);
            qcircuit << RY(vTargetQubit[0], vdAngle[2]) << S_dagger
                << RY(vTargetQubit[0], vdAngle[1]) << S(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
    }
    else if (m_valid_qgate_matrix[0][0] == "RZ")
    {
        if (m_valid_qgate_matrix[0][1] == "RX")
        {
            qcircuit << RZ(vTargetQubit[0], vdAngle[2]) << RX(vTargetQubit[0], vdAngle[1]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "RY")
        {
            qcircuit << RZ(vTargetQubit[0], vdAngle[2]) << RY(vTargetQubit[0], vdAngle[1]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "H")
        {
            qcircuit << RZ(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << RZ(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "X1")
        {
            auto xgate = X1(vTargetQubit[0]);
            xgate.setDagger(1);
            qcircuit << RZ(vTargetQubit[0], vdAngle[2]) << xgate
                << RZ(vTargetQubit[0], vdAngle[1]) << X1(vTargetQubit[0]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "Y1")
        {
            auto ygate = Y1(vTargetQubit[0]);
            ygate.setDagger(1);
            qcircuit << RZ(vTargetQubit[0], vdAngle[2]) << ygate
                << RZ(vTargetQubit[0], vdAngle[1]) << Y1(vTargetQubit[0]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
    }
    else if (m_valid_qgate_matrix[0][0] == "U1")
    {
        if (m_valid_qgate_matrix[0][1] == "RX")
        {
            qcircuit << U1(vTargetQubit[0], vdAngle[2]) << RX(vTargetQubit[0], vdAngle[1]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "RY")
        {
            qcircuit << U1(vTargetQubit[0], vdAngle[2]) << RY(vTargetQubit[0], vdAngle[1]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "H")
        {
            qcircuit << U1(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << U1(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "X1")
        {
            auto xgate = X1(vTargetQubit[0]);
            xgate.setDagger(1);
            qcircuit << U1(vTargetQubit[0], vdAngle[2]) << xgate
                << U1(vTargetQubit[0], vdAngle[1]) << X1(vTargetQubit[0]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (m_valid_qgate_matrix[0][1] == "Y1")
        {
            auto ygate = Y1(vTargetQubit[0]);
            ygate.setDagger(1);
            qcircuit << U1(vTargetQubit[0], vdAngle[2]) << ygate
                << U1(vTargetQubit[0], vdAngle[1]) << Y1(vTargetQubit[0]) << U1(vTargetQubit[0], vdAngle[0]);
        }
    }
	else if ((m_valid_qgate_matrix[0][0] == "U3") || (m_valid_qgate_matrix[0][0] == "U4"))
	{
	    return;
	}
    else
    {
	    QCERR_AND_THROW_ERRSTR(run_fail, "Unknown internal error");
    }
    
    if(cur_node->isDagger())
    {
        qcircuit.setDagger(qcircuit.isDagger()^true);
    }

    replace_qcircuit(cur_node.get(), qcircuit, parent_node.get());
}

void DeleteUnitQNode::execute(std::shared_ptr<AbstractQGateNode>  cur_node, 
    std::shared_ptr<QNode> parent_node)
{
	if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
	{
		return;
	}

    auto qgate = cur_node->getQGate();
	const auto type = qgate->getGateType();
	if ((ECHO_GATE == type) || (BARRIER_GATE == type))
	{
		return;
	}

    QStat qmatrix;
    qgate->getMatrix(qmatrix);

    if (qmatrix.size() == SingleGateMatrixSize
        && abs(qmatrix[0] - qmatrix[3]) < ZeroJudgement
        && abs(abs(qmatrix[0]) - 1) < ZeroJudgement)
    {
        if (CIRCUIT_NODE == parent_node->getNodeType())
        {
            auto pQCircuitNode = dynamic_cast<AbstractQuantumCircuit *>(parent_node.get());
            if (pQCircuitNode != nullptr)
            {
                auto aiter = pQCircuitNode->getFirstNodeIter();
                for (; aiter != pQCircuitNode->getEndNodeIter(); ++aiter)
                {
                    auto temp = dynamic_cast<QNode *>(cur_node.get());
                    if (temp == (*aiter).get())
                    {
                        break;
                    }
                }
                aiter = pQCircuitNode->deleteQNode(aiter);
            }
        }
        else if (PROG_NODE == parent_node->getNodeType())
        {
            auto pQProgNode = dynamic_cast<AbstractQuantumProgram *>(parent_node.get());
            auto aiter = pQProgNode->getFirstNodeIter();
            for (; aiter != pQProgNode->getEndNodeIter(); ++aiter)
            {
                auto temp = dynamic_cast<QNode *>(cur_node.get());
                if (temp == (*aiter).get())
                {
                    break;
                }
            }
            aiter = pQProgNode->deleteQNode(aiter);
        }
    }

}

/******************************************************************
Name        : TraversalOptimizationMerge
Description : Traversal optimization merge algorithm
argin       : pNode    Target Node pointer
argout      : pNode    Target Node pointer
Return      :
******************************************************************/
void TransformDecomposition::TraversalOptimizationMerge(QProg & prog)
{
	decompose_double_qgate(prog);

	meta_gate_transform(prog);
}

void TransformDecomposition::decompose_double_qgate(QProg & prog, bool b_decompose_multiple_gate /*= true*/)
{
	flatten(prog, true);

	//double gate to : x + cnot + cu
	Traversal::traversal(prog.getImplementationPtr(), m_decompose_double_gate);
	
	// decompose cu
	Traversal::traversal(prog.getImplementationPtr(), m_decompose_control_unitary_single_qgate);

	if (b_decompose_multiple_gate)
	{
		if (!(CheckMultipleControlQGate().exist_multiple_gate(prog)))
		{
			return;
		}

		Traversal::traversal(prog.getImplementationPtr(), m_decompose_multiple_control_qgate);
		Traversal::traversal(prog.getImplementationPtr(), m_cancel_control_qubit_vector);
		Traversal::traversal(prog.getImplementationPtr(), m_decompose_control_unitary_single_qgate);
	}
}

void TransformDecomposition::meta_gate_transform(QProg& prog)
{
	flatten(prog, true);
	Traversal::traversal(prog.getImplementationPtr(), m_control_single_qgate_to_metadata_double_qgate);
	Traversal::traversal(prog.getImplementationPtr(), m_unitary_single_qgate_to_metadata_single_qgate);
	merge_continue_single_gate_to_u3(prog);
	Traversal::traversal(prog.getImplementationPtr(), m_delete_unit_qnode);
}

void TransformDecomposition::merge_continue_single_gate_to_u3(QProg& prog)
{
	if (m_valid_qgate_matrix[0][0] == "U3")
	{
		sub_cir_optimizer(prog, {}, QCircuitOPtimizerMode::Merge_U3);
	}
}

/*****************************************************************
Name        : TransformDecomposition
Description : structure TransformDecomposition
argin       : ValidQGateMatrix  Validated instruction set
QGateMatrix       Original instruction set
argout      :
Return      :
*****************************************************************/
TransformDecomposition::
TransformDecomposition(vector<vector<string>> &valid_qgate_matrix,
	vector<vector<string>> &qgate_matrix,
	QuantumMachine *quantum_machine) :
	m_control_single_qgate_to_metadata_double_qgate(quantum_machine,
		valid_qgate_matrix),
	m_unitary_single_qgate_to_metadata_single_qgate(qgate_matrix, valid_qgate_matrix),
	m_decompose_control_unitary_single_qgate(valid_qgate_matrix),
	m_decompose_double_gate(valid_qgate_matrix),
    m_quantum_machine(quantum_machine), m_valid_qgate_matrix(valid_qgate_matrix)
{
	if ((valid_qgate_matrix[MetadataGateType::METADATA_SINGLE_GATE].size() == 0) ||
		(valid_qgate_matrix[MetadataGateType::METADATA_DOUBLE_GATE].size() == 0))
	{
		QCERR_AND_THROW_ERRSTR(init_fail, "Error: The selected underlying QGate is not a valid metadata composition, refer to:\
			https://qpanda-tutorial.readthedocs.io/zh/latest/QGateValidity.html");
	}
}

TransformDecomposition::~TransformDecomposition()
{
}

void MergeSingleGate::execute(std::shared_ptr<AbstractQuantumCircuit>  cur_node, 
    std::shared_ptr<QNode> parent_node)
{
    if (!cur_node)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    auto aiter = cur_node->getFirstNodeIter();

    /*
    * Traversal PNode's children node
    */
    for (; aiter != cur_node->getEndNodeIter(); ++aiter)
    {
        int node_type = (*aiter)->getNodeType();

        /*
        * If it is not a gate type, the algorithm goes deeper
        */
        if (CLASS_COND_NODE == node_type)
            continue;
        else if (CIRCUIT_NODE == node_type)
        {
            Traversal::traversal(
                dynamic_pointer_cast<AbstractQuantumCircuit>(*aiter),false,
                *this
                );

            continue;
        }

        AbstractQGateNode * cur_gate_node =
            dynamic_pointer_cast<AbstractQGateNode>(*aiter).get();

        if (cur_gate_node->getTargetQubitNum()
            +cur_gate_node->getControlQubitNum() >= 2)
            continue;

        auto next_iter = aiter.getNextIter();

        AbstractQGateNode * next_gate_node = nullptr;

        /*
        * Loop through the nodes behind the target node
        * and execute the merge algorithm
        */
        int next_node_type = NODE_UNDEFINED;
        while (next_iter != cur_node->getEndNodeIter())
        {
            int next_node_type = (*next_iter)->getNodeType();

            if (CLASS_COND_NODE == next_node_type)
                continue;
            else if (GATE_NODE != next_node_type)
                break;

            next_gate_node = 
                dynamic_pointer_cast<AbstractQGateNode>(*next_iter).get();

            if (next_gate_node->getTargetQubitNum()
                +next_gate_node->getControlQubitNum() == 1)
            {
                QVec CurQubitVector;
                cur_gate_node->getQuBitVector(CurQubitVector);

                QVec NextQubitVector;
                next_gate_node->getQuBitVector(NextQubitVector);

                auto cur_phy_qubit = CurQubitVector[0]->getPhysicalQubitPtr();
                auto next_phy_qubit = NextQubitVector[0]->getPhysicalQubitPtr();

                if ((nullptr == cur_phy_qubit) || (nullptr == next_phy_qubit))
                {
                    QCERR("Unknown internal error");
                    throw std::runtime_error("Unknown internal error");
                }

                /*
                * Determine if it is the same qubit
                */
                if (cur_phy_qubit->getQubitAddr() == 
                    next_phy_qubit->getQubitAddr())
                {
                    auto pCurQGate = cur_gate_node->getQGate();
                    auto pNextQGate = next_gate_node->getQGate();

                    if ((nullptr == pCurQGate) || (nullptr == pNextQGate))
                    {
                        QCERR("Unknown internal error");
                        throw std::runtime_error("Unknown internal error");
                    }

                    /*
                    * Merge node
                    */
                    QStat CurMatrix, NextMatrix;
                    pCurQGate->getMatrix(CurMatrix);
                    pNextQGate->getMatrix(NextMatrix);
                    QStat newMatrix = NextMatrix * CurMatrix;
                    auto temp = U4(newMatrix, CurQubitVector[0]);
                    auto pCurtItem = aiter.getPCur();
                    pCurtItem->setNode(dynamic_pointer_cast<QNode>(temp.getImplementationPtr()));
                    cur_gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(pCurtItem->getNode()).get();
                    next_iter = cur_node->deleteQNode(next_iter);
                }
            }
            next_iter = next_iter.getNextIter();
        }
    }
}

void MergeSingleGate::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node,
    std::shared_ptr<QNode> parent_node)
{
    if (!cur_node)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    auto aiter = cur_node->getFirstNodeIter();

    /*
    * Traversal PNode's children node
    */
    for (; aiter != cur_node->getEndNodeIter(); ++aiter)
    {
        int node_type = (*aiter)->getNodeType();

        /*
        * If it is not a gate type, the algorithm goes deeper
        */
        if (CLASS_COND_NODE == node_type)
            continue;
        else if (CIRCUIT_NODE == node_type)
        {
            Traversal::traversal(
                dynamic_pointer_cast<AbstractQuantumCircuit>(*aiter),
                false,
                *this
                );
            continue;
        }
        else if (GATE_NODE != node_type)
        {
            Traversal::traversalByType(*aiter, parent_node, *this);
            continue;
        }

        AbstractQGateNode * cur_gate_node =
            dynamic_pointer_cast<AbstractQGateNode>(*aiter).get();

        if (cur_gate_node->getTargetQubitNum() + 
            cur_gate_node->getControlQubitNum() >= 2)
            continue;

        auto next_iter = aiter.getNextIter();

        AbstractQGateNode * next_gate_node = nullptr;

        /*
        * Loop through the nodes behind the target node
        * and execute the merge algorithm
        */
        int next_node_type = NODE_UNDEFINED;
        while (next_iter != cur_node->getEndNodeIter())
        {
            int next_node_type = (*next_iter)->getNodeType();

            if (CLASS_COND_NODE == next_node_type)
                continue;
            else if (GATE_NODE != next_node_type)
                break;

            next_gate_node =
                dynamic_pointer_cast<AbstractQGateNode>(*next_iter).get();

            if (next_gate_node->getTargetQubitNum() + 
                next_gate_node->getControlQubitNum() == 1)
            {
                QVec CurQubitVector;
                cur_gate_node->getQuBitVector(CurQubitVector);

                QVec NextQubitVector;
                next_gate_node->getQuBitVector(NextQubitVector);

                auto cur_phy_qubit = CurQubitVector[0]->getPhysicalQubitPtr();
                auto next_phy_qubit = NextQubitVector[0]->getPhysicalQubitPtr();

                if ((nullptr == cur_phy_qubit) || (nullptr == next_phy_qubit))
                {
                    QCERR("Unknown internal error");
                    throw std::runtime_error("Unknown internal error");
                }

                /*
                * Determine if it is the same qubit
                */
                if (cur_phy_qubit->getQubitAddr() ==
                    next_phy_qubit->getQubitAddr())
                {
                    auto pCurQGate = cur_gate_node->getQGate();
                    auto pNextQGate = next_gate_node->getQGate();

                    if ((nullptr == pCurQGate) || (nullptr == pNextQGate))
                    {
                        QCERR("Unknown internal error");
                        throw std::runtime_error("Unknown internal error");
                    }

                    /*
                    * Merge node
                    */
                    QStat CurMatrix, NextMatrix;
                    pCurQGate->getMatrix(CurMatrix);
                    pNextQGate->getMatrix(NextMatrix);
                    QStat newMatrix = NextMatrix * CurMatrix;
                    
                    auto temp = U4(newMatrix, CurQubitVector[0]);
                    auto pCurtItem = aiter.getPCur();
                    pCurtItem->setNode(dynamic_pointer_cast<QNode>(temp.getImplementationPtr()));
                    cur_gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(pCurtItem->getNode()).get();
                    next_iter = cur_node->deleteQNode(next_iter);
                }
            }
            next_iter = next_iter.getNextIter();
        }
    }
}

/*******************************************************************
*                      public interface
********************************************************************/
void QPanda::decompose_multiple_control_qgate(QProg& prog, QuantumMachine *quantum_machine, const std::string& config_data/* = CONFIG_PATH*/)
{
	if (!(CheckMultipleControlQGate().exist_multiple_gate(prog)))
	{
		transform_to_base_qgate(prog, quantum_machine, config_data);
		return;
	}

	QuantumMetadata meta_data(config_data);
	std::vector<string> vec_single_gates;
	std::vector<string> vec_double_gates;
	meta_data.getQGate(vec_single_gates, vec_double_gates);

	std::vector<std::vector<std::string>> gates(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
	std::vector<std::vector<std::string>> valid_gate(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
	for (auto& item : vec_single_gates)
	{
		gates[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(item);
	}
	for (auto& item : vec_double_gates)
	{
		gates[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(item);
	}
	SingleGateTypeValidator::GateType(gates[MetadataGateType::METADATA_SINGLE_GATE],
		valid_gate[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
	DoubleGateTypeValidator::GateType(gates[MetadataGateType::METADATA_DOUBLE_GATE],
		valid_gate[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */

	auto p_transf_decompos = std::make_shared<TransformDecomposition>(valid_gate, gates, quantum_machine);
	p_transf_decompos->TraversalOptimizationMerge(prog);
}

void QPanda::decompose_multiple_control_qgate(QCircuit& cir, QuantumMachine *quantum_machine, const std::string& config_data/* = CONFIG_PATH*/)
{
	QProg tmp_prog(cir);
	decompose_multiple_control_qgate(tmp_prog, quantum_machine, config_data);

	cir = QProgFlattening::prog_flatten_to_cir(tmp_prog);
}

void QPanda::transform_to_base_qgate(QProg& prog, QuantumMachine *quantum_machine, const std::string& config_data/* = CONFIG_PATH*/)
{
	if ((CheckMultipleControlQGate().exist_multiple_gate(prog)))
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The target quantum circuit or program cannot contain multiple-control gates.");
	}

	QuantumMetadata meta_data(config_data);
	std::vector<string> vec_single_gates;
	std::vector<string> vec_double_gates;
	meta_data.getQGate(vec_single_gates, vec_double_gates);

	std::vector<std::vector<std::string>> gates(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
	std::vector<std::vector<std::string>> valid_gate(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
	for (auto& item : vec_single_gates)
	{
		gates[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(item);
	}
	for (auto& item : vec_double_gates)
	{
		gates[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(item);
	}
	SingleGateTypeValidator::GateType(gates[MetadataGateType::METADATA_SINGLE_GATE],
		valid_gate[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
	DoubleGateTypeValidator::GateType(gates[MetadataGateType::METADATA_DOUBLE_GATE],
		valid_gate[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */

	auto p_transf_decompos = std::make_shared<TransformDecomposition>(valid_gate, gates, quantum_machine);
	p_transf_decompos->decompose_double_qgate(prog, false);
	p_transf_decompos->meta_gate_transform(prog);
}

void QPanda::transform_to_base_qgate(QCircuit& cir, QuantumMachine *quantum_machine, const std::string& config_data/* = CONFIG_PATH*/)
{
	QProg tmp_prog(cir);
	transform_to_base_qgate(tmp_prog, quantum_machine, config_data);

	cir = QProgFlattening::prog_flatten_to_cir(tmp_prog);
}