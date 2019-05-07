/******************************************************************
Filename     : TransformDecomposition.cpp
Creator      : Menghan Dou��cheng xue
Create time  : 2018-07-04
Description  : Quantum program adaptation metadata instruction set
*******************************************************************/
#include "TransformDecomposition.h"
#include "QPanda.h"
#include "QGateCompare.h"
#include "Utilities/ComplexMatrix.h"
#include "QPandaException.h"
#include "Core/Utilities/Utilities.h"
#define iunit qcomplex_t(0,1)
USING_QPANDA
using namespace std;


inline double getArgument(qcomplex_t cNumber)
{
    if (cNumber.imag() >= 0)
    {
        return acos(cNumber.real() / sqrt(cNumber.real()*cNumber.real() + cNumber.imag()*cNumber.imag()));
    }
    else
    {
        return -acos(cNumber.real() / sqrt(cNumber.real()*cNumber.real() + cNumber.imag()*cNumber.imag()));
    }
}

double transformMatrixToAxis(QStat &QMatrix, axis &Axis)
{

    double dRotateAngle;

    double a0 = abs((QMatrix[0] + QMatrix[3])) / 2.0;
    double a1 = abs((QMatrix[1] + QMatrix[2])) / 2.0;
    double a2 = abs((QMatrix[1] - QMatrix[2])*qcomplex_t(0, 1)) / 2.0;
    double a3 = abs((QMatrix[0] - QMatrix[3])) / 2.0;

    dRotateAngle = acos(abs(a0));

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

        Axis.nx = a1;
        Axis.ny = a2;
        Axis.nz = a3;

        double dSum = Axis.nx*Axis.nx + Axis.ny*Axis.ny + Axis.nz*Axis.nz;
        dSum = sqrt(dSum);

        Axis.nx = Axis.nx / dSum;
        Axis.ny = Axis.ny / dSum;
        Axis.nz = Axis.nz / dSum;
    }

    return 2 * dRotateAngle;

}

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
        if (OriginAxis.ny >= 0)
        {
            dAlpha = acos(OriginAxis.nx / sqrt(OriginAxis.nx*OriginAxis.nx + OriginAxis.ny * OriginAxis.ny));
        }
        else
        {
            dAlpha = -acos(OriginAxis.nx / sqrt(OriginAxis.nx*OriginAxis.nx + OriginAxis.ny * OriginAxis.ny));
        }
    }

    dTheta = acos(OriginAxis.nz);

    qcomplex_t cTemp1 = QMatrix[0] * cos(dTheta / 2) +
        QMatrix[1] * sin(dTheta / 2)*qcomplex_t(cos(dAlpha), sin(dAlpha));
    qcomplex_t cTemp2 = QMatrix[2] * cos(dTheta / 2) +
        QMatrix[3] * sin(dTheta / 2)*qcomplex_t(cos(dAlpha), sin(dAlpha));

    if (abs(abs(cTemp1) - 1) < ZeroJudgement)
    {
        dTheta = 0;
        dAlpha = 0;
    }
    else if (abs(abs(cTemp2) - 1) < ZeroJudgement)
    {
        dTheta = PI;
        dAlpha = 0;
    }
    else
    {
        dTheta = 2 * acos(abs(cTemp1));
        dAlpha = getArgument(cTemp2) - getArgument(cTemp1);
    }

    NewAxis.nx = sin(dTheta)*cos(dAlpha);
    NewAxis.ny = sin(dTheta)*sin(dAlpha);
    NewAxis.nz = cos(dTheta);

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

void DecomposeDoubleQGate::execute(AbstractQGateNode * pNode,
    QNode * pParentNode)
{
    if (nullptr == pNode)
    {
        QCERR("pnode is null");
        throw invalid_argument("pnode is null");
    }
    QuantumGate* qGate;
    qGate = pNode->getQGate();

    if (pNode->getTargetQubitNum() == 1)
    {
        return;
    }

    QVec vQubit;

    if (pNode->getQuBitVector(vQubit) <= 0)
    {
        QCERR("the num of qubit vector error ");
        throw runtime_error("the num of qubit vector error");
    }

    QStat vMatrix;
    qGate->getMatrix(vMatrix);

    QStat vMatrix1(DoubleGateMatrixSize, 0);
    QStat vMatrix2(SingleGateMatrixSize, 0);

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

                vMatrix2[0] = qcomplex_t(vMatrix[5 * i].real(), -vMatrix[5 * i].imag()) / dSum;
                vMatrix2[1] = qcomplex_t(vMatrix[SingleGateMatrixSize * j + i].real(),
                    -vMatrix[SingleGateMatrixSize * j + i].imag()) / dSum;
                vMatrix2[2] = vMatrix[SingleGateMatrixSize * j + i] / dSum;
                vMatrix2[3] = -vMatrix[5 * i] / dSum;

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
                        << CU(vMatrix2, vQubit[0], vQubit[1])
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

    auto qCircuitDagger = qCircuit.dagger();

    insertQCircuit(pNode, qCircuitDagger, pParentNode);
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
        dAlpha = getArgument(pNodeMatrix[0] / (QMatrix[0] * QMatrix[0] + QMatrix[1] * QMatrix[2]));
    }
    else
    {
        dAlpha = getArgument(pNodeMatrix[1] / (QMatrix[0] * QMatrix[1] + QMatrix[1] * QMatrix[3]));
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

    QStat qMatrix;
    qGate->getMatrix(qMatrix);

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

            auto qNode2 = U4(qMatrix, vTargetQubit[0]);
            qNode2.setControl(vDownQubits);

            if (vDownQubits.size() >= 3)
            {
                vTempQubits.insert(vTempQubits.begin(), vUpQubits.begin(),
                    vUpQubits.begin() + vDownQubits.size() - 2);
            }

            auto qCircuit2 = secondStepOfMultipleControlQGateDecomposition(&qNode2, vTempQubits);

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

            auto qNode4 = U4(qMatrix, vTargetQubit[0]);

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

    QuantumGate* qGate = pNode->getQGate();

    qGate->getMatrix(qMatrix);

    if (vControlQubit.size() >2 && (vControlQubit.size() - AncillaQubitVector.size() == 2) && vTargetQubit.size() == 1)
    {
        vqtemp[0] = vControlQubit[vControlQubit.size() - 1];
        vqtemp[1] = AncillaQubitVector[AncillaQubitVector.size() - 1];

        auto qGate1 = U4(qMatrix, vTargetQubit[0]);

        qGate1.setControl(vqtemp);

        qCircuit << decomposeTwoControlSingleQGate(&qGate1);
        qCircuit << tempStepOfMultipleControlQGateDecomposition(vControlQubit, AncillaQubitVector);
        qCircuit << decomposeTwoControlSingleQGate(&qGate1);
        qCircuit << tempStepOfMultipleControlQGateDecomposition(vControlQubit, AncillaQubitVector);
    }
    else if (vControlQubit.size() == 2)
    {
        vqtemp[0] = vControlQubit[0];
        vqtemp[1] = vControlQubit[1];

        auto qGate2 = U4(qMatrix, vTargetQubit[0]);
        qGate2.setControl(vqtemp);

        qCircuit << decomposeTwoControlSingleQGate(&qGate2);
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

void DecomposeMultipleControlQGate::execute(AbstractQGateNode *node, QNode * parent_node)
{
    QVec vTargetQubit;

    if (node->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QVec vControlQubit;

    if (CIRCUIT_NODE == parent_node->getNodeType())
    {
        AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(parent_node);
        pQcir->getControlVector(vControlQubit);
    }

    if (node->getControlVector(vControlQubit) <= 0)
    {
        return;
    }

    QuantumGate* qgate = node->getQGate();

    QStat qMatrix;
    qgate->getMatrix(qMatrix);

    QStat vMatrix;
    QStat matrixdagger;

    QGateExponentArithmetic(node, 0.5, vMatrix);

    auto qCircuit = CreateEmptyCircuit();

    auto qGate0 = CU(vMatrix, vControlQubit[vControlQubit.size() - 1], vTargetQubit[0]);
    qGate0.setDagger(1);

    if (vControlQubit.size() == 1)
    {
        qCircuit << CU(qMatrix, vControlQubit[0], vTargetQubit[0]);
    }
    else if (vControlQubit.size() == 2)
    {
        //pNode->setControl(vControlQubit);
        qCircuit << decomposeTwoControlSingleQGate(node);
    }
    else if (vControlQubit.size() == 3)
    {
        auto qGate = U4(vMatrix, vTargetQubit[0]);

        vector<Qubit*> vTempQubit;

        vTempQubit.push_back(vControlQubit[0]);
        vTempQubit.push_back(vControlQubit[1]);

        qGate.setControl(vTempQubit);

        qCircuit << CU(vMatrix, vControlQubit[2], vTargetQubit[0])
            << decomposeToffoliQGate(vControlQubit[2], { vControlQubit[0],vControlQubit[1] })
            << qGate0 << decomposeToffoliQGate(vControlQubit[2], { vControlQubit[0],vControlQubit[1] })
            << decomposeTwoControlSingleQGate(&qGate);
    }
    else if (vControlQubit.size() > 3)
    {

        Qubit* temp = vControlQubit[vControlQubit.size() - 1];

        auto qGate1 = X(temp);

        vControlQubit.pop_back();

        qGate1.setControl(vControlQubit);

        auto qCircuit1 = firstStepOfMultipleControlQGateDecomposition(&qGate1, vTargetQubit[0]);
        auto qCircuit2 = firstStepOfMultipleControlQGateDecomposition(&qGate1, vTargetQubit[0]);

        auto qGate2 = U4(vMatrix, vTargetQubit[0]);
        qGate2.setControl(vControlQubit);

        auto qCircuit3 = firstStepOfMultipleControlQGateDecomposition(&qGate2, temp);

        qCircuit << CU(vMatrix, vControlQubit[vControlQubit.size() - 1], vTargetQubit[0]) << qCircuit1         //CV and CC..C-NOT
            << qGate0 << qCircuit2 << qCircuit3;
    }

    insertQCircuit(node, qCircuit, parent_node);
}

void DecomposeMultipleControlQGate::execute(AbstractQuantumCircuit * node, QNode * parent_node)
{
    if (nullptr == node)
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
        AbstractQuantumCircuit *parent_qcircuit = dynamic_cast<AbstractQuantumCircuit*>(parent_node);
        QVec vControlQubit;
        parent_qcircuit->getControlVector(vControlQubit);
        node->setControl(vControlQubit);
    }

    Traversal::traversal(node, this, false);
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
void DecomposeControlUnitarySingleQGate::execute(AbstractQGateNode * node, QNode * parent_node)
{
    if (node->getTargetQubitNum() == 1)
    {
        return;
    }

    auto target_qubit = node->popBackQuBit();
    auto control_qubit = node->popBackQuBit();

    node->PushBackQuBit(target_qubit);

    vector<Qubit *> vControlQubit = { control_qubit };

    node->setControl(vControlQubit);

    auto qgate = node->getQGate();

    if (nullptr == qgate)
    {
        QCERR("qgate is null");
        throw runtime_error("qgate is null");
    }

    QVec qubitVector;

    if (node->getQuBitVector(qubitVector) <= 0)
    {
        QCERR("the size of qubit vector is error");
        throw runtime_error("the size of qubit vector is error");
    }

    auto targetQubit = qubitVector[0];

    auto pU4 = new QGATE_SPACE::U4(qgate->getAlpha(),
        qgate->getBeta(),
        qgate->getGamma(),
        qgate->getDelta());
    delete(qgate);
    node->setQGate(pU4);
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
     execute(AbstractQGateNode * node,
     QNode * parent_node)
{

    string sGateName = m_valid_qgate_matrix[1][0];

    if (sGateName.size() <= 0)
    {
        QCERR("the size of sGateName is error");
        throw runtime_error("the size of sGateName is error");
    }

    QVec vTargetQubit;
    if (node->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QVec vControlQubit;
    if (node->getControlVector(vControlQubit) != 1)
    {
        return;
    }

    if (CIRCUIT_NODE == parent_node->getNodeType())
    {
        AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(parent_node);
        pQcir->getControlVector(vControlQubit);
    }
    QuantumGate* qgate = node->getQGate();

    double dAlpha = qgate->getAlpha();
    double dBeta = qgate->getBeta();
    double dDelta = qgate->getDelta();
    double dGamma = qgate->getGamma();

    auto qCircuit = CreateEmptyCircuit();

    QStat QMatrix(SingleGateMatrixSize, 0);

    QMatrix[0] = 1;
    QMatrix[3] = qcomplex_t(cos(dAlpha), sin(dAlpha));

    auto gat = U4(QMatrix, vControlQubit[0]);

    auto qSwap = CreateEmptyCircuit();
    auto qSwapDagger = CreateEmptyCircuit();

    if (m_adjacent_matrix.size() != 0)
    {
        int iBeginNumber = (int)vControlQubit[0]->getPhysicalQubitPtr()->getQubitAddr();
        int iEndNumber = (int)vTargetQubit[0]->getPhysicalQubitPtr()->getQubitAddr();

        vector<int> viShortestConnection;

        GraphDijkstra gd(m_adjacent_matrix);

        int iDistance = gd.getShortestPath(iBeginNumber, iEndNumber, viShortestConnection);

        if (viShortestConnection.size() > 2)
        {
            qSwap = swapQGate(viShortestConnection, sGateName);
            qSwapDagger = swapQGate(viShortestConnection, sGateName);
            qSwapDagger.setDagger(true);
        }
    }
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

    insertQCircuit(node, qCircuit, parent_node);
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
            QStat QMatrix = { cos(PI / 4),-sin(PI / 4),sin(PI / 4),cos(PI / 4) };
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
            QStat QMatrix = { cos(PI / 4),-iunit * sin(PI / 4),-iunit * sin(PI / 4) ,cos(PI / 4) };
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
            QStat QMatrix = { cos(PI / 4),-sin(PI / 4),sin(PI / 4),cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (valid_qgate_matrix[0][1] == "X1")
        {
            QStat QMatrix = { cos(PI / 4),-iunit * sin(PI / 4),-iunit * sin(PI / 4) ,cos(PI / 4) };
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
            dAlpha = acos(base.n1.nx / sqrt(base.n1.nx*base.n1.nx + base.n1.ny * base.n1.ny));
        }
        else
        {
            dAlpha = -acos(base.n1.nx / sqrt(base.n1.nx*base.n1.nx + base.n1.ny * base.n1.ny));
        }
    }

    dTheta = acos(base.n1.nz);

    QStat UnitaryMatrix;
    UnitaryMatrix.resize(SingleGateMatrixSize);

    UnitaryMatrix[0] = qcomplex_t(cos(-dTheta / 2), 0);
    UnitaryMatrix[1] = qcomplex_t(sin(dTheta / 2)*cos(dAlpha), -sin(dTheta / 2)*sin(dAlpha));
    UnitaryMatrix[2] = qcomplex_t(-sin(dTheta / 2)*cos(dAlpha), -sin(dTheta / 2)*sin(dAlpha));
    UnitaryMatrix[3] = qcomplex_t(cos(-dTheta / 2), 0);

    axis TargetAxis;

    double dBeta = transformMatrixToAxis(qmatrix, TargetAxis);
    double dBeta1;
    double dBeta2;
    double dBeta3;

    axis NewBaseAxis;
    axis NewTargetAxis;

    rotateAxis(UnitaryMatrix, base.n2, NewBaseAxis);
    rotateAxis(UnitaryMatrix, TargetAxis, NewTargetAxis);

    QStat NewMatrix(SingleGateMatrixSize);

    NewMatrix[0] = qcomplex_t(cos(dBeta / 2), -sin(dBeta / 2)*NewTargetAxis.nz);
    NewMatrix[1] = qcomplex_t(-sin(dBeta / 2)*NewTargetAxis.ny, -sin(dBeta / 2)*NewTargetAxis.nx);
    NewMatrix[2] = qcomplex_t(sin(dBeta / 2)*NewTargetAxis.ny, -sin(dBeta / 2)*NewTargetAxis.nx);
    NewMatrix[3] = qcomplex_t(cos(dBeta / 2), sin(dBeta / 2)*NewTargetAxis.nz);

    qcomplex_t cTemp = NewMatrix[0] * NewMatrix[3];

    double dTemp = (1 - cTemp.real()) / (1 - NewBaseAxis.nz*NewBaseAxis.nz);

    dBeta2 = 2 * asin(sqrt(dTemp));

    qcomplex_t cTemp1(cos(dBeta2 / 2), -sin(dBeta2 / 2)*NewBaseAxis.nz);
    qcomplex_t cTemp2(-sin(dBeta2 / 2)*NewBaseAxis.ny, -sin(dBeta2 / 2)*NewBaseAxis.nx);

    if (abs(abs(cTemp) - 1) < ZeroJudgement)
    {
        dBeta3 = 0;
        dBeta1 = -2 * getArgument(NewMatrix[0] / cTemp1);
    }
    else if (abs(cTemp) < ZeroJudgement)
    {
        dBeta3 = 0;
        dBeta1 = -2 * getArgument(NewMatrix[1] / cTemp2);
    }
    else
    {
        cTemp1 = NewMatrix[0] / cTemp1;
        cTemp2 = NewMatrix[1] / cTemp2;
        dBeta1 = -getArgument(cTemp1) - getArgument(cTemp2);
        dBeta3 = -getArgument(cTemp1) + getArgument(cTemp2);
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
void DecomposeUnitarySingleQGateIntoMetadataSingleQGate::execute(AbstractQGateNode * node,
    QNode * parent_node)
{
    /*
    * Check if the quantum gate is supported
    */

    if (getUnSupportQGateNumber(*(dynamic_cast<OriginQGate *>(node)), m_qgate_matrix) <= 0)
        return;

    QVec vTargetQubit;
    if (node->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QuantumGate * qgate = node->getQGate();
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
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    insertQCircuit(node, qcircuit, parent_node);
}

void DeleteUnitQNode::execute(AbstractQGateNode * node, QNode * parent_node)
{

    auto qgate = node->getQGate();

    QStat qmatrix;
    qgate->getMatrix(qmatrix);

    if (qmatrix.size() == SingleGateMatrixSize
        && abs(qmatrix[0] - qmatrix[3]) < ZeroJudgement
        && abs(abs(qmatrix[0]) - 1) < ZeroJudgement)
    {
        if (CIRCUIT_NODE == parent_node->getNodeType())
        {
            auto pQCircuitNode = dynamic_cast<AbstractQuantumCircuit *>(parent_node);
            if (pQCircuitNode != nullptr)
            {
                auto aiter = pQCircuitNode->getFirstNodeIter();
                for (; aiter != pQCircuitNode->getEndNodeIter(); ++aiter)
                {
                    auto temp = dynamic_cast<QNode *>(node);
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
            auto pQProgNode = dynamic_cast<AbstractQuantumProgram *>(node);
            auto aiter = pQProgNode->getFirstNodeIter();
            for (; aiter != pQProgNode->getEndNodeIter(); ++aiter)
            {
                auto temp = dynamic_cast<QNode *>(node);
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
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_decompose_double_gate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_decompose_control_unitary_single_qgate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_decompose_multiple_control_qgate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_decompose_control_unitary_single_qgate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_control_single_qgate_to_metadata_double_qgate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_merge_single_gate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_unitary_single_qgate_to_metadata_double_qgate));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_cancel_control_qubit_vector));
    Traversal::traversal(&prog, static_cast<TraversalInterface *>(&m_delete_unit_qnode));
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
    vector<vector<int> > &agjacent_matrix,
    QuantumMachine *quantum_machine) :
    m_control_single_qgate_to_metadata_double_qgate(quantum_machine,
        valid_qgate_matrix,
        agjacent_matrix),
    m_unitary_single_qgate_to_metadata_double_qgate(qgate_matrix, valid_qgate_matrix)
{}



TransformDecomposition::~TransformDecomposition()
{
}

void MergeSingleGate::execute(AbstractQuantumCircuit * node, QNode * parent_node)
{
    if (nullptr == node)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    auto aiter = node->getFirstNodeIter();

    /*
    * Traversal PNode's children node
    */
    for (; aiter != node->getEndNodeIter(); ++aiter)
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
                dynamic_cast<AbstractQuantumCircuit *>((*aiter).get()),
                this,
                false);
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
        while (next_iter != node->getEndNodeIter())
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
                    pCurtItem->setNode(temp.getImplementationPtr());
                    cur_gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(pCurtItem->getNode()).get();
                    next_iter = node->deleteQNode(next_iter);
                }
            }
            next_iter = next_iter.getNextIter();
        }
    }
}

void MergeSingleGate::execute(AbstractQuantumProgram * node, QNode * parent_node)
{
    if (nullptr == node)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    auto aiter = node->getFirstNodeIter();

    /*
    * Traversal PNode's children node
    */
    for (; aiter != node->getEndNodeIter(); ++aiter)
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
                dynamic_cast<AbstractQuantumCircuit *>((*aiter).get()),
                this,
                false);
            continue;
        }
        else if (GATE_NODE != node_type)
        {
            Traversal::traversalByType((*aiter).get(), parent_node, this);
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
        while (next_iter != node->getEndNodeIter())
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
                    pCurtItem->setNode(temp.getImplementationPtr());
                    cur_gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(pCurtItem->getNode()).get();
                    next_iter = node->deleteQNode(next_iter);
                }
            }
            next_iter = next_iter.getNextIter();
        }
    }
}
