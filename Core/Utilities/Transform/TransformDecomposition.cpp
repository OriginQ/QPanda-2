/******************************************************************
Filename     : TransformDecomposition.cpp
Creator      : Menghan Dou¡¢cheng xue
Create time  : 2018-07-04
Description  : Quantum program adaptation metadata instruction set
*******************************************************************/
#include "TransformDecomposition.h"
#include "QPanda.h"
#include "QGateCompare.h"
#include "Utilities/ComplexMatrix.h"
#define iunit qcomplex_t(0,1)
USING_QPANDA
using namespace std;
inline double getArgument(qcomplex_t num)
{
    if (num.imag() >= 0)
    {
        return acos(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag()));
    }
    else
    {
        return -acos(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag()));
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
TransformDecomposition(vector<vector<string>> &ValidQGateMatrix,
    vector<vector<string>> &QGateMatrix,
    vector<vector<int> > &vAdjacentMatrix)
{
    /*
    * Initialize member variable
    */
    if (ValidQGateMatrix[0][0] == "RX")
    {
        base.n1 = { 1,0,0 };
        if (ValidQGateMatrix[0][1] == "RY")
        {
            base.n2 = { 0,1,0 };
        }
        else if (ValidQGateMatrix[0][1] == "RZ" || ValidQGateMatrix[0][1] == "U1")
        {
            base.n2 = { 0,0,1 };
        }
        else if (ValidQGateMatrix[0][1] == "H")
        {
            QStat QMatrix = { 1 / SQRT2,1 / SQRT2 ,1 / SQRT2 ,-1 / SQRT2 };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "Y1")
        {
            QStat QMatrix = { cos(PI / 4),-sin(PI / 4),sin(PI / 4),cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "Z1")
        {
            QStat QMatrix = { qcomplex_t(cos(PI / 4),-sin(PI / 4)),0,0,qcomplex_t(cos(PI / 4),+sin(PI / 4)) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "S")
        {
            QStat QMatrix = { 1,0,0,iunit };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "T")
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
    else if (ValidQGateMatrix[0][0] == "RY")
    {
        base.n1 = { 0,1,0 };
        if (ValidQGateMatrix[0][1] == "RX")
        {
            base.n2 = { 1,0,0 };
        }
        else if (ValidQGateMatrix[0][1] == "RZ" || ValidQGateMatrix[0][1] == "U1")
        {
            base.n2 = { 0,0,1 };
        }
        else if (ValidQGateMatrix[0][1] == "X1")
        {
            QStat QMatrix = { cos(PI / 4),-iunit * sin(PI / 4),-iunit * sin(PI / 4) ,cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "Z1")
        {
            QStat QMatrix = { qcomplex_t(cos(PI / 4),-sin(PI / 4)),0,0,qcomplex_t(cos(PI / 4),sin(PI / 4)) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "S")
        {
            QStat QMatrix = { 1,0,0,iunit };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "T")
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
    else if (ValidQGateMatrix[0][0] == "RZ" || ValidQGateMatrix[0][0] == "U1")
    {
        base.n1 = { 0,0,1 };
        if (ValidQGateMatrix[0][1] == "RX")
        {
            base.n2 = { 1,0,0 };
        }
        else if (ValidQGateMatrix[0][1] == "RY")
        {
            base.n2 = { 0,1,0 };
        }
        else if (ValidQGateMatrix[0][1] == "H")
        {
            QStat QMatrix = { 1 / SQRT2,1 / SQRT2 ,1 / SQRT2 ,-1 / SQRT2 };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "Y1")
        {
            QStat QMatrix = { cos(PI / 4),-sin(PI / 4),sin(PI / 4),cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "X1")
        {
            QStat QMatrix = { cos(PI / 4),-iunit * sin(PI / 4),-iunit * sin(PI / 4) ,cos(PI / 4) };
            rotateAxis(QMatrix, base.n1, base.n2);
        }
        else if (ValidQGateMatrix[0][1] == "S")
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

    m_sValidQGateMatrix = ValidQGateMatrix;
    m_sQGateMatrix = QGateMatrix;
    m_iAdjacentMatrix = vAdjacentMatrix;
}



TransformDecomposition::~TransformDecomposition()
{
}

void TransformDecomposition::matrixMultiplicationOfSingleQGate(QStat & LeftMatrix, QStat & RightMatrix)
{
    QStat QMatrix(SingleGateMatrixSize);

    QMatrix[0] = LeftMatrix[0] * RightMatrix[0] + LeftMatrix[1] * RightMatrix[2];
    QMatrix[1] = LeftMatrix[0] * RightMatrix[1] + LeftMatrix[1] * RightMatrix[3];
    QMatrix[2] = LeftMatrix[2] * RightMatrix[0] + LeftMatrix[3] * RightMatrix[2];
    QMatrix[3] = LeftMatrix[2] * RightMatrix[1] + LeftMatrix[3] * RightMatrix[3];

    RightMatrix = QMatrix;
}
void TransformDecomposition::rotateAxis(QStat & QMatrix, axis & OriginAxis, axis& NewAxis)
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
void TransformDecomposition::matrixMultiplicationOfDoubleQGate(QStat & LeftMatrix, QStat & RightMatrix)
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

void TransformDecomposition::generateMatrixOfTwoLevelSystem(QStat & NewMatrix, QStat & OldMatrix, size_t Row, size_t Column)
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

/*****************************************************************
Name        : decomposeDoubleQGate
Description : DoubleGate conversion to CU and singleGate
argin       : pNode              Target gate pointer
pParentNode        Target gate's parent node
traversalAlgorithm traversalAlgorithm pointer
argout      :
Return      :
******************************************************************/
void QPanda::decomposeDoubleQGate(AbstractQGateNode * pNode,
    QNode * pParentNode,
    TransformDecomposition * traversalAlgorithm)
{
    if (nullptr == pNode)
    {
        QCERR("pnode is null");
        throw invalid_argument("pnode is null");
    }
    QuantumGate* qGate;
    qGate = pNode->getQGate();

    if (pNode->getQuBitNum() == 1)
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

                traversalAlgorithm->generateMatrixOfTwoLevelSystem(vMatrix1, vMatrix2, i, j);

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
                traversalAlgorithm->matrixMultiplicationOfDoubleQGate(vMatrix, vMatrix1);
            }
        }
    }

    auto qCircuitDagger = qCircuit.dagger();

    traversalAlgorithm->insertQCircuit(pNode, qCircuitDagger, pParentNode);


}

double TransformDecomposition::transformMatrixToAxis(QStat &QMatrix, axis &Axis)
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

void TransformDecomposition::QGateExponentArithmetic(AbstractQGateNode * pNode, double Exponent, QStat & QMatrix)
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

void TransformDecomposition::transformAxisToMatrix(axis &Axis, double Angle, QStat &QMatrix)
{
    QMatrix.resize(SingleGateMatrixSize);

    QMatrix[0] = qcomplex_t(cos(Angle / 2), -sin(Angle / 2)*Axis.nz);
    QMatrix[1] = qcomplex_t(-sin(Angle / 2)*Axis.ny, -sin(Angle / 2)*Axis.nx);
    QMatrix[2] = qcomplex_t(sin(Angle / 2)*Axis.ny, -sin(Angle / 2)*Axis.nx);
    QMatrix[3] = qcomplex_t(cos(Angle / 2), sin(Angle / 2)*Axis.nz);
}

QCircuit TransformDecomposition::
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

QCircuit TransformDecomposition::
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

QCircuit TransformDecomposition::decomposeToffoliQGate(Qubit * TargetQubit, vector<Qubit*> ControlQubits)
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
QCircuit TransformDecomposition::decomposeTwoControlSingleQGate(AbstractQGateNode * pNode)
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
QCircuit TransformDecomposition::tempStepOfMultipleControlQGateDecomposition(vector<Qubit*> ControlQubits, vector<Qubit*> AncillaQubits)
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

/******************************************************************
Name        : decomposeMultipleControlQGate
Description : Multiple control gate conversion to quantum circuit
argin       : pNode              Target gate pointer
pParentNode        Target gate's parent node
traversalAlgorithm traversalAlgorithm pointer
argout      :
Return      :
******************************************************************/
void QPanda::decomposeMultipleControlQGate(AbstractQGateNode *pNode, QNode * pParentNode, TransformDecomposition * traversalAlgorithm)
{
    QVec vTargetQubit;

    if (pNode->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QVec vControlQubit;

    if (CIRCUIT_NODE == pParentNode->getNodeType())
    {
        AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(pParentNode);
        pQcir->getControlVector(vControlQubit);
    }

    if (pNode->getControlVector(vControlQubit) <= 0)
    {
        return;
    }

    QuantumGate* qgate = pNode->getQGate();

    QStat qMatrix;
    qgate->getMatrix(qMatrix);

    QStat vMatrix;
    QStat matrixdagger;

    traversalAlgorithm->QGateExponentArithmetic(pNode, 0.5, vMatrix);

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
        qCircuit << traversalAlgorithm->decomposeTwoControlSingleQGate(pNode);
    }
    else if (vControlQubit.size() == 3)
    {
        auto qGate = U4(vMatrix, vTargetQubit[0]);

        vector<Qubit*> vTempQubit;

        vTempQubit.push_back(vControlQubit[0]);
        vTempQubit.push_back(vControlQubit[1]);

        qGate.setControl(vTempQubit);

        qCircuit << CU(vMatrix, vControlQubit[2], vTargetQubit[0])
            << traversalAlgorithm->decomposeToffoliQGate(vControlQubit[2], { vControlQubit[0],vControlQubit[1] })
            << qGate0 << traversalAlgorithm->decomposeToffoliQGate(vControlQubit[2], { vControlQubit[0],vControlQubit[1] })
            << traversalAlgorithm->decomposeTwoControlSingleQGate(&qGate);
    }
    else if (vControlQubit.size() > 3)
    {

        Qubit* temp = vControlQubit[vControlQubit.size() - 1];

        auto qGate1 = X(temp);

        vControlQubit.pop_back();

        qGate1.setControl(vControlQubit);

        auto qCircuit1 = traversalAlgorithm->firstStepOfMultipleControlQGateDecomposition(&qGate1, vTargetQubit[0]);
        auto qCircuit2 = traversalAlgorithm->firstStepOfMultipleControlQGateDecomposition(&qGate1, vTargetQubit[0]);

        auto qGate2 = U4(vMatrix, vTargetQubit[0]);
        qGate2.setControl(vControlQubit);

        auto qCircuit3 = traversalAlgorithm->firstStepOfMultipleControlQGateDecomposition(&qGate2, temp);

        qCircuit << CU(vMatrix, vControlQubit[vControlQubit.size() - 1], vTargetQubit[0]) << qCircuit1         //CV and CC..C-NOT
            << qGate0 << qCircuit2 << qCircuit3;
    }

    traversalAlgorithm->insertQCircuit(pNode, qCircuit, pParentNode);
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
void QPanda::decomposeControlUnitarySingleQGate(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition * traversalAlgorithm)
{

    if (pNode->getQuBitNum() == 1)
    {
        return;
    }

    auto pTargetQuBit = pNode->popBackQuBit();
    auto pControlQuBit = pNode->popBackQuBit();

    pNode->PushBackQuBit(pTargetQuBit);

    vector<Qubit *> vControlQubit = { pControlQuBit };

    pNode->setControl(vControlQubit);

    auto pQGate = pNode->getQGate();

    auto qg = pNode->getQGate();

    if (nullptr == pQGate)
    {
        QCERR("pQGate is null");
        throw runtime_error("pQGate is null");
    }

    QVec qubitVector;

    if (pNode->getQuBitVector(qubitVector) <= 0)
    {
        QCERR("the size of qubit vector is error");
        throw runtime_error("the size of qubit vector is error");
    }

    auto targetQubit = qubitVector[0];

    auto pU4 = new QGATE_SPACE::U4(pQGate->getAlpha(),
        pQGate->getBeta(),
        pQGate->getGamma(),
        pQGate->getDelta());
    delete(pQGate);
    pNode->setQGate(pU4);
}

extern QCircuit QPanda::swapQGate(vector<int> ShortestWay, string MetadataQGate);
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
void QPanda::decomposeControlSingleQGateIntoMetadataDoubleQGate(AbstractQGateNode * pNode,
    QNode * pParentNode, TransformDecomposition * traversalAlgorithm)
{

    string sGateName = traversalAlgorithm->m_sValidQGateMatrix[1][0];

    if (sGateName.size() <= 0)
    {
        QCERR("the size of sGateName is error");
        throw runtime_error("the size of sGateName is error");
    }

    QVec vTargetQubit;
    if (pNode->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QVec vControlQubit;
    if (pNode->getControlVector(vControlQubit) != 1)
    {
        return;
    }

    if (CIRCUIT_NODE == pParentNode->getNodeType())
    {
        AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(pParentNode);
        pQcir->getControlVector(vControlQubit);
    }
    QuantumGate* qgate = pNode->getQGate();

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

    if (traversalAlgorithm->m_iAdjacentMatrix.size() != 0)
    {
        int iBeginNumber = (int)vControlQubit[0]->getPhysicalQubitPtr()->getQubitAddr();
        int iEndNumber = (int)vTargetQubit[0]->getPhysicalQubitPtr()->getQubitAddr();

        vector<int> viShortestConnection;

        GraphDijkstra gd(traversalAlgorithm->m_iAdjacentMatrix);

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

    traversalAlgorithm->insertQCircuit(pNode, qCircuit, pParentNode);
}

/******************************************************************
Name        : insertQCircuit
Description : insert QCircuit into parent node
quantum gate
argin       : pGateNode   The replaced target node
qCircuit    Inserted quantum circuit
pParentNode The inserted parent node
argout      : pParentNode The inserted parent node
Return      :
******************************************************************/
void TransformDecomposition::insertQCircuit(AbstractQGateNode * pGateNode, QCircuit & qCircuit, QNode * pParentNode)
{
    if ((nullptr == pParentNode) || (nullptr == pGateNode))
    {
        QCERR("param is nullptr");
        throw invalid_argument("param is nullptr");
    }

    int iNodeType = pParentNode->getNodeType();

    if (CIRCUIT_NODE == iNodeType)
    {
        auto pParentCircuit = dynamic_cast<AbstractQuantumCircuit *>(pParentNode);

        if (nullptr == pParentCircuit)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        auto aiter = pParentCircuit->getFirstNodeIter();

        if (pParentCircuit->getEndNodeIter() == aiter)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        for (; aiter != pParentCircuit->getEndNodeIter(); ++aiter)
        {
            auto temp = dynamic_cast<QNode *>(pGateNode);
            if (temp == (*aiter).get())
            {
                break;
            }
        }

        aiter = pParentCircuit->deleteQNode(aiter);

        if (nullptr == aiter.getPCur())
        {
            pParentCircuit->pushBackNode(&qCircuit);
        }
        else
        {
            pParentCircuit->insertQNode(aiter, &qCircuit);
        }

    }
    else if (PROG_NODE == iNodeType)
    {
        auto pParentQProg = dynamic_cast<AbstractQuantumProgram *>(pParentNode);

        if (nullptr == pParentQProg)
        {
            QCERR("parent node type error");
            throw invalid_argument("parent node type error");
        }

        auto aiter = pParentQProg->getFirstNodeIter();

        if (pParentQProg->getEndNodeIter() == aiter)
        {
            QCERR("unknow error");
            throw runtime_error("unknow error");
        }

        for (; aiter != pParentQProg->getEndNodeIter(); ++aiter)
        {
            auto temp = dynamic_cast<QNode *>(pGateNode);
            if (temp == (*aiter).get())
            {
                break;
            }
        }
        pParentQProg->insertQNode(aiter, &qCircuit);
        aiter = pParentQProg->deleteQNode(aiter);

    }
    else if (QIF_START_NODE == iNodeType)
    {
        auto pParentIf = dynamic_cast<AbstractControlFlowNode *>(pParentNode);

        if (nullptr == pParentIf)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        if (pGateNode == (AbstractQGateNode *)pParentIf->getTrueBranch())
        {
            pParentIf->setTrueBranch(&qCircuit);
        }
        else if (pGateNode == (AbstractQGateNode *)pParentIf->getFalseBranch())
        {
            pParentIf->setFalseBranch(&qCircuit);
        }
        else
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

    }
    else if (WHILE_START_NODE == iNodeType)
    {
        auto pParentIf = dynamic_cast<AbstractControlFlowNode *>(pParentNode);

        if (nullptr == pParentIf)
        {
            QCERR("parent if type is error");
            throw runtime_error("parent if type is error");
        }


        if (pGateNode == (AbstractQGateNode *)pParentIf->getTrueBranch())
        {
            pParentIf->setTrueBranch(&qCircuit);
        }
        else
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
    }
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

}

void TransformDecomposition::getDecompositionAngle(QStat & QMatrix, vector<double> & vdAngle)
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

    double dBeta = transformMatrixToAxis(QMatrix, TargetAxis);
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
void QPanda::decomposeUnitarySingleQGateIntoMetadataSingleQGate(AbstractQGateNode * pNode,
    QNode * pParentNode,
    TransformDecomposition * traversalAlgorithm)
{
    /*
    * Check if the quantum gate is supported
    */

    if (QGateCompare::countQGateNotSupport(pNode, traversalAlgorithm->m_sQGateMatrix) <= 0)
        return;

    QVec vTargetQubit;

    if (pNode->getQuBitVector(vTargetQubit) != 1)
    {
        return;
    }

    QuantumGate * qGate = pNode->getQGate();
    QStat QMatrix;
    qGate->getMatrix(QMatrix);

    vector<double> vdAngle;

    traversalAlgorithm->getDecompositionAngle(QMatrix, vdAngle);

    auto qCircuit = CreateEmptyCircuit();

    if (traversalAlgorithm->m_sValidQGateMatrix[0][0] == "RX")
    {
        if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RY")
        {
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << RY(vTargetQubit[0], vdAngle[1])
                << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RZ")
        {
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << RZ(vTargetQubit[0], vdAngle[1])
                << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "U1")
        {
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << U1(vTargetQubit[0], vdAngle[1]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "H")
        {
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << RX(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "Y1")
        {
            auto ygate = Y1(vTargetQubit[0]);
            ygate.setDagger(1);
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << ygate
                << RX(vTargetQubit[0], vdAngle[1]) << Y1(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "Z1")
        {
            auto zgate = Z1(vTargetQubit[0]);
            zgate.setDagger(1);
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << zgate
                << RX(vTargetQubit[0], vdAngle[1]) << Z1(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "S")
        {
            auto S_dagger = S(vTargetQubit[0]);
            S_dagger.setDagger(1);
            qCircuit << RX(vTargetQubit[0], vdAngle[2]) << S_dagger
                << RX(vTargetQubit[0], vdAngle[1]) << S(vTargetQubit[0]) << RX(vTargetQubit[0], vdAngle[0]);
        }
    }
    else if (traversalAlgorithm->m_sValidQGateMatrix[0][0] == "RY")
    {
        if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RX")
        {
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << RX(vTargetQubit[0], vdAngle[1]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RZ")
        {
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << RZ(vTargetQubit[0], vdAngle[1]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "U1")
        {
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << U1(vTargetQubit[0], vdAngle[1]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "H")
        {
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << RY(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "X1")
        {
            auto xgate = X1(vTargetQubit[0]);
            xgate.setDagger(1);
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << xgate
                << RY(vTargetQubit[0], vdAngle[1]) << X1(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "Z1")
        {
            auto zgate = Z1(vTargetQubit[0]);
            zgate.setDagger(1);
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << zgate
                << RY(vTargetQubit[0], vdAngle[1]) << Z1(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "S")
        {
            auto S_dagger = S(vTargetQubit[0]);
            S_dagger.setDagger(1);
            qCircuit << RY(vTargetQubit[0], vdAngle[2]) << S_dagger
                << RY(vTargetQubit[0], vdAngle[1]) << S(vTargetQubit[0]) << RY(vTargetQubit[0], vdAngle[0]);
        }
    }
    else if (traversalAlgorithm->m_sValidQGateMatrix[0][0] == "RZ")
    {
        if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RX")
        {
            qCircuit << RZ(vTargetQubit[0], vdAngle[2]) << RX(vTargetQubit[0], vdAngle[1]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RY")
        {
            qCircuit << RZ(vTargetQubit[0], vdAngle[2]) << RY(vTargetQubit[0], vdAngle[1]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "H")
        {
            qCircuit << RZ(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << RZ(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "X1")
        {
            auto xgate = X1(vTargetQubit[0]);
            xgate.setDagger(1);
            qCircuit << RZ(vTargetQubit[0], vdAngle[2]) << xgate
                << RZ(vTargetQubit[0], vdAngle[1]) << X1(vTargetQubit[0]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "Y1")
        {
            auto ygate = Y1(vTargetQubit[0]);
            ygate.setDagger(1);
            qCircuit << RZ(vTargetQubit[0], vdAngle[2]) << ygate
                << RZ(vTargetQubit[0], vdAngle[1]) << Y1(vTargetQubit[0]) << RZ(vTargetQubit[0], vdAngle[0]);
        }
    }
    else if (traversalAlgorithm->m_sValidQGateMatrix[0][0] == "U1")
    {
        if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RX")
        {
            qCircuit << U1(vTargetQubit[0], vdAngle[2]) << RX(vTargetQubit[0], vdAngle[1]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "RY")
        {
            qCircuit << U1(vTargetQubit[0], vdAngle[2]) << RY(vTargetQubit[0], vdAngle[1]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "H")
        {
            qCircuit << U1(vTargetQubit[0], vdAngle[2]) << H(vTargetQubit[0])
                << U1(vTargetQubit[0], vdAngle[1]) << H(vTargetQubit[0]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "X1")
        {
            auto xgate = X1(vTargetQubit[0]);
            xgate.setDagger(1);
            qCircuit << U1(vTargetQubit[0], vdAngle[2]) << xgate
                << U1(vTargetQubit[0], vdAngle[1]) << X1(vTargetQubit[0]) << U1(vTargetQubit[0], vdAngle[0]);
        }
        else if (traversalAlgorithm->m_sValidQGateMatrix[0][1] == "Y1")
        {
            auto ygate = Y1(vTargetQubit[0]);
            ygate.setDagger(1);
            qCircuit << U1(vTargetQubit[0], vdAngle[2]) << ygate
                << U1(vTargetQubit[0], vdAngle[1]) << Y1(vTargetQubit[0]) << U1(vTargetQubit[0], vdAngle[0]);
        }
    }
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    traversalAlgorithm->insertQCircuit(pNode, qCircuit, pParentNode);
}

void QPanda::deleteUnitQnode(AbstractQGateNode * pNode, QNode * pParentNode, TransformDecomposition *)
{

    auto qGate = pNode->getQGate();

    QStat qMatrix;
    qGate->getMatrix(qMatrix);

    if (qMatrix.size() == SingleGateMatrixSize
        && abs(qMatrix[0] - qMatrix[3]) < ZeroJudgement
        && abs(abs(qMatrix[0]) - 1) < ZeroJudgement)
    {
        if (CIRCUIT_NODE == pParentNode->getNodeType())
        {
            auto pQCircuitNode = dynamic_cast<AbstractQuantumCircuit *>(pParentNode);
            if (pQCircuitNode != nullptr)
            {
                auto aiter = pQCircuitNode->getFirstNodeIter();
                for (; aiter != pQCircuitNode->getEndNodeIter(); ++aiter)
                {
                    auto temp = dynamic_cast<QNode *>(pNode);
                    if (temp == (*aiter).get())
                    {
                        break;
                    }
                }
                aiter = pQCircuitNode->deleteQNode(aiter);
            }

        }
        else if (PROG_NODE == pParentNode->getNodeType())
        {
            auto pQProgNode = dynamic_cast<AbstractQuantumProgram *>(pNode);
            auto aiter = pQProgNode->getFirstNodeIter();
            for (; aiter != pQProgNode->getEndNodeIter(); ++aiter)
            {
                auto temp = dynamic_cast<QNode *>(pNode);
                if (temp == (*aiter).get())
                {
                    break;
                }
            }
            aiter = pQProgNode->deleteQNode(aiter);
        }
    }

}

QCircuit QPanda::swapQGate(vector<int> ShortestWay, string MetadataQGate)
{
    auto qCircuit = CreateEmptyCircuit();

    Qubit *qTemp1 = nullptr;
    Qubit *qTemp2 = nullptr;

    if (MetadataQGate == "CNOT")
    {
        for (auto iter = ShortestWay.begin(); iter != ShortestWay.end() - 2; iter++)
        {
            if (qAlloc(*iter) != nullptr && qAlloc(*(iter + 1)) != nullptr)
            {
                qTemp1 = qAlloc(*iter);
                qTemp2 = qAlloc(*(iter + 1));

            }
            else
            {
                QCERR("Unknown internal error");
                throw runtime_error("Unknown internal error");
            }
            qCircuit << CNOT(qTemp1, qTemp2) << CNOT(qTemp2, qTemp1) << CNOT(qTemp1, qTemp2);
        }
    }
    else if (MetadataQGate == "CZ")
    {
        for (auto iter = ShortestWay.begin(); iter != ShortestWay.end() - 2; iter++)
        {
            if (qAlloc(*iter) != nullptr && qAlloc(*(iter + 1)) != nullptr)
            {
                qTemp1 = qAlloc(*iter);
                qTemp2 = qAlloc(*(iter + 1));

            }
            else
            {
                QCERR("Unknown internal error");
                throw runtime_error("Unknown internal error");
            }
            qCircuit << H(qTemp1) << CZ(qTemp1, qTemp2) << H(qTemp1)
                << H(qTemp2) << CZ(qTemp2, qTemp1) << H(qTemp2)
                << H(qTemp1) << CZ(qTemp1, qTemp2) << H(qTemp1);
        }
    }
    else if (MetadataQGate == "ISWAP")
    {
        for (auto iter = ShortestWay.begin(); iter != ShortestWay.end() - 2; iter++)
        {
            if (qAlloc(*iter) != nullptr && qAlloc(*(iter + 1)) != nullptr)
            {
                qTemp1 = qAlloc(*iter);
                qTemp2 = qAlloc(*(iter + 1));

            }
            auto qgate = iSWAP(qTemp1, qTemp2);
            qgate.setDagger(true);
            qCircuit << RZ(qTemp2, PI / 2) << qgate << RX(qTemp1, -PI / 2) << qgate << RZ(qTemp1, -PI / 2)
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
    return qCircuit;
}

/******************************************************************
Name        : checkControlFlowBranch
Description : Check if the branch is a QGate type
argin       : pNode Target Node pointer
argout      :
Return      :
******************************************************************/
void TransformDecomposition::checkControlFlowBranch(QNode * pNode)
{
    int iChildNodeType = -1;

    if (nullptr != pNode)
    {
        iChildNodeType = pNode->getNodeType();

        if (GATE_NODE != iChildNodeType)
        {
            mergeSingleGate(pNode);
        }
    }
}

/******************************************************************
Name        : mergeSingleGate
Description : merge single gate
argin       : pNode  Target node pointer
argout      :
Return      :
******************************************************************/
void TransformDecomposition::mergeSingleGate(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is a nullptr");
        throw invalid_argument("pNode is a nullptr");
    }

    int iNodeType = pNode->getNodeType();

    /*
    * Check node type
    */
    if (QIF_START_NODE == iNodeType)
    {
        auto pIfNode = dynamic_cast<AbstractControlFlowNode *>(pNode);
        mergeControlFlowSingleGate(pIfNode, iNodeType);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        auto pWhileNode = dynamic_cast<AbstractControlFlowNode *>(pNode);
        mergeControlFlowSingleGate(pWhileNode, iNodeType);
    }
    else if (PROG_NODE == iNodeType)
    {
        auto pProg = dynamic_cast<AbstractQuantumProgram *>(pNode);
        mergeCircuitandProgSingleGate(pProg);
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        auto pQCircuit = dynamic_cast<AbstractQuantumCircuit *>(pNode);
        mergeCircuitandProgSingleGate(pQCircuit);
    }
    else
    {
        return;
    }
}

/******************************************************************
Name        : mergeControlFlowSingleGate
Description : Merge control flow single gate
argin       : pNode      Target Node pointer
iNodeType  Node Type
argout      :
Return      :
******************************************************************/
void TransformDecomposition::mergeControlFlowSingleGate(AbstractControlFlowNode * pNode, int iNodeType)
{
    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    if (QIF_START_NODE == iNodeType)
    {
        QNode *pChildNode = pNode->getTrueBranch();
        checkControlFlowBranch(pChildNode);

        pChildNode = pNode->getFalseBranch();
        checkControlFlowBranch(pChildNode);
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        QNode *pChildNode = pNode->getTrueBranch();
        checkControlFlowBranch(pChildNode);
    }
}

void TransformDecomposition::cancelControlQubitVector(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("pNode is nullptr");
        throw invalid_argument("pNode is nullptr");
    }

    int iNodeType = pNode->getNodeType();

    if (PROG_NODE == iNodeType)
    {
        auto pProg = dynamic_cast<AbstractQuantumProgram *>(pNode);
        if (nullptr == pProg)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        for (auto aiter = pProg->getFirstNodeIter(); aiter != pProg->getEndNodeIter(); aiter++)
        {
            cancelControlQubitVector((*aiter).get());
        }
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        auto pQCircuit = dynamic_cast<AbstractQuantumCircuit *>(pNode);
        if (nullptr == pQCircuit)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        pQCircuit->clearControl();

        for (auto aiter = pQCircuit->getFirstNodeIter(); aiter != pQCircuit->getEndNodeIter(); aiter++)
        {
            cancelControlQubitVector((*aiter).get());
        }
    }
    else if (QIF_START_NODE == iNodeType)
    {
        auto pIf = dynamic_cast<AbstractControlFlowNode *>(pNode);
        if (nullptr == pIf)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        cancelControlQubitVector(pIf->getTrueBranch());
        if (nullptr != pIf->getFalseBranch())
            cancelControlQubitVector(pIf->getFalseBranch());
    }
    else if (WHILE_START_NODE == iNodeType)
    {
        auto pWhile = dynamic_cast<AbstractControlFlowNode *>(pNode);
        if (nullptr == pWhile)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        cancelControlQubitVector(pWhile->getTrueBranch());
    }
    else
    { }

    return;
}

/******************************************************************
Name        : Traversal
Description : Traversal AbstractControlFlowNode's child Node
argin       : pNode      Target Node pointer
function   Optimization of quantum program algorithm
argout      : pNode      Target Node pointer
Return      :
******************************************************************/
void TransformDecomposition::Traversal(AbstractControlFlowNode * pControlNode, TraversalDecompositionFunction function, int iType)
{
    if (nullptr == pControlNode)
    {
        QCERR("pControlNode is nullptr");
        throw invalid_argument("pControlNode is nullptr");
    }

    auto pNode = dynamic_cast<QNode *>(pControlNode);

    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    auto iNodeType = pNode->getNodeType();
    if (WHILE_START_NODE == iNodeType)
    {
        TraversalByType(pControlNode->getTrueBranch(), pNode, function, iType);
    }
    else if (QIF_START_NODE == iNodeType)
    {
        TraversalByType(pControlNode->getTrueBranch(), pNode, function, iType);
        auto pFalseBranchNode = pControlNode->getFalseBranch();

        if (nullptr != pFalseBranchNode)
        {
            TraversalByType(pControlNode->getFalseBranch(), pNode, function, iType);
        }
    }
}

/******************************************************************
Name        : Traversal
Description : Traversal AbstractQuantumCircuit's child Node
argin       : pNode      Target Node pointer
function   Optimization of quantum program algorithm
argout      : pNode      Target Node pointer
Return      :
******************************************************************/
void TransformDecomposition::Traversal(AbstractQuantumCircuit * pQCircuit, TraversalDecompositionFunction function, int iType)
{
    if (nullptr == pQCircuit)
    {
        QCERR("pQCircuit is nullptr");
        throw invalid_argument("pQCircuit is nullptr");
    }

    auto aiter = pQCircuit->getFirstNodeIter();

    if (aiter == pQCircuit->getEndNodeIter())
        return;

    auto pNode = dynamic_cast<QNode *>(pQCircuit);

    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    /*
    * Traversal QCircuit's child node
    */
    while (aiter != pQCircuit->getEndNodeIter())
    {
        auto next = aiter.getNextIter();
        TraversalByType((*aiter).get(), pNode, function, iType);
        aiter = next;
    }
}

/******************************************************************
Name        : Traversal
Description : Traversal AbstractQuantumProgram's child Node
argin       : pNode      Target Node pointer
function   Optimization of quantum program algorithm
argout      : pNode      Target Node pointer
Return      :
******************************************************************/
inline void TransformDecomposition::Traversal(AbstractQuantumProgram * pProg, TraversalDecompositionFunction function, int iType)
{
    if (nullptr == pProg)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    auto aiter = pProg->getFirstNodeIter();

    if (aiter == pProg->getEndNodeIter())
        return;
    auto pNode = dynamic_cast<QNode *>(pProg);

    if (nullptr == pNode)
    {
        QCERR("pNode is nullptr");
        throw invalid_argument("pNode is nullptr");
    }

    while (aiter != pProg->getEndNodeIter())
    {
        auto next = aiter.getNextIter();
        TraversalByType((*aiter).get(), pNode, function, iType);
        aiter = next;
    }
}

/******************************************************************
Name        : TraversalByType
Description : Traversal by type
argin       : pNode      Target Node pointer
ParentNode Target parent Node
function   Optimization of quantum program algorithm
argout      : ParentNode Target parent Node
Return      :
******************************************************************/
void TransformDecomposition::
TraversalByType(QNode * pNode, QNode * ParentNode, TraversalDecompositionFunction function, int iType)
{
    int iNodeType = pNode->getNodeType();

    if (NODE_UNDEFINED == iNodeType)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }


    /*
    * Check node type
    */
    if (GATE_NODE == iNodeType)
    {
        auto pGateNode = dynamic_cast<AbstractQGateNode *>(pNode);

        if (nullptr == pGateNode)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        function(pGateNode, ParentNode, this);            /* Call optimization algorithm */
    }
    else if (CIRCUIT_NODE == iNodeType)
    {
        auto pQCircuitNode = dynamic_cast<AbstractQuantumCircuit *>(pNode);

        if (nullptr == pQCircuitNode)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }

        if (3 == iType)
        {
            if (CIRCUIT_NODE == ParentNode->getNodeType())
            {
                AbstractQuantumCircuit *pQcir = dynamic_cast<AbstractQuantumCircuit*>(ParentNode);
                QVec vControlQubit;
                pQcir->getControlVector(vControlQubit);

                pQCircuitNode->setControl(vControlQubit);
            }
        }
        Traversal(pQCircuitNode, function, iType);
    }
    else if (PROG_NODE == iNodeType)
    {
        auto pQProgNode = dynamic_cast<AbstractQuantumProgram *>(pNode);

        if (nullptr == pQProgNode)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        Traversal(pQProgNode, function, iType);
    }
    else if ((WHILE_START_NODE == iNodeType) || (QIF_START_NODE == iNodeType))
    {
        auto pControlFlowNode = dynamic_cast<AbstractControlFlowNode *>(pNode);

        if (nullptr == pControlFlowNode)
        {
            QCERR("Unknown internal error");
            throw runtime_error("Unknown internal error");
        }
        Traversal(pControlFlowNode, function, iType);
    }
    else if (MEASURE_GATE == iNodeType)
    {
        return;
    }
    else
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
}

/******************************************************************
Name        : TraversalOptimizationMerge
Description : Traversal optimization merge algorithm
argin       : pNode    Target Node pointer
argout      : pNode    Target Node pointer
Return      :
******************************************************************/
void TransformDecomposition::TraversalOptimizationMerge(QNode * pNode)
{
    if (nullptr == pNode)
    {
        QCERR("this param is not QNode");
        throw invalid_argument("this param is not QNode");
    }

    int iNodeType = pNode->getNodeType();

    if ((GATE_NODE == iNodeType) || (MEASURE_GATE == iNodeType))
    {
        QCERR("the param cannot be a QGate or Measure");
        throw invalid_argument("the param cannot be a QGate or Measure");
    }

    /*
    * Begin Traversal optimization merge algorithm
    */
    TraversalByType(pNode, nullptr, &decomposeDoubleQGate, 1);
    TraversalByType(pNode, nullptr, &decomposeControlUnitarySingleQGate, 2);
    TraversalByType(pNode, nullptr, &decomposeMultipleControlQGate, 3);
    TraversalByType(pNode, nullptr, &decomposeControlUnitarySingleQGate, 2);
    TraversalByType(pNode, nullptr, &decomposeControlSingleQGateIntoMetadataDoubleQGate, 4);
    //mergeSingleGate(pNode);
    TraversalByType(pNode, nullptr, &decomposeUnitarySingleQGateIntoMetadataSingleQGate, 5);
    cancelControlQubitVector(pNode);
    TraversalByType(pNode, nullptr, deleteUnitQnode, 6);
}
