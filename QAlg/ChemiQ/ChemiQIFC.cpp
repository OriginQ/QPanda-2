#include "QAlg/ChemiQ/ChemiQIFC.h"
#include "Core/Utilities/Tools/QString.h"

DLLEXPORT ChemiQ* initialize(char* dir)
{
    ChemiQ* chemiq = new ChemiQ();
    chemiq->initialize(dir);
    return chemiq;
}
DLLEXPORT void finalize(ChemiQ* chemiq)
{
    if (chemiq != NULL)
    {
        std::cout << "deleting..." << std::endl;
        chemiq->finalize();
        delete chemiq;
        chemiq = NULL;
    }

    std::cout << "delete success" << std::endl;
}

DLLEXPORT void setMolecule(ChemiQ* chemiq, char* molecule)
{
    chemiq->setMolecule(molecule);
}

DLLEXPORT void setMolecules(ChemiQ* chemiq, char* molecules)
{
    QString str(molecules);
    auto str_list = str.split(";");
    std::vector<std::string> vec;

    for (auto& i : str_list)
    {
        vec.push_back(i.data());
    }


    chemiq->setMolecules(vec);
}

DLLEXPORT void setMultiplicity(ChemiQ* chemiq, int multiplicity)
{
    chemiq->setMultiplicity(multiplicity);
}

DLLEXPORT void setCharge(ChemiQ* chemiq, int charge)
{
    chemiq->setCharge(charge);
}

DLLEXPORT void setBasis(ChemiQ* chemiq, char* basis)
{
    chemiq->setBasis(basis);
}

DLLEXPORT void setEqTolerance(ChemiQ* chemiq, double val)
{
    chemiq->setEqTolerance(val);
}

DLLEXPORT void setTransformType(ChemiQ* chemiq, int type)
{
    chemiq->setTransformType(TransFormType(type));
}

DLLEXPORT void setUccType(ChemiQ* chemiq, int type)
{
    chemiq->setUccType(UccType(type));
}

DLLEXPORT void setOptimizerType(ChemiQ* chemiq, int type)
{
    chemiq->setOptimizerType(OptimizerType(type));
}

DLLEXPORT void setOptimizerIterNum(ChemiQ* chemiq, int value)
{
    chemiq->setOptimizerIterNum(value);
}

DLLEXPORT void setOptimizerFuncCallNum(ChemiQ* chemiq, int value)
{
    chemiq->setOptimizerFuncCallNum(value);
}

DLLEXPORT void setOptimizerXatol(ChemiQ* chemiq, double value)
{
    chemiq->setOptimizerXatol(value);
}

DLLEXPORT void setOptimizerFatol(ChemiQ* chemiq, double value)
{
    chemiq->setOptimizerFatol(value);
}

DLLEXPORT void setLearningRate(ChemiQ* chemiq, double value)
{
    chemiq->setLearningRate(value);
}

DLLEXPORT void setEvolutionTime(ChemiQ* chemiq, double value)
{
    chemiq->setEvolutionTime(value);
}

DLLEXPORT void setHamiltonianSimulationSlices(ChemiQ* chemiq, int value)
{
    chemiq->setHamiltonianSimulationSlices(value);
}

DLLEXPORT void setSaveDataDir(ChemiQ* chemiq, char* dir)
{
    chemiq->setSaveDataDir(dir);
}

DLLEXPORT int getQubitsNum(ChemiQ* chemiq)
{
    return chemiq->getQubitsNum();
}

DLLEXPORT bool exec(ChemiQ* chemiq)
{
    return chemiq->exec();
}

DLLEXPORT const char* getLastError(ChemiQ* chemiq)
{
    return chemiq->getLastError().c_str();
}
