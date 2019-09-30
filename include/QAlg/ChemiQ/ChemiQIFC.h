#pragma once

#include "QAlg/ChemiQ/ChemiQ.h"

using namespace QPanda;

extern "C" {
    DLLEXPORT ChemiQ* initialize(char* dir);
    DLLEXPORT void finalize(ChemiQ* chemiq);
    DLLEXPORT void setMolecule(ChemiQ* chemiq, char* molecule);
    DLLEXPORT void setMolecules(ChemiQ* chemiq, char* molecules);
    DLLEXPORT void setMultiplicity(ChemiQ* chemiq, int multiplicity);
    DLLEXPORT void setCharge(ChemiQ* chemiq, int charge);
    DLLEXPORT void setBasis(ChemiQ* chemiq, char* basis);
    DLLEXPORT void setTransformType(ChemiQ* chemiq, int type);
    DLLEXPORT void setUccType(ChemiQ* chemiq, int type);
    DLLEXPORT void setOptimizerType(ChemiQ* chemiq, int type);
    DLLEXPORT void setOptimizerIterNum(ChemiQ* chemiq, int value);
    DLLEXPORT void setOptimizerFuncCallNum(ChemiQ* chemiq, int value);
    DLLEXPORT void setOptimizerXatol(ChemiQ* chemiq, double value);
    DLLEXPORT void setOptimizerFatol(ChemiQ* chemiq, double value);
    DLLEXPORT void setLearningRate(ChemiQ* chemiq, double value);
    DLLEXPORT void setEvolutionTime(ChemiQ* chemiq, double value);
    DLLEXPORT void setHamiltonianSimulationSlices(ChemiQ* chemiq, int value);
    DLLEXPORT void setSaveDataDir(ChemiQ* chemiq, char* dir);
    DLLEXPORT bool exec(ChemiQ* chemiq);
}
