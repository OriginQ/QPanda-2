#include "QAlg/ChemiQ/ChemiQ.h"
#include <string.h>
#include <iostream>

using namespace std;
USING_QPANDA

int main(int argc, char* argv[])
{
    std::string molecules = "H 0 0 0,H 0 0 0.7";
    int charge = 0;
    int multiplicity = 1;
    UccType ucctype = UccType::UCCSD;
    TransFormType transform_type = TransFormType::Jordan_Wigner;
    char* basis = "sto-3g";
    double eq_tolerance = 1e-8;
    char* datadir = "./data/";
    char* psi4dir = "D:/ChemiQ/ChemiQ/";
    OptimizerType optimizer_type = OptimizerType::NELDER_MEAD;
    int iters = 1000;
    int fcalls = 0;
    double xatol = 1e-4;
    double fatol = 1e-4;
    double evolution_time = 1.0;
    int slices = 1;
    double learning_rate = 0.2;
    bool get_qubits = false;

    
    QMachineType machine_type = QMachineType::CPU_SINGLE_THREAD;

    if (argc == 2 && !strcmp(argv[1], "--help")) {
        cout << "--molecule " << "H 0 0 0,H 0 0 0.74;H 0 0 0, H 0 0 0.85 (default = H 0 0 0,H 0 0 0.74)" << endl;
        cout << "--charge 1 (default = 0)" << endl;
        cout << "--multiplicity 1 (default = 1)" << endl;
        cout << "--basis sto-3g (default = sto-3g)" << endl;
        cout << "--EQTolerance 1e-8 (default = 1e-8)" << endl;
        cout << "--UCC UCCS/UCCSD (default = UCCS)" << endl;
        cout << "--transform JW/Parity/BK (default = JW)" << endl;
        cout << "--optimizer NELDER_MEAD/POWELL/GRADIENT (default = NELDER_MEAD)" << endl;
        cout << "--iters 1000 (default = 1000)" << endl;
        cout << "--fcalls 1000 (default = iters)" << endl;
        cout << "--xatol 1e-4 (default = 1e-4)" << endl;
        cout << "--fatol 1e-4 (default = 1e-4)" << endl;
        cout << "--evolution_time 1.0 (default = 1.0)" << endl;
        cout << "--hamiltonian_simulation_slices 3 (default = 3)" << endl;
        cout << "--learning_rate 0.2 (default = 0.2)" << endl;
        cout << "--datadir ./data/ (default = ./data/)" << endl;
        cout << "--backend CPU_SINGLE_THREAD/CPU/GPU/NOISE (default CPU_SINGLE_THREAD)" << endl;
        cout << "--psi4dir \"\" (default = \"\")" << endl;
        cout << "--getQubits" << endl;
        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << endl;
    }
    
    for (int i = 1; i < argc;) {
        if (!strcmp(argv[i], "--molecule")) {
            molecules = argv[i+1];
            cout << argv[i + 1] << endl;
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--charge")) {
            charge = atoi(argv[i + 1]);
            i = i + 2; 
        } 
        else if (!strcmp(argv[i], "--multiplicity")) {
            multiplicity = atoi(argv[i + 1]);
            i = i + 2; 
        }    
        else if (!strcmp(argv[i], "--basis")) {
            basis = (argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--EQTolerance")) {
            eq_tolerance = atof(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--UCC")) {
            if (strcmp("UCCSD", argv[i + 1])) {
                ucctype = UccType::UCCSD;
            }
            else if (!strcmp("UCCS", argv[i + 1])) {
                ucctype = UccType::UCCS;
            }
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--transform")) {
            if (!strcmp("JW", argv[i + 1])) {
                transform_type = TransFormType::Jordan_Wigner;
            }
            else if (!strcmp("Parity", argv[i + 1])) {
                transform_type = TransFormType::Parity;
            }
            else if (!strcmp("BK", argv[i + 1])) {
                transform_type = TransFormType::Bravyi_Ktaev;
            }
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--optimizer")) {
            if (!strcmp("NELDER_MEAD", argv[i + 1])) {
                optimizer_type = OptimizerType::NELDER_MEAD;
            }
            else if (!strcmp("POWELL", argv[i + 1])) {
                optimizer_type = OptimizerType::POWELL;
            }
            else if (!strcmp("GRADIENT", argv[i + 1])) {
                optimizer_type = OptimizerType::GRADIENT;
            }
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--iters")) {
            iters = atoi(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--fcalls")) {
            fcalls = atoi(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--xatol")) {
            xatol = atof(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--fatol")) {
            fatol = atof(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--evolution_time")) {
            evolution_time = atof(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--hamiltonian_simulation_slices")) {
            slices = atoi(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--learning_rate")) {
            learning_rate = atof(argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--datadir")) {
            datadir = (argv[i + 1]);
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--backend")) {
            if (!strcmp("CPU_SINGLE_THREAD", argv[i + 1])) {
                machine_type = QMachineType::CPU_SINGLE_THREAD;
            }
            else if (!strcmp("CPU", argv[i + 1])) {
                machine_type = QMachineType::CPU;
            }
            else if (!strcmp("GPU", argv[i + 1])) {
                machine_type = QMachineType::GPU;
            }
            else if (!strcmp("NOISE", argv[i + 1])) {
                machine_type = QMachineType::NOISE;
            }
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--psi4dir")){
            psi4dir = argv[i + 1];
            i = i + 2; 
        }
        else if (!strcmp(argv[i], "--getQubits")){
            get_qubits = true;
            i = i + 1;
        }
    }

    if (fcalls == 0)
    {
        fcalls = iters;
    }

    ChemiQ chemiq;
    chemiq.initialize(psi4dir);
    chemiq.setMoleculesStr(molecules);
    chemiq.setCharge(charge);
    chemiq.setMultiplicity(multiplicity);
    chemiq.setBasis(basis);
    chemiq.setEqTolerance(eq_tolerance);
    chemiq.setUccType(ucctype);
    chemiq.setTransformType(transform_type);
    chemiq.setOptimizerType(optimizer_type);
    chemiq.setOptimizerIterNum(iters);
    chemiq.setOptimizerFuncCallNum(fcalls);
    chemiq.setOptimizerXatol(xatol);
    chemiq.setOptimizerFatol(fatol);
    chemiq.setEvolutionTime(evolution_time);
    chemiq.setHamiltonianSimulationSlices(slices);
    chemiq.setLearningRate(learning_rate);
    chemiq.setSaveDataDir(datadir);
    chemiq.setQuantumMachineType(machine_type);
    //chemiq.setToGetHamiltonianFromFile(true);
    //chemiq.setHamiltonianGenerationOnly(true);

    if (get_qubits)
    {
        int ret = chemiq.getQubitsNum();
        if (ret < 0)
        {
            std::cout << chemiq.getLastError() << std::endl;
        }
        std::cout <<"QubitsNum : "<< ret << std::endl;
        chemiq.finalize();

        return ret;
    }
    else
    {
        if (!chemiq.exec())
        {
            std::cout << chemiq.getLastError() << std::endl;
        }

        auto energies = chemiq.getEnergies();
        for (auto& i : energies)
        {
            printf("%0.18lf\n", i);
        }

        chemiq.finalize();

        return 0;
    }
}
