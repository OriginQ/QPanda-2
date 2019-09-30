#include "QAlg/ChemiQ/ChemiQ.h"
#include <string.h>
#include <iostream>

using namespace std;
USING_QPANDA

int main(int argc, char* argv[])
{
    vector_s molecules = { "H 0 0 0\nH 0 0 0.74" };
    int charge = 0;
    int iters = 1000;
    int multiplicity = 1;
    char* filename = "output.out";
    char* datadir = "./data/";
    char* psi4dir = "D:/ChemiQ_release/";

    if (argc == 2 && !strcmp(argv[1], "--help")) {
        cout << "--molecule " << "H 0 0 0\\nH 0 0 0.74 (necessary)" << endl;
        cout << "--charge 1 (default = 0)" << endl;
        cout << "--iters 1000 (default = 1000)" << endl;
        cout << "--file output.out (default = output.out)" << endl;
        cout << "--datadir ./data/ (default = ./data/)" << endl;
        cout << "--psi4dir D:/ChemiQ_release/ (default = D:/ChemiQ_release/)" << endl;
        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << endl;
    }
    QPanda::UccType ucctype = UccType::UCCS;
    OptimizerType optimizer_type = OptimizerType::NELDER_MEAD;
    TransFormType transform_type = TransFormType::Jordan_Wigner;
    char* basis = "sto-3g";
    bool restore = false;
    QMachineType machine_type = QMachineType::CPU_SINGLE_THREAD;
    for (int i = 1; i < argc;) {
        if (!strcmp(argv[i], "--molecule")) {
            molecules.clear();
            molecules.push_back(argv[i + 1]);
            cout << argv[i + 1] << endl;
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--charge")) {
            charge = atoi(argv[i + 1]);
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--iters")) {
            iters = atoi(argv[i + 1]);
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--multiplicity")) {
            multiplicity = atoi(argv[i + 1]);
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--filename")) {
            filename = (argv[i + 1]);
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--datadir")) {
            datadir = (argv[i + 1]);
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--UCC")) {
            if (strcmp("UCCSD", argv[i + 1])) {
                ucctype = UccType::UCCSD;
            }
            else if (!strcmp("UCCS", argv[i + 1])) {
                ucctype = UccType::UCCS;
            }
            i = i + 2; continue;
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
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--transform")) {
            if (!strcmp("JW", argv[i + 1])) {
                transform_type = TransFormType::Jordan_Wigner;
            }
            else if (!strcmp("BK", argv[i + 1])) {
                transform_type = TransFormType::Bravyi_Ktaev;
            }
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--basis")) {
            basis = (argv[i + 1]);
            i = i + 2; continue;
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
            i = i + 2; continue;
        }
        else if (!strcmp(argv[i], "--psi4dir"))
        {
            psi4dir = argv[i + 1];
            i = i + 2; continue;
        }
    }

    /*for (int i = 0; i < 50; i++)
    {
        std::string str = "H 0 0 0\nH 0 0 ";
        str = str + std::to_string(0.2 + 0.05 * i);
        molecules.push_back(str);
    }*/
    ChemiQ chemiq;
    //chemiq.initialize("C:/Users/Agony/Desktop/ChemiQ");
    chemiq.initialize(psi4dir);
    //chemiq.setMoleculer(vector_s({"H 0 0 0","H 0 0 0.74"}));
    chemiq.setMolecules(molecules);
    chemiq.setBasis(basis);
    chemiq.setMultiplicity(multiplicity);
    chemiq.setCharge(charge);
    chemiq.setTransformType(transform_type);
    chemiq.setUccType(ucctype);
    chemiq.setOptimizerType(optimizer_type);
    //chemiq.setOptimizerType(OptimizerType::GRADIENT);
    chemiq.setOptimizerIterNum(iters);
    chemiq.setOptimizerFuncCallNum(iters);
    chemiq.setSaveDataDir(datadir);
    chemiq.setQuantumMachineType(machine_type);

    if (!chemiq.exec())
    {
        std::cout << chemiq.getLastError() << std::endl;
    }
    else
    {
        auto energies = chemiq.getEnergies();
        //std::cout << chemiq.getEnergy() << std::endl;
        ofstream out(filename, ios::app);

        out << "molecule:" << molecules[0] << endl;
        out << "charge:" << charge << endl;
        out << "iters:" << iters << endl;
        out << "multiplicity" << multiplicity << endl;
        out << "energy" << energies[0] << endl;
        out.close();
    }
    chemiq.finalize();

    return 0;
}
