#ifndef PSI4WRAPPER_H
#define PSI4WRAPPER_H

#include <string>
#include <vector>

#include "QAlg/DataStruct.h"
#include "QAlg/Components/Operator/FermionOperator.h"

namespace QPanda {

class DLLEXPORT Psi4Wrapper
{
public:
    Psi4Wrapper();

    void setMolecule(const std::string &molecule)
    {
        m_molecule = molecule;
    }

    std::string getMolecule()
    {
        return m_molecule;
    }

    void setMultiplicity(int multiplicity)
    {
        m_multiplicity = multiplicity;
    }

    int getMultiplicity()
    {
        return m_multiplicity;
    }

    void setCharge(int charge)
    {
        m_charge = charge;
    }

    int getCharge()
    {
        return m_charge;
    }

    void setBasis(const std::string basis)
    {
        m_basis = basis;
    }

    std::string getBasis()
    {
        return m_basis;
    }

    void setEqTolerance(const double val)
    {
        m_eq_tolerance = val;
    }

    double getEqTolerance()
    {
        return m_eq_tolerance;
    }

    std::string getLastError()
    {
        return m_last_error;
    }

    std::string getData()
    {
        return m_data;
    }

    void initialize(const std::string& dir);
    bool run();
    void finalize();
//private:
//    FermionOperator parsePsi4DataToFermion(const std::string &data);
private:
    std::string m_molecule;
    int m_multiplicity;
    int m_charge;
    std::string m_basis;
    double m_eq_tolerance;

    std::string m_last_error;
    std::string m_data;
};

}

#endif // PSI4WRAPPER_H
