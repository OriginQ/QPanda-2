#ifndef PSI4WRAPPER_H
#define PSI4WRAPPER_H

#include <string>
#include <vector>

#include "Components/DataStruct.h"
#include "Components/Operator/FermionOperator.h"

namespace QPanda {

/**
* @brief wrapper class for Psi4.
* @ingroup ChemiQ
*/
class DLLEXPORT Psi4Wrapper
{
public:
	/**
	* @brief Construct of Psi4Wrapper
	*/
    Psi4Wrapper();

	/**
    * @brief set molecule
    * @param[in] string& the name of molecule
    */
    void setMolecule(const std::string &molecule)
    {
        m_molecule = molecule;
    }

	/**
	* @brief get molecule
	* @return the name string of the molecule
	*/
    std::string getMolecule()
    {
        return m_molecule;
    }

	/**
	* @brief set multiplicity
	* @param[in] int multiplicity val
	*/
    void setMultiplicity(int multiplicity)
    {
        m_multiplicity = multiplicity;
    }

	/**
	* @brief get multiplicity
	* @return return the val of the multiplicity
	*/
    int getMultiplicity()
    {
        return m_multiplicity;
    }

	/**
	* @brief set charge
	* @param[in] int charge val
	*/
    void setCharge(int charge)
    {
        m_charge = charge;
    }

	/**
	* @brief get charge
	* @return return the val of charge
	*/
    int getCharge()
    {
        return m_charge;
    }

	/**
	* @brief set Basis
	* @param[in] std::string the string of Basis
	*/
    void setBasis(const std::string basis)
    {
        m_basis = basis;
    }

	/**
	* @brief get Basis
	* @return return the val of Basis
	*/
    std::string getBasis()
    {
        return m_basis;
    }

	/**
	* @brief set Eq Tolerance
	* @param[in] double the val of Tolerance
	*/
    void setEqTolerance(const double val)
    {
        m_eq_tolerance = val;
    }

	/**
	* @brief get Eq Tolerance
	* @return return the val of Tolerance
	*/
    double getEqTolerance()
    {
        return m_eq_tolerance;
    }

	/**
	* @brief get last error string
	* @return return the last error string
	*/
    std::string getLastError()
    {
        return m_last_error;
    }

	/**
	* @brief get the data
	* @return return the data string
	*/
    std::string getData()
    {
        return m_data;
    }

	/**
	* @brief Initialize
	* @param[in] std::string& the dir of chemiq
	*/
    void initialize(const std::string& dir);

	/**
	* @brief run Psi4
	*/
    bool run();

	/**
	* @brief release resource
	*/
    void finalize();

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
