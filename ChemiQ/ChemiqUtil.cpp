#include "ChemiqUtil.h"
#include "complex.h"

QPANDA_BEGIN
size_t getElectronNum(const std::string &atom)
{
    auto iter = g_kAtomElectrons.find(atom);
    if (iter != g_kAtomElectrons.end())
    {
        return iter->second;
    }
    else
    {
        QCERR(atom + " is not in the first 18 elements of the periodic table ");
    }

    return 0;
}

PauliOperator JordanWignerTransform(const OrbitalActVec &fermion_item)
{
    PauliOperator pauli({"", 1});

    for (auto &j : fermion_item)
    {
        auto op_qubit = j.first;
        std::string op_str;
        for (size_t i = 0; i < op_qubit; i++)
        {
            op_str += "Z" + std::to_string(i) + " ";
        }

        std::string op_str1 = op_str + "X" + std::to_string(op_qubit);
        std::string op_str2 = op_str + "Y" + std::to_string(op_qubit);

		QPauliMap map;
        map.insert(std::make_pair(op_str1, 0.5));
        if (j.second)
        {
            map.insert(std::make_pair(op_str2, complex_d(0,-0.5)));
        }
        else
        {
            map.insert(std::make_pair(op_str2, complex_d(0, 0.5)));
        }

        pauli *= PauliOperator(map);
    }

    return pauli;
}

PauliOperator JordanWignerTransform(const FermionOperator &fermion)
{
    auto data = fermion.data();
    PauliOperator pauli;
    for (auto &i : data)
    {
        pauli += JordanWignerTransform(i.first.first)*i.second;
    }

    return pauli;
}

VarPauliOperator JordanWignerTransform(const VarFermionOperator &fermion)
{
    auto data = fermion.data();
    VarPauliOperator pauli;
    for (auto &i : data)
    {
        PauliOperator one_pauli = JordanWignerTransform(i.first.first);
        for (auto &j : one_pauli.data())
        {
            pauli += VarPauliOperator(j.first.second,
                complex_var(i.second.real() * j.second.real()- i.second.imag()*j.second.imag(),
                    i.second.real()*j.second.imag() + i.second.imag() * j.second.real()));
        }
    }

    return pauli;
}


PauliOperator ParityTransform(const OrbitalActVec &fermion_item, size_t maxqubit)
{
	PauliOperator pauli({"", 1});

	for (auto &j : fermion_item)
	{
		auto op_qubit = j.first;
		std::string op_str;
		auto max = maxqubit - 1;		
        for (size_t i = op_qubit + 1; i <= max; i++)
		{
			op_str += "x" + std::to_string(i) + " ";
		}
		
		std::string op_str1;
		
		if (op_qubit >= 1)
		{
			size_t m = op_qubit - 1;
			op_str1 = "z" + std::to_string(m) + " " + "x" + std::to_string(op_qubit) + " " + op_str;
		}
		else
		{
			op_str1 = "x" + std::to_string(op_qubit) + " " + op_str;
		}
		
		std::string op_str2 = "y" + std::to_string(op_qubit) + " " + op_str;

		QPauliMap map;
		map.insert(std::make_pair(op_str1, 0.5));
		if (j.second)
		{
			map.insert(std::make_pair(op_str2, complex_d(0, -0.5)));
		}
		else
		{
			map.insert(std::make_pair(op_str2, complex_d(0, 0.5)));
		}

		pauli *= PauliOperator(map);
		
	}

	return pauli;
}

PauliOperator ParityTransform(const FermionOperator &fermion)
{
	auto data = fermion.data();
	auto maxqubit = fermion.getMaxIndex();

	PauliOperator pauli;
	for (auto &i : data)
	{
		pauli += ParityTransform(i.first.first,maxqubit)*i.second;
	}

	return pauli;
}

VarPauliOperator ParityTransform(const VarFermionOperator &fermion)
{
	auto data = fermion.data();
	auto maxqubit = fermion.getMaxIndex();

	VarPauliOperator pauli;
	for (auto &i : data)
	{
		PauliOperator one_pauli = ParityTransform(i.first.first, maxqubit);
		for (auto &j : one_pauli.data())
		{
			pauli += VarPauliOperator(j.first.second,
				complex_var(i.second.real() * j.second.real() - i.second.imag()*j.second.imag(),
					i.second.real()*j.second.imag() + i.second.imag() * j.second.real()));
		}
	}
	
	return pauli;
}

std::vector<Eigen::MatrixXi> BKMatrix(size_t qn)
{
	size_t k = 1;
	Eigen::MatrixXi beta2(2, 2), _beta2(2, 2);
	beta2 << 1, 0,
		1, 1;
	_beta2 << 1, 0,
		1, 1;
	while (size_t n = pow(2, k) < qn)
	{
		k++;
	}
	Eigen::MatrixXi pi(size_t(pow(2, k)), size_t(pow(2, k)));
	for (size_t i = 0; i <= pow(2, k) - 1; i++)
	{
		for (size_t j = 0; j <= pow(2, k) - 1; j++)
		{
			if (i <= j)
			{
				pi(i, j) = 0;
			}
			else
			{
				pi(i, j) = 1;
			}
		}
	}

	Eigen::MatrixXi beta, _beta;
	for (size_t i = 1; i <= k; i++)
	{
		if (i == 1)
		{
			beta = beta2;
			_beta = _beta2;
		}
		else
		{
			size_t size = pow(2, i - 1);
			Eigen::MatrixXi betax(size, size);
			betax = beta;
			Eigen::MatrixXi _betax(size, size);
			_betax = _beta;
			beta.resize(2 * size, 2 * size);
			_beta.resize(2 * size, 2 * size);
			beta.topLeftCorner(size, size) = betax;
			beta.bottomRightCorner(size, size) = betax;
			beta.bottomLeftCorner(size, size).setZero();
			beta.bottomLeftCorner(1, size).setOnes();
			beta.topRightCorner(size, size).setZero();
			_beta.topLeftCorner(size, size) = _betax;
			_beta.bottomRightCorner(size, size) = _betax;
			_beta.bottomLeftCorner(size, size).setZero();
			_beta(2 * size - 1, size - 1) = 1;
			_beta.topRightCorner(size, size).setZero();
		}
	}
	Eigen::MatrixXi pi_beta = pi * _beta;
	for (size_t i = 0; i <= pow(2, k) - 1; i++)
	{
		for (size_t j = 0; j <= pow(2, k) - 1; j++)
		{
			pi_beta(i, j) = (pi_beta(i, j)) % 2;
		}
	}
	std::vector<Eigen::MatrixXi> BK;
	BK.insert(BK.begin(), beta);
	BK.insert(BK.end(), pi_beta);
	BK.insert(BK.end(), _beta);
	std::cout << BK[0] << std::endl;
	std::cout << BK[1] << std::endl;
	std::cout << BK[2] << std::endl;
	return BK;
}
PauliOperator BravyiKitaevTransform(const OrbitalActVec &fermion_item, size_t maxqubit, std::vector<Eigen::MatrixXi> BK)
{
	auto beta = BK[0];
	auto pi_beta = BK[1];
	auto _beta = BK[2];
	size_t k = 1;
	while (size_t n = pow(2, k) < maxqubit)
	{
		k++;
	}
	PauliOperator pauli({ "", 1 });
	for (auto &j : fermion_item)
	{
		auto op_qubit = j.first;
		std::string op_str;
		for (size_t i = op_qubit + 1; i < pow(2, k); i++)
		{
			if (1 == beta(i, op_qubit))
				op_str = op_str + "X" + std::to_string(i) + " ";
		}

		std::string op_str1 = op_str + "X" + std::to_string(op_qubit) + " ";
		for (signed int i = op_qubit - 1; i >= 0; i--)
		{
			if (1 == pi_beta(op_qubit, i))
				op_str1 = op_str1 + "Z" + std::to_string(i) + " ";
		}

		std::string op_str2 = op_str + "Y" + std::to_string(op_qubit) + " ";
		if (0 == op_qubit % 2)
		{
 			for (signed int i = op_qubit - 1; i >= 0; i--)
			{
				if (1 == pi_beta(op_qubit, i))
					op_str2 = op_str2 + "Z" + std::to_string(i) + " ";
			}
		}
		else
		{
			for (signed int i = op_qubit - 1; i >= 0; i--)
			{
				if ((1 == pi_beta(op_qubit, i)) && (0 == _beta(op_qubit, i)))
					op_str2 = op_str2 + "Z" + std::to_string(i) + " ";
			}
		}

		QPauliMap map;
		map.insert(std::make_pair(op_str1, 0.5));
		if (j.second)
		{
			map.insert(std::make_pair(op_str2, complex_d(0, -0.5)));
		}
		else
		{
			map.insert(std::make_pair(op_str2, complex_d(0, 0.5)));
		}

		pauli *= PauliOperator(map);
	}

	return pauli;
}

PauliOperator BravyiKitaevTransform(const FermionOperator &fermion, std::vector<Eigen::MatrixXi> BK)
{
	auto data = fermion.data();
	auto maxqubit = fermion.getMaxIndex();
	PauliOperator pauli;
	for (auto &i : data)
	{
		pauli += BravyiKitaevTransform(i.first.first, maxqubit, BK)*i.second;
	}

	return pauli;
}

VarPauliOperator BravyiKitaevTransform(const VarFermionOperator &fermion, std::vector<Eigen::MatrixXi> BK)
{
	auto data = fermion.data();
	auto maxqubit = fermion.getMaxIndex();
	VarPauliOperator pauli;
	for (auto &i : data)
	{
		PauliOperator one_pauli = BravyiKitaevTransform(i.first.first, maxqubit, BK);
		for (auto &j : one_pauli.data())
		{
			pauli += VarPauliOperator(j.first.second,
				complex_var(i.second.real() * j.second.real() - i.second.imag()*j.second.imag(),
					i.second.real()*j.second.imag() + i.second.imag() * j.second.real()));
		}
	}
	
	return pauli;
}


size_t getCCS_N_Trem(size_t qn, size_t en)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    return (qn - en) * en;
}

size_t getCCSD_N_Trem(size_t qn, size_t en)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    return (qn - en) * en + (qn - en)* (qn -en - 1) * en * (en - 1) / 4;
}

FermionOperator getCCS(
        size_t qn,
        size_t en,
        const vector_d &para_vec)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    if (qn == en)
    {
        return FermionOperator();
    }

    if (getCCS_N_Trem(qn, en) != para_vec.size())
    {
        std::string err = "CCS para error!";
        QCERR(err);
        throw std::runtime_error(err);
    }

    FermionOp<complex_d>::FermionMap map;
    size_t cnt = 0;
    for (size_t i = 0; i < en; i++)
    {
        for (auto ex = en; ex < qn; ex++)
        {
            map.insert(std::make_pair(std::to_string(ex)+"+ "
                                      + std::to_string(i),
                                      para_vec[cnt]));

            cnt++;
        }
    }

    return FermionOperator(map);
}

VarFermionOperator getCCS(
        size_t qn,
        size_t en,
        var &para)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    if (qn == en)
    {
        return VarFermionOperator();
    }

    if (getCCS_N_Trem(qn, en) != para.getValue().rows())
    {
        std::string err = "CCS para error!";
        QCERR(err);
        throw std::runtime_error(err);
    }

    FermionOp<complex_var>::FermionMap map;
    size_t cnt = 0;
    for (size_t i = 0; i < en; i++)
    {
        for (auto ex = en; ex < qn; ex++)
        {
            map.insert(std::make_pair(std::to_string(ex)+"+ "
                                      + std::to_string(i),
                                      complex_var(para[cnt], 0)));

            cnt++;
        }
    }

    return VarFermionOperator(map);
}

VarFermionOperator getCCS(size_t qn, size_t en, std::vector<var>& para)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    if (qn == en)
    {
        return VarFermionOperator();
    }

    if (getCCS_N_Trem(qn, en) != para.size())
    {
        std::string err = "CCS para error!";
        QCERR(err);
        throw std::runtime_error(err);
    }

    FermionOp<complex_var>::FermionMap map;
    size_t cnt = 0;
    for (size_t i = 0; i < en; i++)
    {
        for (auto ex = en; ex < qn; ex++)
        {
            map.insert(std::make_pair(std::to_string(ex) + "+ "
                + std::to_string(i),
                complex_var(para[cnt], 0)));

            cnt++;
        }
    }

    return VarFermionOperator(map);
}

FermionOperator getCCSD(size_t qn, size_t en, const vector_d &para_vec)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    if (qn == en)
    {
        return FermionOperator();
    }

    if (getCCSD_N_Trem(qn, en) != para_vec.size())
    {
        std::string err = "CCSD para error!";
        QCERR(err);
        throw std::runtime_error(err);
    }

    FermionOp<complex_d>::FermionMap map;

    size_t cnt = 0;
    for (size_t i = 0; i < en; i++)
    {
        for (auto ex = en; ex < qn; ex++)
        {
            map.insert(std::make_pair(std::to_string(ex) + "+ "
                + std::to_string(i),
                para_vec[cnt]));

            cnt++;
        }
    }

    for (size_t i = 0; i < en; i++)
    {
        for (size_t j = i + 1; j < en; j++)
        {
            for (size_t ex1 = en; ex1 < qn; ex1++)
            {
                for (size_t ex2 = ex1 + 1; ex2 < qn; ex2++)
                {
                    map.insert(std::make_pair(std::to_string(ex2)+"+ "+
                                              std::to_string(ex1)+"+ "+
                                              std::to_string(j) + " "+
                                              std::to_string(i),
                                              para_vec[cnt]));
                    cnt++;
                }
            }
        }
    }

    return FermionOperator(map);
}

VarFermionOperator getCCSD(size_t qn, size_t en, var &para)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    if (qn == en)
    {
        return VarFermionOperator();
    }

    if (getCCSD_N_Trem(qn, en) != para.getValue().rows())
    {
        std::string err = "CCSD para error!";
        QCERR(err);
        throw std::runtime_error(err);
    }

    FermionOp<complex_var>::FermionMap map;
    size_t cnt = 0;
    for (size_t i = 0; i < en; i++)
    {
        for (auto ex = en; ex < qn; ex++)
        {
            map.insert(std::make_pair(std::to_string(ex) + "+ "
                + std::to_string(i),
                complex_var(para[cnt], 0)));

            cnt++;
        }
    }

    for (size_t i = 0; i < en; i++)
    {
        for (size_t j = i + 1; j < en; j++)
        {
            for (size_t ex1 = en; ex1 < qn; ex1++)
            {
                for (size_t ex2 = ex1 + 1; ex2 < qn; ex2++)
                {
                    map.insert(std::make_pair(std::to_string(ex2)+"+ "+
                                              std::to_string(ex1)+"+ "+
                                              std::to_string(j) + " "+
                                              std::to_string(i),
                                              complex_var(para[cnt], 0)));
                    cnt++;
                }
            }
        }
    }

    return VarFermionOperator(map);
}

VarFermionOperator getCCSD(size_t qn, size_t en, std::vector<var>& para)
{
    if (qn < en)
    {
        std::string err = "Qubit num is less than electron num.";
        QCERR(err);
        throw std::runtime_error(err);
    }

    if (qn == en)
    {
        return VarFermionOperator();
    }

    if (getCCSD_N_Trem(qn, en) != para.size())
    {
        std::string err = "CCSD para error!";
        QCERR(err);
        throw std::runtime_error(err);
    }

    FermionOp<complex_var>::FermionMap map;
    size_t cnt = 0;
    for (size_t i = 0; i < en; i++)
    {
        for (auto ex = en; ex < qn; ex++)
        {
            map.insert(std::make_pair(std::to_string(ex) + "+ "
                + std::to_string(i),
                complex_var(para[cnt], 0)));

            cnt++;
        }
    }

    for (size_t i = 0; i < en; i++)
    {
        for (size_t j = i + 1; j < en; j++)
        {
            for (size_t ex1 = en; ex1 < qn; ex1++)
            {
                for (size_t ex2 = ex1 + 1; ex2 < qn; ex2++)
                {
                    map.insert(std::make_pair(std::to_string(ex2) + "+ " +
                        std::to_string(ex1) + "+ " +
                        std::to_string(j) + " " +
                        std::to_string(i),
                        complex_var(para[cnt], 0)));
                    cnt++;
                }
            }
        }
    }

    return VarFermionOperator(map);
}

PauliOperator transCC2UCC(const PauliOperator &cc)
{
    return complex_d(0, 1)*(cc - cc.dagger());
}

VarPauliOperator transCC2UCC(const VarPauliOperator &cc)
{
    VarPauliOperator pauli;
    for (auto& i : cc.data()) 
    {
        pauli += VarPauliOperator(i.first.second,
            complex_var(-2 * i.second.imag(), 0));
    }
    return pauli;
}

VQC simulateHamiltonian(
        QVec &qubit_vec,
        VarPauliOperator &pauli,
        double t,
        size_t slices)
{
    VQC circuit;
    if ((0 == qubit_vec.size()) ||
        (0 == pauli.data().size()) ||
        (0 == slices))
    {
        return circuit;
    }

    for (auto i = 0u; i < slices; i++)
    {
        for (auto j = 0u; j < pauli.data().size(); j++)
        {
            auto item = pauli.data();
            auto term = item[j].first.first;
			
            circuit.insert(simulateOneTerm(
                qubit_vec,                       
                term,
                item[j].second.real(),
                t / slices));
        }
    }

    return circuit;
}

VQC simulateOneTerm(
        QVec &qubit_vec,
        const QTerm &hamiltonian_term,
        const var &coef,
        double t)
{
    VQC circuit;
    if ((0 == qubit_vec.size()) ||
        (0 == hamiltonian_term.size()))
    {
        return circuit;
    }

    QCircuit transform;
    QVec tmp_vec;
    auto iter = hamiltonian_term.begin();
    for (; iter != hamiltonian_term.end(); iter++)
    {
        auto key = iter->first;
        auto value = iter->second;

        char ch = toupper(value);
        switch (ch)
        {
        case 'X':
            transform << H(qubit_vec[key]);
            break;
        case 'Y':
            transform << RX(qubit_vec[key], Q_PI_2);
            break;
        case 'Z':
            break;
        default:
            std::string err = "bad Pauli item.";
            QCERR(err);
            throw std::runtime_error(err);
        }
        tmp_vec.emplace_back(qubit_vec[key]);
    }

    circuit.insert(transform);
    circuit.insert(simulateZTerm(tmp_vec, coef, t));
    //circuit.insert(transform.dagger());
    
    for (auto iter = hamiltonian_term.begin(); iter != hamiltonian_term.end(); iter++)
    {
        auto key = iter->first;
        auto value = iter->second;

        char ch = toupper(value);
        switch (ch)
        {
        case 'X':
            circuit.insert(H(qubit_vec[key]));
            break;
        case 'Y':
            circuit.insert(RX(qubit_vec[key], -Q_PI_2));
            break;
        case 'Z':
            break;
        default:
            std::string err = "bad Pauli item.";
            QCERR(err);
            throw std::runtime_error(err);
        }
    }

    return circuit;
}

VQC simulateZTerm(QVec &qubit_vec, const var &coef, double t)
{
    VQC circuit;
    if (0 == qubit_vec.size())
    {
        return circuit;
    }
    else if (1 == qubit_vec.size())
    {
        circuit.insert(VQG_RZ(qubit_vec[0], 2*coef*t));
    }
    else
    {
        for (auto i = 0u; i < qubit_vec.size() - 1; i++)
        {
            circuit.insert(CNOT(qubit_vec[i], qubit_vec[qubit_vec.size()-1]));
        }
        circuit.insert(VQG_RZ(qubit_vec[qubit_vec.size() - 1], 2*coef*t));
        for (auto i = 0u; i < qubit_vec.size() - 1; i++)
        {
            circuit.insert(CNOT(qubit_vec[i], qubit_vec[qubit_vec.size()-1]));
        }
    }

    return circuit;
}

FermionOperator parsePsi4DataToFermion(const std::string& data)
{
    FermionOperator::FermionMap fermion_map;
    QString contents(data);
    auto contents_vec = contents.split("\r\n", QString::SkipEmptyParts);
    for (size_t i = 0; i < contents_vec.size(); i++)
    {
        const auto& item = contents_vec[i];
        auto item_vec = item.split(":", QString::SkipEmptyParts);
        if (2 != item_vec.size())
        {
            QCERR("Psi4 data format error!");
            throw std::runtime_error("Psi4 data format error!");
        }

        auto real_value = item_vec[1].toDouble();
        complex_d value(real_value);
        auto len = item_vec[0].size();
        auto inner_str = item_vec[0].mid(1, len - 2);
        auto inner_vec =
            inner_str.splitByStr("), (", QString::SkipEmptyParts);

        std::string key;
        if (!inner_vec.empty())
        {
            inner_vec[0] = inner_vec[0].mid(1);
            auto& last = inner_vec[inner_vec.size() - 1];
            last = last.left(last.size() - 1);

            for (size_t j = 0; j < inner_vec.size(); j++)
            {
                auto tmp_vec =
                    inner_vec[j].split(",", QString::SkipEmptyParts);

                if (2 != tmp_vec.size())
                {
                    QCERR("Psi4 data content format error!");
                    throw std::runtime_error("Psi4 data content format error!");
                }

                int second = tmp_vec[1].toInt();

                if (j != 0)
                {
                    key += " ";
                }

                key += tmp_vec[0].data();
                if (second == 1)
                {
                    key += "+";
                } 
            }
        }

        fermion_map.insert(std::make_pair(key, real_value));
    }

    return FermionOperator(fermion_map).normal_ordered();
}

QPANDA_END
