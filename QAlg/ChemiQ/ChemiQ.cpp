#include "QAlg/ChemiQ/ChemiQ.h"
#include "Components/ChemiQ/ChemiqUtil.h"
#include "Core/Utilities/Tools/QString.h"
#include "Core/QuantumMachine/QuantumMachineFactory.h"
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "Components/HamiltonianSimulation/HamiltonianSimulation.h"
#include "Variational/Optimizer.h"
#include <sys/stat.h>
#include <cstdlib>
#include <chrono>


const std::string ERR_LOG = "error.log";
const std::string BASE_FILE = "base.dat";
const std::string PROGRESS_FILE = "progress.dat";
const std::string RESULT_FILE = "result.dat";
const std::string MOLECULE_FILE_PR = "molecule_";
const std::string OPTIMIZED_FILE_PR = "optimized_";
const std::string OPTIMIZER_CACHE_FILE_PR = "optimizer_cache_";
const std::string FILE_SUFFIX = ".dat";
const std::string GRADIENT_CACHE_TAG = "GRADIENT CACHE FILE";

namespace QPanda {

	ChemiQ::ChemiQ()
	{
	}

	ChemiQ::~ChemiQ()
	{
		if (nullptr != m_machine.get())
		{
			m_machine->finalize();
		}
	}

	void ChemiQ::initialize(const std::string& dir)
	{
		m_psi4_wapper.initialize(dir);
	}

	void ChemiQ::finalize()
	{
		m_psi4_wapper.finalize();
	}

    int ChemiQ::getQubitsNum()
    {
        if (m_molecules.empty())
        {
            return 0;
        }

        m_psi4_wapper.setMolecule(m_molecules[0]);

		/*get the second quantized Hamiltonian*/
		if (!m_psi4_wapper.run())
		{
			m_last_err = m_psi4_wapper.getLastError();
			writeExecLog(false);
			return -1;
		}

		auto fermion_data = parsePsi4DataToFermion(m_psi4_wapper.getData());

        PauliOperator pauli;
        /*Transform second quantized Hamiltonian to Pauli Hamiltonian */
        if (m_transform_type == TransFormType::Jordan_Wigner)
        {
            pauli = JordanWignerTransform(fermion_data);
        }
        else if (m_transform_type == TransFormType::Parity)
        {
            pauli = ParityTransform(fermion_data);
        }
        else if (m_transform_type == TransFormType::Bravyi_Ktaev)
        {
            size_t m_q = fermion_data.getMaxIndex();
            pauli = BravyiKitaevTransform(fermion_data, BKMatrix(m_q));
        }

        return pauli.getMaxIndex();
    }

	bool ChemiQ::exec()
	{
		std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
		m_machine.reset(QuantumMachineFactory::GetFactoryInstance()
            .CreateByType(m_quantum_machine_type));
        m_machine->init();
		writeProgress(0);
		// check file exists
		std::string base_file = m_save_data_dir + "/" + BASE_FILE;
#ifdef _MSC_VER
		using convert_typeX = std::codecvt_utf8<wchar_t>;
		std::wstring_convert<convert_typeX, wchar_t> converterX;

		auto w_base_file = converterX.from_bytes(base_file);
		if (_waccess(w_base_file.c_str(), 0) != -1)
#else
		struct stat buffer;
		if (stat(base_file.c_str(), &buffer) == 0)
#endif // WIN32
		{
			updateBaseData();
			m_break_restoration = true;
		}

		m_energies.resize(m_molecules.size(), 0);
		for (auto i = 0u; i < m_molecules.size(); i++)
		{
			m_process_i = i;
			m_electron_num = getMoleculerElectronNum(m_molecules[i]);

			if ((m_process_i < m_last_process_i) && (!m_hamiltonian_gen_only))
			{
				// gen pauli data from catche file
				if (!getLastIthMoleculeResult(m_process_i + 1))
				{
					writeExecLog(false);
					return false;
				}
			}
			else
			{
				if (m_hamiltonian_in_file && (!m_hamiltonian_gen_only))
				{
					// get user defined pauli data from file
					if (!getLastIthMoleculeResult(m_process_i + 1))
					{
						writeExecLog(false);
						return false;
					}
				}
				else
				{
					// gen pauli data from psi4 
					if (!getDataFromPsi4(m_process_i))
					{
						writeExecLog(false);
						return false;
					}

					saveMoleculeOptimizedResult(
						i + 1, m_molecules[i],
						m_pauli.toString(), QOptimizationResult());

					if (m_hamiltonian_gen_only)
					{
						writeProgress(std::max(m_optimizer_iter_num,
							m_optimizer_func_call_num));
						continue;
					}
				}
			}

			m_qn = m_pauli.getMaxIndex();

			if (m_transform_type == TransFormType::Bravyi_Ktaev)
			{
				m_BK = BKMatrix(m_qn);
			}

			if (m_ucc_type == UccType::UCCS)
			{
				m_para_num = getCCS_N_Trem(m_qn, m_electron_num);
			}
			else
			{
				m_para_num = getCCSD_N_Trem(m_qn, m_electron_num);
			}

			writeBaseData();
			initOptimizedPara(m_para_num);

			m_func_calls = 0;
			if (!m_save_data_dir.empty())
			{
				std::string filename = m_save_data_dir + "/" + OPTIMIZED_FILE_PR
					+ std::to_string(i + 1) + FILE_SUFFIX;

#ifdef _MSC_VER
				using convert_typeX = std::codecvt_utf8<wchar_t>;
				std::wstring_convert<convert_typeX, wchar_t> converterX;

				auto w_file = converterX.from_bytes(filename);
				if ((m_process_i < m_last_process_i) &&
					(_waccess(w_file.c_str(), 0) != -1))
#else
				struct stat check_buffer;
				if ((m_process_i < m_last_process_i) &&
					(stat(filename.c_str(), &check_buffer) == 0))
#endif // WIN32
				{
					if (!getLastIthMoleculeOptimizedPara(filename))
					{
						writeExecLog(false);
						return false;
					}
				}
				else
				{
					m_optimizer_data_db = OriginCollection(filename, false);
					m_optimizer_data_db = { "index", "energy", "para" };

					m_last_fcalls = 0;

				}
			}

			m_qubit_vec = m_machine->allocateQubits(m_qn);
			QOptimizationResult optimized_result;
			if (m_optimizer_type >= OptimizerType::GRADIENT)
			{
				optimized_result = optimizeByGradient();
			}
			else
			{
				optimized_result = optimizeByNoGradient();
			}
			m_energies[i] = optimized_result.fun_val;
			m_machine->Free_Qubits(m_qubit_vec);

			if (!m_save_data_dir.empty())
			{
				m_optimizer_data_db.write();
			}

			saveMoleculeOptimizedResult(
				i + 1, m_molecules[i], m_pauli.toString(), optimized_result);
		}

		saveResult();
		writeExecLog(true);

		std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start);
		std::cout << "time use:" << time_used.count() << "s" << std::endl;

		return true;

	}

	void ChemiQ::initOptimizedPara(size_t size)
	{
		m_optimized_para.resize(size);
		if (m_random_para)
		{
			srand(time(0));
			for (auto i = 0u; i < m_para_num; i++)
			{
				m_optimized_para[i] = rand() % 314 / 100.0;
			}
		}
		else
		{
			if (m_default_optimized_para.size() == size)
			{
				m_optimized_para = m_default_optimized_para;
			}
			else
			{
				std::fill(m_optimized_para.begin(), m_optimized_para.end(), 0.5);
			}
		}
	}

	size_t ChemiQ::getMoleculerElectronNum(const std::string &moleculer) const
	{
		auto atoms = QString(moleculer).split("\n", QString::SkipEmptyParts);

		size_t num = 0;
		for (auto &i : atoms)
		{
			auto item_list = i.split(" ", QString::SkipEmptyParts);

			if (item_list.size() != 4)
			{
				QCERR("Molecule format error! " + i.data());
				throw std::runtime_error("Molecule format error! " + i.data());
			}

			num += getElectronNum(item_list[0].data());
		}

		return num;
	}

	QCircuit ChemiQ::prepareInitialState(QVec& qlist, size_t en)
	{
		if (qlist.size() < en)
		{
			return QCircuit();
		}

		QCircuit circuit;
		for (size_t i = 0; i < en; i++)
		{
			circuit << X(qlist[i]);
		}

		return circuit;
	}

	QOptimizationResult ChemiQ::optimizeByGradient()
	{
		auto tmp_para = m_optimized_para;
		auto tmp_para2 = m_optimized_para;
		auto min_energy = std::numeric_limits<double>::max();
		auto tmp_energy = min_energy;
		auto tmp_energy2 = min_energy;

		std::string filename = m_save_data_dir + "/" +
			OPTIMIZER_CACHE_FILE_PR + std::to_string(m_process_i + 1) + FILE_SUFFIX;

#ifdef _MSC_VER
		using convert_typeX = std::codecvt_utf8<wchar_t>;
		std::wstring_convert<convert_typeX, wchar_t> converterX;

		auto w_file = converterX.from_bytes(filename);
		if (m_break_restoration &&
			(_waccess(w_file.c_str(), 0) != -1))
#else
		struct stat buffer;
		if (m_break_restoration &&
			(stat(filename.c_str(), &buffer) == 0))
#endif // WIN32
		{
			restoreGradientOptimizerParaFromCache(
				m_optimized_para,
				tmp_para,
				tmp_para2,
				min_energy,
				tmp_energy,
				tmp_energy2,
				m_last_fcalls
			);
		}

		std::cout << "Process: " << m_process_i << std::endl;
		std::vector<var> para;
		for (auto i = 0u; i < m_optimized_para.size(); i++)
		{
			std::cout << m_optimized_para[i] << " ";
			para.emplace_back(var(m_optimized_para[i], true));
		}
		std::cout << std::endl;
		auto optimizer = genGradientOptimizer(para);
		auto leaves = optimizer->get_variables();


		size_t i = m_last_fcalls;
		for (; i < m_optimizer_iter_num; i++)
		{
			optimizer->run(leaves);
			for (auto j = 0u; j < para.size(); j++)
			{
				tmp_para[j] = eval(para[j], true)(0, 0);
				if (m_disp)
				{
					std::cout << eval(para[j], true)(0, 0) << " ";
				}
			}
			std::cout << std::endl;
			tmp_energy = optimizer->get_loss();

			if (tmp_energy < min_energy)
			{
				min_energy = tmp_energy;
				m_optimized_para = tmp_para;
			}

			if (m_disp)
			{
				std::cout << " iter: " << i + 1 << " loss : "
					<< tmp_energy << std::endl;
			}


			if (!m_save_data_dir.empty())
			{
				m_optimizer_data_db.insertValue(i + 1, tmp_energy, tmp_para);
				writeProgress(i + 1);
			}

			//save optimizer cache
			saveGradientOptimizerCacheFile(
				m_optimized_para,
				tmp_para,
				tmp_para2,
				min_energy,
				tmp_energy,
				tmp_energy2,
				i + 1
			);

			// the domain and function-value convergence test
			if (testTermination(tmp_para, tmp_para2, tmp_energy, tmp_energy2))
			{
				break;
			}
			else
			{
				tmp_para2 = tmp_para;
				tmp_energy2 = tmp_energy;
			}
		}

		QOptimizationResult result;
		result.fcalls = i/* + m_last_fcalls*/;
		result.iters = i /*+ m_last_iters*/;
		result.fun_val = min_energy;
		result.para = m_optimized_para;

		return result;
	}

	QOptimizationResult ChemiQ::optimizeByNoGradient()
	{
		auto optimizer = OptimizerFactory::makeOptimizer(m_optimizer_type);
		if (nullptr == optimizer.get())
		{
			QCERR("Create optimizer failed!");
			throw std::runtime_error("Create optimizer failed!");
		}

		optimizer->setMaxIter(m_optimizer_iter_num);
		optimizer->setMaxFCalls(m_optimizer_func_call_num);
		optimizer->setXatol(m_xatol);
		optimizer->setFatol(m_fatol);
		optimizer->setDisp(m_disp);
		optimizer->registerFunc(std::bind(&ChemiQ::callVQE,
			this,
			std::placeholders::_1,
			m_pauli.toHamiltonian()),
			m_optimized_para);

		if (!m_save_data_dir.empty())
		{
			std::string filename = m_save_data_dir + "/" +
				OPTIMIZER_CACHE_FILE_PR + std::to_string(m_process_i + 1)
				+ FILE_SUFFIX;
			optimizer->setCacheFile(filename);
		}

		optimizer->setRestoreFromCacheFile(m_break_restoration);

		optimizer->exec();
		auto result = optimizer->getResult();

		return result;
	}

	bool ChemiQ::ParityCheck(size_t state, const QTerm &paulis) const
	{
		size_t check = 0;
		for (auto iter = paulis.begin(); iter != paulis.end(); iter++)
		{
			auto value = state >> iter->first;
			if ((value % 2) == 1)
			{
				check++;
			}
		}

		return 1 == check % 2;
	}

	std::shared_ptr<Optimizer>
		ChemiQ::genGradientOptimizer(std::vector<var>& para)
	{
		VarFermionOperator fermion_cc;
		if (m_ucc_type == UccType::UCCS)
		{
			fermion_cc = getCCS(m_qn, m_electron_num, para);
		}
		else
		{
			fermion_cc = getCCSD(m_qn, m_electron_num, para);
		}

		VarPauliOperator pauli_cc;
		if (m_transform_type == TransFormType::Jordan_Wigner)
		{
			pauli_cc = JordanWignerTransform(fermion_cc);
		}
		else if (m_transform_type == TransFormType::Parity)
		{
			pauli_cc = ParityTransform(fermion_cc);
		}
		else if (m_transform_type == TransFormType::Bravyi_Ktaev)
		{
			pauli_cc = BravyiKitaevTransform(fermion_cc, m_BK);
		}

		VarPauliOperator ucc = transCC2UCC(pauli_cc);

		VQC vqc;

		vqc.insert(prepareInitialState(m_qubit_vec, m_electron_num));
		//vqc.insert(X(m_qubit_vec[0])).insert(X(m_qubit_vec[2]));
		vqc.insert(simulateHamiltonian(
			m_qubit_vec,
			ucc,
			m_evolutionTime,
			m_hamiltonian_simulation_slices));

		var loss = qop(vqc, m_pauli, m_machine.get(), m_qubit_vec);

		return VanillaGradientDescentOptimizer::minimize(
			loss,
			m_learning_rate,
			0);
	}

	QResultPair ChemiQ::callVQE(
		const vector_d &para,
		const QHamiltonian &hamiltonian)
	{
		FermionOperator fermion_cc;
		if (m_ucc_type == UccType::UCCS)
		{
			fermion_cc = getCCS(m_qn, m_electron_num, para);
		}
		else
		{
			fermion_cc = getCCSD(m_qn, m_electron_num, para);
		}

		PauliOperator pauli_cc;
		if (m_transform_type == TransFormType::Jordan_Wigner)
		{
			pauli_cc = JordanWignerTransform(fermion_cc);
		}
		else if (m_transform_type == TransFormType::Parity)
		{
			pauli_cc = ParityTransform(fermion_cc);
		}
		else if (m_transform_type == TransFormType::Bravyi_Ktaev)
		{
			pauli_cc = BravyiKitaevTransform(fermion_cc, m_BK);
		}



		PauliOperator ucc = transCC2UCC(pauli_cc);
		QHamiltonian ucc_hamiltonian = ucc.toHamiltonian();

		double expectation = 0.0;
		for (size_t i = 0; i < hamiltonian.size(); i++)
		{
			expectation += getExpectation(ucc_hamiltonian, hamiltonian[i]);
		}

		if (!m_save_data_dir.empty())
		{
			m_func_calls++;
			m_optimizer_data_db.insertValue(
				m_func_calls + m_last_fcalls,
				expectation, para);

			if ((m_func_calls + m_last_fcalls) % 10 == 0)
			{
				writeProgress(m_func_calls + m_last_fcalls);
			}
		}
		return std::make_pair("", expectation);
	}

	double ChemiQ::getExpectation(
		const QHamiltonian &unitary_cc,
		const QHamiltonianItem &component)
	{
		QProg prog;

		prog << prepareInitialState(m_qubit_vec, m_electron_num)
			<< simulateHamiltonian(
				m_qubit_vec,
				unitary_cc,
				m_evolutionTime,
				m_hamiltonian_simulation_slices);

		for (auto iter : component.first)
		{
			if (iter.second == 'X')
			{
				prog << H(m_qubit_vec[iter.first]);
			}
			else if (iter.second == 'Y')
			{
				prog << RX(m_qubit_vec[iter.first], PI / 2);
			}
		}

		m_machine->directlyRun(prog);

		double expectation = 0.0;

		auto temp = dynamic_cast<IdealMachineInterface *>(m_machine.get());
		if (nullptr == temp)
		{
			QCERR("m_machine is not ideal machine");
			throw std::runtime_error("m_machine is not ideal machine");
		}
		auto result = temp->PMeasure(m_qubit_vec, -1);

		for (auto i = 0u; i < result.size(); i++)
		{
			if (ParityCheck(result[i].first, component.first))
			{
				expectation -= result[i].second;
			}
			else
			{
				expectation += result[i].second;
			}
		}

		return expectation * component.second;
	}

	bool ChemiQ::testTermination(
		const vector_d& p1,
		const vector_d& p2,
		double e1,
		double e2) const
	{
		vector_d err(p1.size(), 0);
		for (auto i = 0u; i < p1.size(); i++)
		{
			err[i] = fabs(p1[i] - p2[i]);
		}

		bool flag = true;
		for (auto& i : err)
		{
			if (i > m_xatol)
			{
				flag = false;
				break;
			}
		}

		return flag && (fabs(e1 - e2) < m_fatol);
	}

	bool ChemiQ::saveResult() const
	{
		if (m_save_data_dir.empty() || m_hamiltonian_gen_only)
		{
			return true;
		}

		OriginCollection result(m_save_data_dir + "/" + RESULT_FILE, false);
		result = { "calc_node_index", "molecule", "energy" };
		for (auto i = 0u; i < m_molecules.size(); i++)
		{
			result.insertValue(i + 1, m_molecules[i], m_energies[i]);
		}

		if (!result.write())
		{
			return false;
		}

		OriginCollection progress(m_save_data_dir + "/" + PROGRESS_FILE, false);
		auto iters = m_optimizer_iter_num;
		if (m_optimizer_type < OptimizerType::GRADIENT)
		{
			iters = std::max(m_optimizer_iter_num, m_optimizer_func_call_num);
		}

		auto total_num = m_molecules.size() * iters;
		progress = { "cur_iters", "total_num", "progress" };

		progress.insertValue(total_num, total_num, 1.0);

		return progress.write();
	}

	bool ChemiQ::saveMoleculeOptimizedResult(
		size_t index,
		const std::string& molecule,
		const std::string& pauli,
		const QOptimizationResult& result) const
	{
		if (m_save_data_dir.empty())
		{
			return true;
		}

		std::string filename = m_save_data_dir + "/" + MOLECULE_FILE_PR +
			std::to_string(index) + FILE_SUFFIX;

		OriginCollection collection(filename, false);
		collection = { "molecule", "energy", "iters", "fcalls", "para", "pauli" };

		collection.insertValue(molecule, result.fun_val, result.iters,
			result.fcalls, result.para, pauli);

		return collection.write();
	}

	bool ChemiQ::writeBaseData() const
	{
		std::string filename = m_save_data_dir + "/" + BASE_FILE;

		OriginCollection collection(filename, false);
		collection = { "electronic_num", "orbit_num",
					   "optimizer_para_num", "calc_node_num","cur_calc_index" };

		auto tmp_process = (m_process_i + 1) < m_last_process_i
			? m_last_process_i : m_process_i + 1;
		collection.insertValue(m_electron_num, m_qn, m_para_num,
			m_molecules.size(), tmp_process);

		return collection.write();
	}

	bool ChemiQ::writeExecLog(bool exec_flag) const
	{
		if (m_save_data_dir.empty())
		{
			return true;
		}

		std::string filename = m_save_data_dir + "/" + ERR_LOG;

		OriginCollection collection(filename, false);
		collection = { "status", "message" };

		collection.insertValue(exec_flag ? 0 : -1, m_last_err);

		return collection.write();
	}

	bool ChemiQ::writeProgress(size_t iter_num) const
	{
		if (m_save_data_dir.empty())
		{
			return true;
		}

		auto iters = m_optimizer_iter_num;
		if (m_optimizer_type < OptimizerType::GRADIENT)
		{
			iters = std::max(m_optimizer_iter_num, m_optimizer_func_call_num);
		}

		auto total_num = m_molecules.size() * iters;

		auto cur_iters = m_process_i * iters + iter_num;
		double progress = cur_iters * 1.0 / total_num;

		std::string filename = m_save_data_dir + "/" + PROGRESS_FILE;

		OriginCollection collection(filename, false);
		collection = { "cur_iters", "total_num", "progress" };

		collection.insertValue(cur_iters, total_num, progress);

		return collection.write();
	}

	bool ChemiQ::updateBaseData()
	{
		if (m_save_data_dir.empty())
		{
			QCERR("save data dir is not set!");
			return false;
		}

		std::string filename = m_save_data_dir + "/" + BASE_FILE;
		OriginCollection base_file;
		if (!base_file.open(filename))
		{
			QCERR(std::string("Open file failed! filename: ") + filename);
			return false;
		}

		m_last_process_i = QString(base_file.getValue("cur_calc_index")[0]).toInt();

		return true;
	}

	bool ChemiQ::getLastIthMoleculeResult(size_t index)
	{
		if (m_save_data_dir.empty())
		{
			QCERR("save data dir is not set!");
			m_last_err = "save data dir is not set!";
			return false;
		}

		std::string filename = m_save_data_dir + "/" + MOLECULE_FILE_PR +
			std::to_string(index) + FILE_SUFFIX;
		OriginCollection molecule_file;
		if (!molecule_file.open(filename))
		{
			QCERR(std::string("Open file failed! filename: ") + filename);
			m_last_err = std::string("Open file failed! filename: ") + filename;
			return false;
		}

		auto pauli = QString(molecule_file.getValue("pauli")[0]);
		auto pauli_item = pauli.split("\n", QString::SkipEmptyParts);
		if (pauli_item.size() < 2)
		{
			QCERR(std::string("Pauli string format error!") + pauli.data());
			m_last_err = std::string("Pauli string format error!") + pauli.data();
			return false;
		}

		PauliOperator tmp_pauli;
		for (auto i = 1u; i < pauli_item.size() - 1; i++)
		{
			auto item_list = pauli_item[i].split(":");
			if (item_list.size() != 2)
			{
				QCERR(std::string("Pauli string format error!") + pauli.data());
				m_last_err = std::string("Pauli string format error!")
					+ pauli.data();
				return false;
			}

			tmp_pauli += PauliOperator(
				item_list[0].data(),
				item_list[1].toDouble());
		}

		m_pauli = tmp_pauli;

		return true;
	}

	bool ChemiQ::getLastIthMoleculeOptimizedPara(const std::string &filename)
	{
		if (!m_optimizer_data_db.open(filename))
		{
			QCERR(std::string("Open file failed! filename: ") + filename);
			m_last_err = std::string("Open file failed! filename: ") + filename;
			return false;
		}

		auto index_vec = m_optimizer_data_db.getValue("index");
		m_last_fcalls = QString(index_vec[index_vec.size() - 1]).toInt();

		auto para_vec = m_optimizer_data_db.getValue("para");
		if (para_vec.empty())
		{
			QCERR("get optimized data failed!");
			m_last_err = "get optimized data failed!";
			return false;
		}

		m_optimized_para = getVectorFromString(para_vec[para_vec.size() - 1]);

		return true;
	}

	bool ChemiQ::getDataFromPsi4(size_t index)
	{
		m_psi4_wapper.setMolecule(m_molecules[index]);
		size_t cnt = 0;
		FermionOperator fermion_data;
		/*get the second quantized Hamiltonian*/
		if (!m_psi4_wapper.run())
		{
			m_last_err = m_psi4_wapper.getLastError();
			return false;
		}

		fermion_data = parsePsi4DataToFermion(m_psi4_wapper.getData());
		std::cout << fermion_data << std::endl;

		/*Transform second quantized Hamiltonian to Pauli Hamiltonian */
		if (m_transform_type == TransFormType::Jordan_Wigner)
		{
			m_pauli = JordanWignerTransform(fermion_data);
		}
		else if (m_transform_type == TransFormType::Parity)
		{
			m_pauli = ParityTransform(fermion_data);
		}
		else if (m_transform_type == TransFormType::Bravyi_Ktaev)
		{
			size_t m_q = fermion_data.getMaxIndex();
			m_pauli = BravyiKitaevTransform(fermion_data, BKMatrix(m_q));
		}

		std::cout << m_pauli << std::endl;

		return true;
	}


	bool ChemiQ::saveGradientOptimizerCacheFile(
		const vector_d& best_paras,
		const vector_d& cur_paras,
		const vector_d& last_paras,
		double b_value,
		double c_value,
		double l_value,
		size_t cur_iter) const
	{
		if (m_save_data_dir.empty())
		{
			return true;
		}

		std::string filename = m_save_data_dir + "/" +
			OPTIMIZER_CACHE_FILE_PR + std::to_string(m_process_i + 1) + FILE_SUFFIX;

		OriginCollection collection(filename, false);
		collection = { "index", "tag", "best_para", "cur_para","last_para",
			"best_value", "cur_value", "last_value", "cur_iter" };


		collection.insertValue(0, GRADIENT_CACHE_TAG, best_paras,
			cur_paras, last_paras, b_value, c_value, l_value, cur_iter);

		return collection.write();
	}

	bool ChemiQ::restoreGradientOptimizerParaFromCache(
		vector_d& best_paras,
		vector_d& cur_paras,
		vector_d& last_paras,
		double& b_value,
		double& c_value,
		double& l_value,
		size_t& cur_iter)
	{
		OriginCollection cache_file;
		std::string filename = m_save_data_dir + "/" +
			OPTIMIZER_CACHE_FILE_PR + std::to_string(m_process_i + 1) + FILE_SUFFIX;
		if (!cache_file.open(filename))
		{
			std::cout << std::string("Open file failed! filename: ") + filename;
			return false;
		}

		std::string tag = cache_file.getValue("tag")[0];
		if (tag != GRADIENT_CACHE_TAG)
		{
			std::cout << "It is not a GRADIENT cache file! Tag: " << tag
				<< std::endl;
			return false;
		}

		best_paras = getVectorFromString(cache_file.getValue("best_para")[0]);
		cur_paras = getVectorFromString(cache_file.getValue("cur_para")[0]);
		last_paras = getVectorFromString(cache_file.getValue("last_para")[0]);
		b_value = QString(cache_file.getValue("best_value")[0]).toDouble();
		c_value = QString(cache_file.getValue("cur_value")[0]).toDouble();
		l_value = QString(cache_file.getValue("last_value")[0]).toDouble();
		m_last_iters = QString(cache_file.getValue("cur_iter")[0]).toInt();

		return true;
	}

	vector_d ChemiQ::getVectorFromString(const std::string& para) const
	{
		auto tmp_para = QString(para).mid(1, para.size() - 2);
		auto item_list = tmp_para.split(",", QString::SkipEmptyParts);

		vector_d data;
		data.resize(item_list.size());
		for (auto i = 0u; i < item_list.size(); i++)
		{
			data[i] = item_list[i].toDouble();
		}

		return data;
	}
}
