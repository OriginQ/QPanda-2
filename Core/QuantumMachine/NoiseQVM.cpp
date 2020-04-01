#include "OriginQuantumMachine.h"
#include "NoiseQPU/NoiseCPUImplQPU.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/QuantumMachine/QProgExecution.h"
#include "QPandaConfig.h"
#ifdef USE_MPI
#include "mpi.h"
#include "Core/Utilities/Tools/Uinteger.h"
#endif
USING_QPANDA
using namespace std;

NoiseQVM::NoiseQVM()
{
    m_gates_matrix = { { "X","Y","Z",
        "T","S","H",
        "RX","RY","RZ",
        "U1" },
    { "CNOT" } };
    m_valid_gates_matrix.resize(m_gates_matrix.size());
}


void NoiseQVM::_getValidGatesMatrix()
{
    if (SingleGateTransferType::SINGLE_GATE_INVALID ==
        SingleGateTypeValidator::GateType(m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE],
            m_valid_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE]))   /* single gate data MetadataValidity */
    {
        finalize();
        QCERR("gates valid error");
        throw runtime_error("gates valid error");
    }
    if (m_gates_matrix.size() >= 2)
    {
        if (DoubleGateTransferType::DOUBLE_GATE_INVALID ==
            DoubleGateTypeValidator::GateType(m_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE],
                m_valid_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE]))   /* double gate data MetadataValidity */
        {
            finalize();
            QCERR("gates valid error");
            throw runtime_error("gates valid error");
        }
    }

}

void NoiseQVM::set_noise_model(NOISE_MODEL model, GateType type, std::vector<double> params_vec)
{
	m_models_vec.push_back(model);

	auto gate_name = TransformQGateType::getInstance()[type];
	m_gates_vec.push_back(gate_name);

	m_params_vecs.push_back(params_vec);
}

void NoiseQVM::init()
{
	if (  !m_models_vec.empty() 
		&& m_models_vec.size() == m_params_vecs.size()
		&& m_params_vecs.size() == m_gates_vec.size() )
	{
		m_doc.Parse("{}");
		Value value_object(rapidjson::kObjectType);
		for (int i = 0; i < m_models_vec.size(); i++)
		{
			Value value_array(rapidjson::kArrayType);
			value_array.PushBack(m_models_vec[i], m_doc.GetAllocator());
			for (auto iter : m_params_vecs[i])
			{
				value_array.PushBack(iter, m_doc.GetAllocator());
			}
			std::string str_gate = m_gates_vec[i];
			Value gate_name(rapidjson::kStringType);
			gate_name.SetString(str_gate.c_str(), (rapidjson::SizeType)str_gate.size(), m_doc.GetAllocator());

			value_object.AddMember(gate_name, value_array, m_doc.GetAllocator());
		}
		m_doc.AddMember("noisemodel", value_object, m_doc.GetAllocator());
	}
	else
	{  
		//default config
		m_doc.Parse("{}");
		auto & alloc = m_doc.GetAllocator();
		Value noise_model_value(rapidjson::kObjectType);
		for (auto a : m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE])
		{
			Value value(rapidjson::kArrayType);
			value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, alloc);
			value.PushBack(0.5, alloc);
			noise_model_value.AddMember(Value().SetString(a.c_str(), alloc).Move(), value, alloc);
		}

		m_doc.AddMember("noisemodel", noise_model_value, alloc);
	}

    try
    {
		_start();
		_getValidGatesMatrix();
		_pGates = new NoisyCPUImplQPU(m_doc);
        _ptrIsNull(_pGates, "NoisyCPUImplQPU");
    }
    catch (const std::exception&e)
    {
        finalize();
        QCERR(e.what());
        throw init_fail(e.what());
    }
}

void NoiseQVM::initGates(rapidjson::Document & doc)
{
    if (!doc.HasMember("gates"))
    {

        QCERR("doc do not include gates");
        throw invalid_argument("doc do not include gates");
    }

    auto doc_iter = doc.FindMember("gates");
    auto & gates = (*doc_iter).value;
    if (!gates.IsArray())
    {
        QCERR("gates is not array");
        throw invalid_argument("gates is not array");
    }
    m_gates_matrix.resize(0);

    for (auto first_layer_iter = gates.Begin();
        first_layer_iter != gates.End();
        first_layer_iter++)
    {
        auto &second_layer = first_layer_iter;
        vector<string> temp;
        if (!first_layer_iter->IsArray())
        {
            if (!first_layer_iter->IsString())
            {
                QCERR("first_layer_iter is not string or array");
                throw invalid_argument("first_layer_iter is not  string or array");
            }
            temp.push_back(first_layer_iter->GetString());
        }
        else
        {
            for (auto second_layer_iter = second_layer->Begin();
                second_layer_iter != second_layer->End();
                second_layer_iter++)
            {
                if (!second_layer_iter->IsString())
                {
                    QCERR("first_layer_iter is not string or array");
                    throw invalid_argument("first_layer_iter is not  string or array");
                }
                std::string string_temp(second_layer_iter->GetString());
                temp.push_back(string_temp);
            }
        }
        m_gates_matrix.push_back(temp);
    }
    m_valid_gates_matrix.reserve(m_gates_matrix.size());
}

void NoiseQVM::init(rapidjson::Document & doc)
{
    if (!doc.HasMember("noisemodel") && (!doc.HasMember("gates")))
    {
        init();
    }
    else
    {
        try
        {
            if (doc.HasMember("gates"))
            {
                initGates(doc);
            }
            m_doc.CopyFrom(doc, m_doc.GetAllocator());
            _start();
            _getValidGatesMatrix();
            _pGates = new NoisyCPUImplQPU(doc);
            _ptrIsNull(_pGates, "NoisyCPUImplQPU");
        }
        catch (const std::exception&e)
        {
            finalize();
            QCERR(e.what());
            throw init_fail(e.what());
        }
    }

}

void NoiseQVM::run(QProg & prog)
{
    try
    {
        /*vector<vector<int>> adjacent_matrixes;
        TransformDecomposition traversal_vec(m_valid_gates_matrix, m_gates_matrix, adjacent_matrixes);
        traversal_vec.TraversalOptimizationMerge(dynamic_cast<QNode *>(&prog));*/
        _pGates->initState(0, 1, _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit());

        QProgExecution prog_exec;
        TraversalConfig config;

        prog_exec.execute(prog.getImplementationPtr(), nullptr, config, _pGates);

        std::map<string, bool>result;
        prog_exec.get_return_value(result);

        /* aiter has been used in line 120 */
        for (auto aiter : result)
        {
            _QResult->append(aiter);
        }
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw run_fail(e.what());
    }

}

std::string NoiseQVM::_ResultToBinaryString(std::vector<ClassicalCondition>& vCBit, QResult* qresult)
{
    string sTemp;
    if (nullptr == qresult)
    {
        QCERR("_QResult is null");
        throw qvm_attributes_error("_QResult is null");
    }
    auto resmap = qresult->getResultMap();
    for (auto c : vCBit)
    {
        auto cbit = c.getExprPtr()->getCBit();
        if (nullptr == cbit)
        {
            QCERR("vcbit is error");
            throw runtime_error("vcbit is error");
        }
        if (resmap[cbit->getName()])
        {
            sTemp.push_back('1');
        }
        else
        {
            sTemp.push_back('0');
        }
    }
    return sTemp;

}
map<string, size_t> NoiseQVM::
runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, int shots)
{
	rapidjson::Document doc;
	doc.Parse("{}");
	auto &alloc = doc.GetAllocator();
	doc.AddMember("shots", shots, alloc);
	return runWithConfiguration(qProg, vCBit, doc);
}

map<string, size_t> NoiseQVM::
runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, rapidjson::Document & param)
{
    map<string, size_t> mResult;
    if (!param.HasMember("shots"))
    {
        QCERR("OriginCollection don't  have shots");
        throw run_fail("runWithConfiguration param don't  have shots");
    }
    size_t shots = 0;
    if (param["shots"].IsUint64())
    {
        shots = param["shots"].GetUint64();
    }
    else
    {
        QCERR("shots data type error");
        throw run_fail("shots data type error");
    }

#ifndef USE_MPI
    for (size_t i = 0; i < shots; i++)
    {
        run(qProg);
        string sResult = _ResultToBinaryString(vCBit, _QResult);
        if (mResult.find(sResult) == mResult.end())
        {
            mResult[sResult] = 1;
        }
        else
        {  
            mResult[sResult] += 1;
        }
    }
#else
    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<int> result(1 << vCBit.size(), 0);
    auto _run=[&]()
    {
        run(qProg);
        string sResult;
        auto str_result = _QResult->getResultMap();

        for(auto a :str_result)
        {
            if(a.second)
            {
                sResult.push_back('1');
            }
            else
            {
                sResult.push_back('0');
            }
        }

        int index = 0;
        for (size_t i = 0; i < sResult.size(); ++i)
        {
            index += (sResult[sResult.size() - i - 1] != '0') << i;
        }
        result[index] += 1;
    };

    if (size < shots)
    {
        int val = shots / size;

        for (int i = 0; i < val; i++)
        {
            _run();
        }

        if (rank < shots%size)
        {
            _run();
        }
    }
    else
    {
        if (rank < shots)  
        {
            _run();
        }
    }

    vector<int> res(1 << vCBit.size(), 0);
    MPI_Reduce(&result[0], &res[0], 1 << vCBit.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&res[0], 1 << vCBit.size(), MPI_INT, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < res.size(); ++i)
    {
        if (res[i])
        {
            mResult.insert(make_pair(integerToBinary(i, vCBit.size()), res[i]));
        }
    }
    MPI_Finalize();

#endif
    return mResult;
}



REGISTER_QUANTUM_MACHINE(NoiseQVM);
