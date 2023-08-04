#pragma once

#include "Core/Core.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/CommunicationProtocol/ProtocolDefinitions.h"

QPANDA_BEGIN

class CommProtocolEncode : public TraversalInterface<QCircuitConfig&>
{

public:

    CommProtocolEncode() {};

    void encode(QProg& prog);
    void encode(CommProtocolConfig config);
    void encode_crc();

    std::vector<char> convert_to_char();
    const std::vector<uint64_t>& data() const { return m_protocol_encode_data; }

    static std::map<GateType, OperationType> m_encode_gate_map;

public:

    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumReset>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQNoiseNode>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQDebugNode>, std::shared_ptr<QNode>, QCircuitConfig &config);

private:

    Qnum convert_params(prob_vec params)
    {
        Qnum params_data(params.size());

        for (size_t i = 0; i < params.size(); i++)
        {
            auto param_mod = std::fmod(params[i] / (2 * PI), 1.0);
            if (param_mod < 0)
                param_mod += 1;
            auto data = std::round(param_mod * (1ull << 16));
            params_data[i] = static_cast<size_t>(data);
        }

        return params_data;
    }

    template <typename T>
    void node_encode(const T& node)
    {
        auto data = node.encode();
        m_protocol_encode_data.insert(m_protocol_encode_data.end(), data.begin(), data.end());
    }

    std::vector<uint64_t> m_protocol_encode_data;

};

std::vector<char> comm_protocol_encode(std::vector<QProg> prog_list, CommProtocolConfig config = {});
std::vector<char> comm_protocol_encode(QProg prog, CommProtocolConfig config = {});

QPANDA_END
