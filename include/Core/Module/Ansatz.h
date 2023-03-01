#ifndef ANSATZ_H
#define ANSATZ_H

#include <Core/Core.h>
#include "Core/Module/Ansatz.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

QPANDA_BEGIN

using Ansatz = std::vector<AnsatzGate>;
using Thetas = std::vector<double>;

class AnsatzCircuit : public TraversalInterface<>
{
public:
    QCircuit qcircuit();

    AnsatzCircuit();
    AnsatzCircuit(QGate&);
    AnsatzCircuit(AnsatzGate&);
    AnsatzCircuit(Ansatz&, const Thetas& = {});
    AnsatzCircuit(QCircuit&, const Thetas& = {});
    AnsatzCircuit(const AnsatzCircuit&, const Thetas& = {});

    void insert(QGate&);
    void insert(Ansatz&);
    void insert(QCircuit&);
    void insert(AnsatzGate&);
    void insert(AnsatzCircuit&, const Thetas& = {});

    template <typename T>
    AnsatzCircuit& operator << (T& node)
    {
        insert(node);
        return (*this);
    }

    void set_thetas(const Thetas&);

    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);

    Ansatz get_ansatz_list() const  { return m_ansatz; }
    Thetas get_thetas_list() const  { return m_thetas; }

private:

    Ansatz m_ansatz;
    Thetas m_thetas;

    //std::map<size_t, size_t> m_theta_map;
};

QPANDA_END

#endif // ANSATZ_H