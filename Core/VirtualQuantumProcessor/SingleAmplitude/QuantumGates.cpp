#include "Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"

#define SQRT2 1.4142135623731
#define PI  3.141592654
static qsize_t edge_count = 0;

static void addSingleGateNonDiagonalVerticeAndEdge(QuantumProgMap & prog_map,
                                                   qstate_t &gate_tensor,
                                                   qsize_t qubit)
{
    EdgeMap * edge_map = prog_map.getEdgeMap();
    ComplexTensor temp(2, gate_tensor);

    VerticeMatrix * vertice_matrix = prog_map.getVerticeMatrix();
    auto vertice_id = vertice_matrix->getQubitVerticeLastID(qubit);
    auto vertice_id2 = vertice_matrix->addVertice(qubit);

    vector<pair<qsize_t, qsize_t>> contect_vertice = 
                       { { qubit,vertice_id2 },
                         { qubit,vertice_id } };
    edge_count++;
    Edge edge(1, temp, contect_vertice);
    edge_map->insert(pair<qsize_t,Edge>(edge_count, edge));
    vertice_matrix->addContectEdge(qubit, vertice_id, edge_count);
    vertice_matrix->addContectEdge(qubit, vertice_id2, edge_count);
}

static void addSingleGateDiagonalVerticeAndEdge(QuantumProgMap & prog_map,
    qstate_t &gate_tensor,
    qsize_t qubit)
{
    EdgeMap * edge_map = prog_map.getEdgeMap();
    ComplexTensor temp(1, gate_tensor);

    VerticeMatrix * vertice_matrix = prog_map.getVerticeMatrix();
    auto vertice_id = vertice_matrix->getQubitVerticeLastID(qubit);

    vector<pair<qsize_t, qsize_t>> contect_vertice =
    { { qubit,vertice_id } };
    edge_count++;
    Edge edge(1, temp, contect_vertice);
    edge_map->insert(pair<qsize_t, Edge>(edge_count, edge));
    vertice_matrix->addContectEdge(qubit, vertice_id, edge_count);
}




static void addDoubleDiagonalGateVerticeAndEdge(QuantumProgMap & prog_map,
                                                qstate_t &gate_tensor,
                                                qsize_t qubit1,
                                                qsize_t qubit2)
{
    EdgeMap * edge_map = prog_map.getEdgeMap();
    ComplexTensor temp(2, gate_tensor);
    VerticeMatrix * vertice_matrix = prog_map.getVerticeMatrix();
    auto vertice_qubit1_id = vertice_matrix->getQubitVerticeLastID(qubit1);

    auto vertice_qubit2_id = vertice_matrix->getQubitVerticeLastID(qubit2);

    
    vector<pair<qsize_t, qsize_t>> contect_vertice =
                { { qubit1,vertice_qubit1_id },
                  { qubit2,vertice_qubit2_id } };

    edge_count++;
    Edge edge(2, temp, contect_vertice);
    edge_map->insert(pair<qsize_t, Edge>(edge_count, edge));
    vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id, edge_count);
    vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id, edge_count);
}



static void addDoubleNonDiagonalGateVerticeAndEdge(QuantumProgMap & prog_map,
                                                   qstate_t &gate_tensor,
                                                   qsize_t qubit1,
                                                   qsize_t qubit2)
{
    EdgeMap * edge_map = prog_map.getEdgeMap();
    ComplexTensor temp(4, gate_tensor);
    VerticeMatrix * vertice_matrix = prog_map.getVerticeMatrix();
    auto vertice_qubit1_id = vertice_matrix->getQubitVerticeLastID(qubit1);
    auto vertice_qubit1_id2 = vertice_matrix->addVertice(qubit1);

    auto vertice_qubit2_id = vertice_matrix->getQubitVerticeLastID(qubit2);
    auto vertice_qubit2_id2 = vertice_matrix->addVertice(qubit2);

    vector<pair<qsize_t, qsize_t>> contect_vertice 
        = { { qubit1,vertice_qubit1_id },
            { qubit2,vertice_qubit2_id },
            { qubit1,vertice_qubit1_id2 },
            { qubit2,vertice_qubit2_id2 } };
    edge_count++;
    Edge edge(2, temp, contect_vertice);
    edge_map->insert(pair<qsize_t, Edge>(edge_count, edge));
    vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id, edge_count);
    vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id2, edge_count);

    vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id, edge_count);
    vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id2, edge_count);
}


void H_Gate(QuantumProgMap & prog_map,qsize_t qubit, bool isDagger)
{
    qstate_t gate_tensor(4, 0);
    gate_tensor[0] =  1 / SQRT2;
    gate_tensor[1] =  1 / SQRT2;
    gate_tensor[2] =  1 / SQRT2;
    gate_tensor[3] = -1 / SQRT2;
    addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void X_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
    qstate_t gate_tensor(4, 0);
    gate_tensor[1] = 1;
    gate_tensor[2] = 1;
    addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void RX_Gate(QuantumProgMap & prog_map, qsize_t qubit,double angle, bool isDagger)
{
	qstate_t gate_tensor(4, 0);
	gate_tensor[0] = cos(angle / 2);
	if (isDagger)
	{
		gate_tensor[1].imag(1 * sin(angle / 2));
		gate_tensor[2].imag(1 * sin(angle / 2));
	}
	else
	{
		gate_tensor[1].imag(-1 * sin(angle / 2));
		gate_tensor[2].imag(-1 * sin(angle / 2));
	}
	gate_tensor[3] = cos(angle / 2);
	addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void Y_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
    qstate_t gate_tensor(4, 0);
    gate_tensor[1].imag(-1) ;
    gate_tensor[2].imag(1);
    addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void RY_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger)
{
	qstate_t gate_tensor(4, 0);
	gate_tensor[0] = cos(angle / 2);
	if (isDagger)
	{
		gate_tensor[1] = sin(angle / 2);
		gate_tensor[2] = -sin(angle / 2);
	}
	else
	{
		gate_tensor[1] = -sin(angle / 2);
		gate_tensor[2] = sin(angle / 2);
	}
	gate_tensor[3] = cos(angle / 2);
	addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void X1_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
	qstate_t gate_tensor(4, 0);
	gate_tensor[0] = 1 / SQRT2;
	if (isDagger)
	{
		gate_tensor[1] = qcomplex_data_t(0, 1 / SQRT2);
		gate_tensor[2] = qcomplex_data_t(0, 1 / SQRT2);
	}
	else
	{
		gate_tensor[1] = qcomplex_data_t(0, -1 / SQRT2);
		gate_tensor[2] = qcomplex_data_t(0, -1 / SQRT2);
	}
	gate_tensor[3] = 1 / SQRT2;
	addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void Y1_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
	qstate_t gate_tensor(4, 0);
	gate_tensor[0] = 1 / SQRT2;
	if (isDagger)
	{
		gate_tensor[1] =  1 / SQRT2;
		gate_tensor[2] = -1 / SQRT2;
	}
	else
	{
		gate_tensor[1] = -1 / SQRT2;
		gate_tensor[2] =  1 / SQRT2;
	}
	gate_tensor[3] = 1 / SQRT2;
	addSingleGateNonDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}



void Z_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
    qstate_t gate_tensor(2, 0);
    gate_tensor[0] = 1;
    gate_tensor[1] = -1;
    addSingleGateDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void RZ_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger)
{
	qstate_t gate_tensor(2, 0);
	gate_tensor[0].real(cos(angle / 2));
	gate_tensor[1].real(cos(angle / 2));
	if (isDagger)
	{
		gate_tensor[0].imag(1 * sin(angle / 2));
		gate_tensor[1].imag(-1 * sin(angle / 2));
	}
	else
	{
		gate_tensor[0].imag(-1 * sin(angle / 2));
		gate_tensor[1].imag(1 * sin(angle / 2));
	}

	addSingleGateDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void U1_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger)
{
    qstate_t gate_tensor(2, 0);
    gate_tensor[0] = 1;
    if (isDagger)
    {
        gate_tensor[1].real(cos(angle));
        gate_tensor[1].imag(sin(angle));
    }
    else
    {
        gate_tensor[1].real(cos(-angle));
        gate_tensor[1].imag(sin(-angle));
    }

    addSingleGateDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void Z1_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
	qstate_t gate_tensor(2, 0);
	if (isDagger)
	{
		gate_tensor[0] = qcomplex_data_t(1 / SQRT2,  1 / SQRT2);
		gate_tensor[1] = qcomplex_data_t(1 / SQRT2, -1 / SQRT2);
	}
	else
	{
		gate_tensor[0] = qcomplex_data_t(1 / SQRT2, -1 / SQRT2);
		gate_tensor[1] = qcomplex_data_t(1 / SQRT2,  1 / SQRT2);
	}
	addSingleGateDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void CZ_Gate(QuantumProgMap & prog_map,qsize_t qubit1,qsize_t qubit2, bool isDagger)
{
    qstate_t gate_tensor(4, 1);
    gate_tensor[0] =  1 ;
    gate_tensor[1] =  1;
    gate_tensor[2] =  1;
    gate_tensor[3] = -1;
    addDoubleDiagonalGateVerticeAndEdge(prog_map,
                                        gate_tensor,
                                        qubit1,
                                        qubit2);
}

void CNOT_Gate(QuantumProgMap &prog_map,qsize_t qubit1,qsize_t qubit2, bool isDagger)
{
    qstate_t gate_tensor(16, 0);
    gate_tensor[0] =  1;
    gate_tensor[5] =  1;
    gate_tensor[11] = 1;
    gate_tensor[14] = 1;
    addDoubleNonDiagonalGateVerticeAndEdge(prog_map,
                                           gate_tensor,
                                           qubit1,
                                           qubit2);
}

void ISWAP_Gate(QuantumProgMap &prog_map, qsize_t qubit1, qsize_t qubit2, bool isDagger)
{
    qstate_t gate_tensor(16, 0);
    gate_tensor[0] = 1;
    gate_tensor[15] = 1;
    if (isDagger)
    {
        gate_tensor[6].imag(1);
        gate_tensor[9].imag(1);
    } 
    else
    {
        gate_tensor[6].imag(-1);
        gate_tensor[9].imag(-1);
    }
    addDoubleNonDiagonalGateVerticeAndEdge(prog_map,
        gate_tensor,
        qubit1,
        qubit2);
}


void SQISWAP_Gate(QuantumProgMap &prog_map, qsize_t qubit1, qsize_t qubit2, bool isDagger)
{
    qstate_t gate_tensor(16, 0);
    gate_tensor[0] = 1;
    gate_tensor[5] = 1 / SQRT2;
    gate_tensor[10] = 1 / SQRT2;
    gate_tensor[15] = 1;
    if (isDagger)
    {
        gate_tensor[6].imag(-1 / SQRT2);
        gate_tensor[9].imag(-1 / SQRT2);
    }
    else
    {
        gate_tensor[6].imag(1 / SQRT2);
        gate_tensor[9].imag(1 / SQRT2);
    }
    addDoubleNonDiagonalGateVerticeAndEdge(prog_map,
        gate_tensor,
        qubit1,
        qubit2);
}

void CR_Gate(QuantumProgMap & prog_map,qsize_t qubit1,qsize_t qubit2, double angle, bool isDagger)
{
	qstate_t gate_tensor(4, 1);
	gate_tensor[0] = 1;
	gate_tensor[1] = 1;
	gate_tensor[2] = 1;
	if (isDagger)
	{
		gate_tensor[3].real(cos(angle));
		gate_tensor[3].imag(-1 * sin(angle));
	}
	else
	{
		gate_tensor[3].real(cos(angle));
		gate_tensor[3].imag(1 * sin(angle));
	}
	addDoubleDiagonalGateVerticeAndEdge(prog_map,
		gate_tensor,
		qubit1,
		qubit2);
}

void T_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
	qstate_t gate_tensor(2, 0);
	gate_tensor[0] = 1;
	if (isDagger)
	{
		gate_tensor[1].real(cos(PI / 4));
		gate_tensor[1].imag(-sin(PI / 4));
	}
	else
	{
		gate_tensor[1].real(cos(PI / 4));
		gate_tensor[1].imag(sin(PI / 4));
	}
	addSingleGateDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}

void S_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger)
{
	qstate_t gate_tensor(2, 0);
	if (isDagger)
	{
		gate_tensor[0] = 1;
		gate_tensor[1].imag(-1);
	}
	else
	{
		gate_tensor[0] = 1;
		gate_tensor[1].imag(1);
	}
	addSingleGateDiagonalVerticeAndEdge(prog_map, gate_tensor, qubit);
}
#include <sstream>
using std::stringstream;
bool integerToBinary(size_t  number, stringstream & ssRet, int ret_len)
{
	unsigned int index;

	for (int i = ret_len -1; i > -1; i--)
	{
		ssRet << ((number >> i) & 1);
	}
	return true;
}



