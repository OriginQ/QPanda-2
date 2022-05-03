#include "Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

USING_QPANDA

void H_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(4, 0);
	gate_tensor[0] = 1 / SQRT2;
	gate_tensor[1] = 1 / SQRT2;
	gate_tensor[2] = 1 / SQRT2;
	gate_tensor[3] = -1 / SQRT2;
}

void X_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(4, 0);
	gate_tensor[1] = 1;
	gate_tensor[2] = 1;
}

void RX_Gate(qstate_t& gate_tensor, double angle, bool isDagger)
{
	gate_tensor.assign(4, 0);
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
}

void Y_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(4, 0);
	gate_tensor[1].imag(-1);
	gate_tensor[2].imag(1);
}

void RY_Gate(qstate_t& gate_tensor, double angle, bool isDagger)
{
	gate_tensor.assign(4, 0);
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
}

void X1_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(4, 0);
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
}

void Y1_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(4, 0);
	gate_tensor[0] = 1 / SQRT2;
	if (isDagger)
	{
		gate_tensor[1] = 1 / SQRT2;
		gate_tensor[2] = -1 / SQRT2;
	}
	else
	{
		gate_tensor[1] = -1 / SQRT2;
		gate_tensor[2] = 1 / SQRT2;
	}
	gate_tensor[3] = 1 / SQRT2;
}

void Z_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(2, 0);
	gate_tensor[0] = 1;
	gate_tensor[1] = -1;
}

void RZ_Gate(qstate_t& gate_tensor, double angle, bool isDagger)
{
	gate_tensor.assign(2, 0);
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
}

void U1_Gate(qstate_t& gate_tensor, double angle, bool isDagger)
{
	gate_tensor.assign(2, 0);
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
}

void U2_Gate(qstate_t& gate_tensor, double phi, double lambda, bool isDagger)
{
	gate_tensor.assign(4, 0);

	auto coefficient = SQRT2 / 2;
	gate_tensor[0] = 1 * coefficient;
	gate_tensor[1].real(-cos(lambda) * coefficient);
	gate_tensor[1].imag(-sin(lambda) * coefficient);

	gate_tensor[2].real(cos(phi) * coefficient);
	gate_tensor[2].imag(sin(phi) * coefficient);
	gate_tensor[3].real(cos(phi + lambda) * coefficient);
	gate_tensor[3].imag(sin(phi + lambda) * coefficient);

	if (isDagger)
	{
		qcomplex_data_t temp;
		temp = gate_tensor[1];
		gate_tensor[1] = gate_tensor[2];
		gate_tensor[2] = temp; 
		for (size_t i = 0; i < 4; i++)
			gate_tensor[i] = qcomplex_data_t(gate_tensor[i].real(), -gate_tensor[i].imag());
	}
}

void U3_Gate(qstate_t& gate_tensor, double theta, double phi, double lambda, bool isDagger)
{
	gate_tensor.assign(4, 0);
	gate_tensor[0] = std::cos(theta / 2);
	gate_tensor[1] = -std::exp(qcomplex_t(0, lambda)) * std::sin(theta / 2);
	gate_tensor[2] = std::exp(qcomplex_t(0, phi)) * std::sin(theta / 2);
	gate_tensor[3] = std::exp(qcomplex_t(0, phi + lambda)) * std::cos(theta / 2);

	if (isDagger)
	{
		qcomplex_data_t temp;
		temp = gate_tensor[1];
		gate_tensor[1] = gate_tensor[2];
		gate_tensor[2] = temp;
		for (size_t i = 0; i < 4; i++)
			gate_tensor[i] = qcomplex_data_t(gate_tensor[i].real(), -gate_tensor[i].imag());
	}
}


void U4_Gate(qstate_t& gate_tensor, double alpha, double beta, double gamma, double delta, bool isDagger)
{
	gate_tensor.assign(4, 0);
	gate_tensor[0] = qcomplex_t(cos(alpha - beta / 2 - delta / 2) * cos(gamma / 2),
		sin(alpha - beta / 2 - delta / 2) * cos(gamma / 2));
	gate_tensor[1] = qcomplex_t(-cos(alpha - beta / 2 + delta / 2) * sin(gamma / 2),
		-sin(alpha - beta / 2 + delta / 2) * sin(gamma / 2));
	gate_tensor[2] = qcomplex_t(cos(alpha + beta / 2 - delta / 2) * sin(gamma / 2),
		sin(alpha + beta / 2 - delta / 2) * sin(gamma / 2));
	gate_tensor[3] = qcomplex_t(cos(alpha + beta / 2 + delta / 2) * cos(gamma / 2),
		sin(alpha + beta / 2 + delta / 2) * cos(gamma / 2));

	if (isDagger)
	{
		qcomplex_data_t temp;
		temp = gate_tensor[1];
		gate_tensor[1] = gate_tensor[2];
		gate_tensor[2] = temp;
		for (size_t i = 0; i < 4; i++)
			gate_tensor[i] = qcomplex_data_t(gate_tensor[i].real(), -gate_tensor[i].imag());
	}
}

void Z1_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(2, 0);
	if (isDagger)
	{
		gate_tensor[0] = qcomplex_data_t(1 / SQRT2, 1 / SQRT2);
		gate_tensor[1] = qcomplex_data_t(1 / SQRT2, -1 / SQRT2);
	}
	else
	{
		gate_tensor[0] = qcomplex_data_t(1 / SQRT2, -1 / SQRT2);
		gate_tensor[1] = qcomplex_data_t(1 / SQRT2, 1 / SQRT2);
	}
}

void CZ_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(4, 1);
	gate_tensor[0] = 1;
	gate_tensor[1] = 1;
	gate_tensor[2] = 1;
	gate_tensor[3] = -1;
}

void CNOT_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(16, 0);
	gate_tensor[0] = 1;
	gate_tensor[5] = 1;
	gate_tensor[11] = 1;
	gate_tensor[14] = 1;
}

void SWAP_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(16, 0);
	gate_tensor[0] = 1;
	gate_tensor[6] = 1;
	gate_tensor[9] = 1;
	gate_tensor[15] = 1;
}

void ISWAP_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(16, 0);
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
}

void SQISWAP_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(16, 0);
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
}

void CR_Gate(qstate_t& gate_tensor, double angle, bool isDagger)
{
	gate_tensor.assign(4, 1);
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
}

void T_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(2, 0);
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
}

void S_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(2, 0);
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
}

void TOFFOLI_Gate(qstate_t& gate_tensor, bool isDagger)
{
	gate_tensor.assign(64, 0);
	gate_tensor[0] = 1;
	gate_tensor[9] = 1;
	gate_tensor[18] = 1;
	gate_tensor[27] = 1;
	gate_tensor[36] = 1;
	gate_tensor[45] = 1;
	gate_tensor[55] = 1;
	gate_tensor[62] = 1;
}


