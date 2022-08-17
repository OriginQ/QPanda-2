#ifndef QUANTUMGATES_H
#define QUANTUMGATES_H
#include "Core/VirtualQuantumProcessor/SingleAmplitude/TensorEngine.h"

void H_Gate(qstate_t& gate_tensor, bool isDagger);
void T_Gate(qstate_t& gate_tensor, bool isDagger);
void S_Gate(qstate_t& gate_tensor, bool isDagger);
void X_Gate(qstate_t& gate_tensor, bool isDagger);
void Y_Gate(qstate_t& gate_tensor, bool isDagger);
void Z_Gate(qstate_t& gate_tensor, bool isDagger);
void X1_Gate(qstate_t& gate_tensor, bool isDagger);
void Y1_Gate(qstate_t& gate_tensor, bool isDagger);
void Z1_Gate(qstate_t& gate_tensor, bool isDagger);

void RX_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void RY_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void RZ_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void U1_Gate(qstate_t& gate_tensor, double angle, bool isDagger);

void U2_Gate(qstate_t& gate_tensor, double phi, double lambda, bool isDagger);
void U3_Gate(qstate_t& gate_tensor, double theta, double phi, double lambda, bool isDagger);
void U4_Gate(qstate_t& gate_tensor, double alpha, double beta, double gamma, double delta, bool isDagger);

void CZ_Gate(qstate_t& gate_tensor, bool isDagger);
void CNOT_Gate(qstate_t& gate_tensor, bool isDagger);
void ISWAP_Gate(qstate_t& gate_tensor, bool isDagger);
void SQISWAP_Gate(qstate_t& gate_tensor, bool isDagger);
void SWAP_Gate(qstate_t& gate_tensor, bool isDagger);

void TOFFOLI_Gate(qstate_t& gate_tensor, bool isDagger);

void CR_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void RXX_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void RYY_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void RZZ_Gate(qstate_t& gate_tensor, double angle, bool isDagger);
void RZX_Gate(qstate_t& gate_tensor, double angle, bool isDagger);


#endif // !QUANTUMGATES_H
