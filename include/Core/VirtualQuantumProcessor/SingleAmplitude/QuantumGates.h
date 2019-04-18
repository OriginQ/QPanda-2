#ifndef QUANTUMGATES_H
#define QUANTUMGATES_H
#include "Core/VirtualQuantumProcessor/SingleAmplitude/TensorEngine.h"
#include <complex>
using std::complex;

void H_Gate(QuantumProgMap &prog_map, qsize_t qubit, bool isDagger);
void T_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void S_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void X_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void Y_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void Z_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void X1_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void Y1_Gate(QuantumProgMap & prog_map, qsize_t qubit, bool isDagger);
void Z1_Gate(QuantumProgMap &prog_map, qsize_t qubit, bool isDagger);

void RX_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger);
void RY_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger);
void RZ_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger);
void U1_Gate(QuantumProgMap & prog_map, qsize_t qubit, double angle, bool isDagger);

void CZ_Gate(QuantumProgMap &prog_map,qsize_t qubit1,qsize_t qubit2,bool isDagger);
void CNOT_Gate(QuantumProgMap &prog_map, qsize_t qubit1, qsize_t qubit2, bool isDagger);
void ISWAP_Gate(QuantumProgMap &prog_map, qsize_t qubit1, qsize_t qubit2, bool isDagger);
void SQISWAP_Gate(QuantumProgMap &prog_map, qsize_t qubit1, qsize_t qubit2, bool isDagger);

void CR_Gate(QuantumProgMap &prog_map,qsize_t qubit1,qsize_t qubit2,double angle, bool isDagger);

#endif // !QUANTUMGATES_H
