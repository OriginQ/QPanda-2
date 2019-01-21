#ifndef TEST_H
#define TEST_H

#include "QPanda.h"
USING_QPANDA
void HHL_Algorithm1();
void entangle();
void controlandDagger();
QProg bell(Qubit* a, Qubit * b);
int Grover(int target);
bool DJalgorithm();
bool HelloWorld();
void ifwhile();

void testQPauliOperator();
void testQAOA();

#endif