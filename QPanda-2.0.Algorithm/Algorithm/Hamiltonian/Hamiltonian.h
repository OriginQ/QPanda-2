/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "QPanda.h"
#include"asa047.h"
class GraphEdge
{
public:
	GraphEdge(Qubit*, Qubit*, double);
	Qubit* first;
	Qubit* second;
	double weight;
};

class Hamiltonian
{
public:
	virtual QCircuit HamiltonianSimulation() = 0;
};

class PauliZSimulation :public Hamiltonian
{
public:
	//exp(-i*theta(sigma_j1 sigma_j2....))
	static QCircuit  HamiltonianSimulation(vector<Qubit*>,double) ;
	friend QCircuit  QAOA(vector<Qubit*> qvec, vector<GraphEdge> graphstate,double,double);
private:

};

vector<GraphEdge>& rigetti(vector<Qubit*>);
QProg rigettimaxcut(vector<Qubit*> , vector<CBit*> ,double,double);



double rigettiQaoaTest(double , double, ofstream&);
double rigettiQaoaTestv2(double gamma, double beta);
double rigettiQaoaTestv3(double gamma, double beta,int shots);
double rigettiQaoaTestv4(double x[]);
double rigettiQaoaTestv1(double gamma, double beta, int shots, ofstream & examplefile);
//double optimize(int);
void qaoaopall();
typedef vector<double> linspace_t;

double optimize(linspace_t beta_vec, linspace_t gamma_vec, int shots);
int printcircuit( QCircuit);
double weightSum(vector<Qubit*>&, vector<bool>&);

inline vector<double> linspace(double start, double stop, size_t stepn)
{
    double step = (stop - start) / stepn;
    vector<double> vec_linspace;
    for (auto i = 0; i <= stepn; i++)
    {
        vec_linspace.push_back(start + step * i);
    }
    return vec_linspace;
}

inline size_t linspace_count(linspace_t ls, double value)
{
     auto temp = (value - ls[0]) / (ls[1] - ls[0]);
	 return (size_t)temp;
}

#endif