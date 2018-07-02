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

#include "Hamiltonian.h"
#include<vector>
#include<fstream>
#include<bitset>
#include <thread>
#include <iomanip>

QCircuit  PauliZSimulation::HamiltonianSimulation(vector<Qubit*> qvec, double theta)
{
	QCircuit  pauliZ = CreateEmptyCircuit();
	for (auto iter = qvec.begin(); iter != qvec.end()-1 ; iter++)
	{
		pauliZ << CNOT(*iter, *(qvec.end()-1));
	}
	pauliZ << RZ(*(qvec.end()-1), 2*theta);
	for (auto iter = qvec.begin(); iter != qvec.end() - 1; iter++)
	{
		pauliZ << CNOT(*iter, *(qvec.end()-1));
	}
	return pauliZ;
}
QCircuit  QAOA(vector<Qubit*> qvec, vector<GraphEdge> graphstate,double gamma,double beta)
{
	QCircuit  qaoacircuit = CreateEmptyCircuit();
	for (auto iter = qvec.begin(); iter != qvec.end(); iter++)
	{
		qaoacircuit << H(*iter);
	}
	//qaoacircuit.
	for (auto iter = graphstate.begin(); iter != graphstate.end(); iter++)
	{
		//global phase
		//qaoacircuit << RX(iter->first)<<RZ(iter->first,0.5*gamma*iter->weight)
		//	<< RX(iter->first) << RZ(iter->first, 0.5*gamma*iter->weight);
		qaoacircuit << CNOT(iter->first, iter->second)
			<< RZ(iter->second, 2*gamma*(iter->weight))
			<< CNOT(iter->first, iter->second);
	}
	for (auto iter = qvec.begin(); iter != qvec.end(); iter++)
	{
		qaoacircuit << RX(*iter,2*beta);
	}
	return qaoacircuit;
}
int printcircuit( QCircuit qcir)
{
	//ofstream outFile;
	//char automobile[50];
	//int year;
	//double a_price;
	//double b_price;

	////创建并且打开文件(没有则创建，有则覆盖)
	//outFile.open("qcircuit.txt", ios::out);
	//cin.getline(automobile, 50);

	//cin >> year;
	//cout << "Enter the origin asking price:";

	//if (!qcir.getHeadNodeIter)
	//{
	//	throw exception("blank circuit");
	//}
	//for (auto iter = qcir.getHeadNodeIter(); iter != qcir.getLastNodeIter; iter++)
	//{
	//	
	//}
	////打开成功　也可以适用outFile.good()
	//if (outFile) {
	//	outFile.precision(2);
	//	outFile << fixed;
	//	outFile.setf(ios_base::showpoint);//展示小数点

	//	outFile << "make the model:" << automobile << endl;
	//	outFile << "year:" << year << endl;
	//	outFile << "a_price:" << a_price << endl;
	//	outFile << "b_price" << b_price << endl;

	//}

	////关闭文本文件输出流
	//outFile.close();

	return 0;
}
//max=10.48
vector<GraphEdge>& rigetti(vector<Qubit*> qvec)
{
	vector<GraphEdge> *maxcut=new vector<GraphEdge>;
	GraphEdge* wetseq0 = new GraphEdge(qvec[0], qvec[5], 0.18);
	(*maxcut).push_back(*wetseq0);
	GraphEdge* wetseq1 = new GraphEdge(qvec[0], qvec[6], 0.49);
	(*maxcut).push_back(*wetseq1);
	GraphEdge* wetseq2 = new GraphEdge(qvec[1], qvec[6], 0.59);
	(*maxcut).push_back(*wetseq2);
	GraphEdge* wetseq3 = new GraphEdge(qvec[1], qvec[7], 0.44);
	(*maxcut).push_back(*wetseq3);
	GraphEdge* wetseq4 = new GraphEdge(qvec[2], qvec[7], 0.56);
	(*maxcut).push_back(*wetseq4);
	GraphEdge* wetseq5 = new GraphEdge(qvec[2], qvec[8], 0.63);
	(*maxcut).push_back(*wetseq5);
	GraphEdge* wetseq6 = new GraphEdge(qvec[4], qvec[9], 0.43);
	(*maxcut).push_back(*wetseq6);
	GraphEdge* wetseq7 = new GraphEdge(qvec[5], qvec[10], 0.23);
	(*maxcut).push_back(*wetseq7);
	GraphEdge* wetseq8 = new GraphEdge(qvec[6], qvec[11], 0.64);
	(*maxcut).push_back(*wetseq8);
	GraphEdge* wetseq9 = new GraphEdge(qvec[7], qvec[12], 0.60);
	(*maxcut).push_back(*wetseq9);
	GraphEdge* wetseq10 = new GraphEdge(qvec[8], qvec[13], 0.36);
	(*maxcut).push_back(*wetseq10);
	GraphEdge* wetseq11 = new GraphEdge(qvec[9], qvec[14], 0.52);
	(*maxcut).push_back(*wetseq11);
	GraphEdge* wetseq12 = new GraphEdge(qvec[10], qvec[15], 0.40);
	(*maxcut).push_back(*wetseq12);
	GraphEdge* wetseq13 = new GraphEdge(qvec[10], qvec[16], 0.41);
	(*maxcut).push_back(*wetseq13);
	GraphEdge* wetseq14 = new GraphEdge(qvec[11], qvec[16], 0.57);
	(*maxcut).push_back(*wetseq14);
	GraphEdge* wetseq15 = new GraphEdge(qvec[11], qvec[17], 0.50);
	(*maxcut).push_back(*wetseq15);
	GraphEdge* wetseq16 = new GraphEdge(qvec[12], qvec[17], 0.71);
	(*maxcut).push_back(*wetseq16);
	GraphEdge* wetseq17 = new GraphEdge(qvec[12], qvec[18], 0.40);
	(*maxcut).push_back(*wetseq17);
	GraphEdge* wetseq18 = new GraphEdge(qvec[13], qvec[18], 0.72);
	(*maxcut).push_back(*wetseq18);
	GraphEdge* wetseq19 = new GraphEdge(qvec[13], qvec[19], 0.81);
	(*maxcut).push_back(*wetseq19);
	GraphEdge* wetseq20 = new GraphEdge(qvec[14], qvec[19], 0.29);
	(*maxcut).push_back(*wetseq20);
	return *maxcut;
}

QProg rigettimaxcut(vector<Qubit*> qv, vector<CBit*> cv,double gamma,double beta)
{
	QProg  maxcutQAOA = CreateEmptyQProg();
	
	vector<GraphEdge> hamiltoniangraph = rigetti(qv);
	maxcutQAOA << QAOA(qv, hamiltoniangraph, gamma, beta);
	for (auto i=0;i<20;i++)
	{
		maxcutQAOA << Measure(qv[i], cv[i]);
	}
	return maxcutQAOA;
}


//QProg rigettimaxcut1(vector<Qubit*> qv, vector<CBit*> cv, double gamma, double beta, GraphEdge gd)
//{
//	QProg  maxcutQAOA = CreateEmptyQProg();
//
//	vector<GraphEdge> hamiltoniangraph = rigetti(qv);
//	maxcutQAOA << QAOA(qv, hamiltoniangraph, gamma, beta);
//	maxcutQAOA << RZ(gd.first) << RZ(gd.second);
//	maxcutQAOA << QAOA(qv, hamiltoniangraph, gamma, beta).dagger();
//	maxcutQAOA << Measure(gd.first, cv[0]) << Measure(gd.first, cv[1]);
//	return maxcutQAOA;
//}
//double rigettiQaoaTest2()
//{
//	init();
//	int qubitnum = 20;
//	vector<Qubit*> qv;
//	for (size_t i = 0u; i < qubitnum; i++)
//	{
//		qv.push_back(qAlloc());
//	}
//	vector<CBit*> cv;
//	int cbitnum = 2;
//	for (size_t i = 0u; i < cbitnum; i++)
//	{
//		cv.push_back(cAlloc());
//	}
//	auto maxcutprog = CreateEmptyQProg();
//	maxcutprog << rigettimaxcut1(qv, cv, gamma, beta,);
//	//maxcutprog << H(qv[0])<<Measure(qv[0],cv[0]);
//	load(maxcutprog);
//	run();
//	auto resultMap = getResultMap();
//}

inline string vecbool2str(vector<bool> vecbool)
{
    stringstream ss;
    for (auto s : vecbool)
        ss << s;
    return ss.str();
}

double rigettiQaoaTest(double gamma,double beta, ofstream & examplefile)
{
	init();
	int qubitnum = 20;
	vector<Qubit*> qv;
	for (size_t i = 0u; i < qubitnum; i++)
	{
		qv.push_back(qAlloc());
	}
	vector<CBit*> cv;
	int cbitnum = 20;
	for (size_t i = 0u; i < cbitnum; i++)
	{
		cv.push_back(cAlloc());
	}
	auto maxcutprog = CreateEmptyQProg();
	maxcutprog << rigettimaxcut(qv, cv, gamma, beta);
	//maxcutprog << H(qv[0])<<Measure(qv[0],cv[0]);
	load(maxcutprog);
	run();
	auto resultMap = getResultMap();
	vector<bool> bitv(20);
	vector<bool> ancilla;
	int i = 0;
	for (auto iter = resultMap.begin(); iter != resultMap.end(); iter++,i++)
	{
		//cout << iter->first << ":" << iter->second << endl;
		bitv[i] = iter->second;
	}
	//double sum = 0;

	//unsigned long ulong = bitv.to_ulong();
	ancilla.assign(bitv.begin()+2, bitv.begin()+12);
	bitv.erase(bitv.begin() + 2, bitv.begin() + 12);
	bitv.insert(bitv.end(), ancilla.begin(), ancilla.end());
	int j = 0;
	for (auto iter = bitv.begin(); iter != bitv.end(); iter++,j++)
	{
		//cout <<j<<":"<< *iter << endl;
	}
	//cout << bitv.end() - bitv.begin() << endl;
	double ssum = weightSum(qv, bitv);
	//cout <<"sum:  "<< ssum << endl;
	//cout << bitv[2] << endl;
   // if (ssum > 9.0)
        examplefile <<
        setw(16) << setprecision(8) << gamma <<
        setw(16) << setprecision(8) << beta <<
        setw(16) << setprecision(8) << ssum <<
        vecbool2str(bitv) <<
        endl;

    cout << "." << endl;

	finalize();
	return ssum;
}


double rigettiQaoaTestv2(double gamma, double beta)
{
    init();
    int qubitnum = 20;
    vector<Qubit*> qv;
    for (size_t i = 0u; i < qubitnum; i++)
    {
        qv.push_back(qAlloc());
    }
    vector<CBit*> cv;
    int cbitnum = 20;
    for (size_t i = 0u; i < cbitnum; i++)
    {
        cv.push_back(cAlloc());
    }
    auto maxcutprog = CreateEmptyQProg();
    maxcutprog << rigettimaxcut(qv, cv, gamma, beta);
    load(maxcutprog);
    run();
    auto resultMap = getResultMap();
    vector<bool> bitv(20);
    vector<bool> ancilla;
    int i = 0;
    for (auto iter = resultMap.begin(); iter != resultMap.end(); iter++, i++)
    {
        bitv[i] = iter->second;
    }
    ancilla.assign(bitv.begin() + 2, bitv.begin() + 12);
    bitv.erase(bitv.begin() + 2, bitv.begin() + 12);
    bitv.insert(bitv.end(), ancilla.begin(), ancilla.end());
    int j = 0;
    double ssum = weightSum(qv, bitv);
    finalize();
    return ssum;
}
double rigettiQaoaTestv3(double gamma, double beta, int shots)
{
    double maxx;
    double ave = 0;
    for (auto i = 0; i < shots; i++)
    {
        maxx = rigettiQaoaTestv2(gamma, beta);
        if (maxx > ave)
        {
            ave = maxx;
        }
    }
    return ave;
}
double rigettiQaoaTestv4(double x[])
{
    double maxx;
    double ave = 0;
    for (auto i = 0; i < 20; i++)
    {
        maxx = rigettiQaoaTestv2(x[0], x[1]);
       /* if (maxx > ave)
        {
            ave = maxx;
        }*/
        ave += maxx;
    }
    ave = -ave / 20;
    cout << "ooooo" << endl;
    return ave;
}

double rigettiQaoaTestv1(double gamma, double beta,int shots, ofstream & examplefile)
{
	double maxx;
	double ave = 0;
	for (auto i = 0; i < shots; i++)
	{
		maxx = rigettiQaoaTest(gamma,beta, examplefile);
		if (maxx > ave)
		{
			ave = maxx;
		}
	}
	return ave;
}
void qaoaopall()
{
    int i;
    int icount;
    int ifault;
    int kcount;
    int konvge;
    int n;
    int numres;
    double reqmin;
    double *start;
    double *step;
    double *xmin;
    double ynewlo;
    n = 2;
    start = new double[n];
    step = new double[n];
    xmin = new double[n];

    cout << "\n";
    cout << "TEST01\n";
    cout << "  Apply NELMIN to ROSENBROCK function.\n";

    start[0] = 0.70685835;
    start[1] = 2.7488936;

    //reqmin = 1.0E-08;
    reqmin = 1.0E-02;

    step[0] = 0.1;
    step[1] = 0.1;

    konvge = 10;
    kcount = 500;

    cout << "\n";
    cout << "  Starting point X:\n";
    cout << "\n";
    for (i = 0; i < n; i++)
    {
        cout << "  " << setw(14) << start[i] << "\n";
    }

    ynewlo = rigettiQaoaTestv4(start);

    cout << "\n";
    cout << "  F(X) = " << ynewlo << "\n";

    nelmin(rigettiQaoaTestv4, n, start, xmin, &ynewlo, reqmin, step,
        konvge, kcount, &icount, &numres, &ifault);

    cout << "\n";
    cout << "  Return code IFAULT = " << ifault << "\n";
    cout << "\n";
    cout << "  Estimate of minimizing value X*:\n";
    cout << "\n";
    for (i = 0; i < n; i++)
    {
        cout << "  " << setw(14) << xmin[i] << "\n";
    }

    cout << "\n";
    cout << "  F(X*) = " << ynewlo << "\n";

    cout << "\n";
    cout << "  Number of iterations = " << icount << "\n";
    cout << "  Number of restarts =   " << numres << "\n";

    delete[] start;
    delete[] step;
    delete[] xmin;

    return;
}
//change 'gamma' and 'beta',make ouput max

double optimize(linspace_t beta_vec,linspace_t gamma_vec, int shots)
{	
    double ave = 0, maxx = 0;
    //double gamma = 0;
    //double beta = 0;
    double maxgamma, maxbeta,maxvalue;
    double temp = 0;
	//double shots = 50;
    ofstream  examplefile;
    examplefile.open("example.txt", ios::app);
    examplefile.flags(ios::left);
    examplefile <<
        setw(16) << "gamma" <<
        setw(16) << "beta" <<
        setw(16) << "value" <<
        setw(16) << "config" << endl;
    for (auto gamma:gamma_vec)
    {        
        for (auto beta:beta_vec)
        {
            ave = 0;
            maxx = rigettiQaoaTestv1(gamma, beta, shots, examplefile);
            if (maxx > ave)
            {
                ave = maxx;
            }
            //ave = ave / times;
            if (temp < ave)
            {
                maxgamma = gamma;
                maxbeta = beta;
                maxvalue = ave;
                temp = ave;
            }
           // cout << "i: " << i << endl;
        }
        cout << linspace_count(gamma_vec, gamma) << "/" << gamma_vec.size() << endl;
    }
    cout << "ave:" << maxvalue << endl;
    cout << "gamma:" << maxgamma << endl;
    cout << "beta:" << maxbeta << endl;
    cout << "ratio: " << maxvalue / 10.48 << endl;
    return maxvalue;
}
double weightSum(vector<Qubit*>& qv, vector<bool>& measureoutcome)
{
	double sum = 0;
	double maxsum = 0;
	vector<GraphEdge> graphstate = rigetti(qv);
	for (auto iter = graphstate.begin(); iter != graphstate.end(); iter++)
	{
		//maxsum += iter->weight;
		for (size_t i = 0; i < 20; i++)
		{
			for (size_t j = 0; j < 20; j++)
			{
				if (iter->first == qv[i] && iter->second == qv[j])
				{
					if (measureoutcome[i] != measureoutcome[j])
					{
						sum += iter->weight;
					}
				}
			}
		}
		//size_t x1 = &(iter->first) - qv.begin();
		//if(iter->first=)
	}
	//cout << "maxsum:" << maxsum << endl;
	return sum;
}
GraphEdge::GraphEdge(Qubit * q1, Qubit *q2, double coef)
{
	this->first = q1;
	this->second = q2;
	this->weight = coef;
}
