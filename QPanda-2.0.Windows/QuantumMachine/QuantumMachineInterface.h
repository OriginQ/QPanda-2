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

#ifndef QUANTUM_MACHINE_INTERFACE_H
#define QUANTUM_MACHINE_INTERFACE_H
#include <map>
#include <vector>

using namespace std;

class QProg;

#define VIRTUAL // only to mark that the class is an abstract class

class QubitAddr 
{
	// Qubit Address Class, the address is expressed as
	// a 2-dimensional coordinate
	// I don't think there is anything needed comment
private:
	int x;
	int y;
public:
	virtual int getX() const;
	virtual int getY() const;
	virtual void setX(int x);
	virtual void setY(int y);
	QubitAddr(int ix, int iy);
    QubitAddr() {};
	virtual ~QubitAddr() {}
};

VIRTUAL class PhysicalQubit
{
	// Interface for the PhysicalQubit

public:
	virtual size_t getQubitAddr() = 0;
	virtual void setQubitAddr(size_t) = 0;
	//virtual void setAdjacentQubit(const vector<PhysicalQubit*>&) = 0;
	//virtual vector<PhysicalQubit*> getAdjacentQubit() const = 0;
	virtual bool getOccupancy() const = 0;
	virtual void setOccupancy(bool) = 0;
	virtual ~PhysicalQubit() {}
};

VIRTUAL class Qubit
{
	// User use this Class

public:
	virtual PhysicalQubit* getPhysicalQubitPtr() const = 0;
	virtual ~Qubit() {}
};

VIRTUAL class QubitPool
{
	// Interface for the QubitPool
	// It is the container of the PhysicalQubit

	// Get the static information of the pool
public:
	virtual size_t getMaxQubit() const = 0;
	virtual size_t getIdleQubit() const = 0;
	//virtual vector<PhysicalQubit*> getAdjacentQubit(PhysicalQubit*) const = 0;

	// Allocator and dellocator
	virtual Qubit* Allocate_Qubit() = 0;
	virtual void Free_Qubit(Qubit*) = 0;
	//virtual void init(size_t iMaxQubit) = 0;
	virtual void clearAll() = 0;
	virtual ~QubitPool() {}
	virtual size_t getPhysicalQubitAddr(Qubit*) = 0;
};

VIRTUAL class CBit 
{
public:
	virtual bool getOccupancy() const = 0; // get the occupancy status of this bit
	virtual string getName() const = 0;
	virtual void setOccupancy(bool) = 0;
	virtual ~CBit() {}
};

VIRTUAL class CMem
{
	// this class is considered as the container of the CBit
public:
	virtual CBit* Allocate_CBit() = 0;
	virtual size_t getMaxMem() const = 0;
	virtual size_t getIdleMem() const = 0;
	virtual void Free_CBit(CBit*) = 0;
	virtual void clearAll() = 0;
	virtual ~CMem() {}
};

VIRTUAL class QResult
{
	// this class contains the result of the quantum measurement
public:
	virtual map<string, bool> getResultMap() const = 0;
	virtual void append(pair<string, bool>) = 0;
	virtual ~QResult() {}
};

VIRTUAL class QMachineStatus
{
	// this class contains the status of the quantum machine
public:
	virtual int getStatusCode() const = 0;
	virtual void setStatusCode(int) = 0;
	virtual ~QMachineStatus() {}
};

VIRTUAL class QuantumMachine
{
public:
	virtual void init() = 0; // to initialize the quantum machine
	virtual Qubit* Allocate_Qubit() = 0; // allocate and return a qubit
	virtual CBit* Allocate_CBit() = 0; // allocate and run a cbit
    virtual void Free_Qubit(Qubit*) = 0; // free a qubit
    virtual void Free_CBit(CBit*) = 0; // free a cbit
	virtual void load(QProg &) = 0; // load a qprog
	virtual void append(QProg&) = 0; // append the qprog after the original
	virtual void run() = 0; // run on the quantum machine
	virtual QMachineStatus* getStatus() const = 0; // get the status of the quantum machine
	virtual QResult* getResult() = 0; // get the result of the quantum program
	virtual void finalize() = 0; // finish the program
	virtual ~QuantumMachine() {} // destructor
};

#endif