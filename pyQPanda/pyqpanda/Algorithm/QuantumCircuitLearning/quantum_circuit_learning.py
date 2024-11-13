from pyqpanda import *
from pyqpanda.utils import *
import copy
import numpy as np
import math
from math import pi
import random
from matplotlib.pyplot import *


def pauliX(qubit_list,coef,t):
    """
    Constructs a quantum circuit with RX gates, based on the given qubit list, coefficients, and time parameter.

        Args:
            qubit_list (list): A list of qubits (integers) over which the RX gates will be applied.
            coef (list): A list of coefficients (floats) determining the rotation angle for each RX gate.
            t (float): The time duration over which the rotation occurs.

        Returns:
            QCircuit: A QuantumCircuit object representing the constructed circuit with RX gates.

        The RX gate is a single-qubit gate that performs a rotation around the X-axis. This function
        sequentially inserts RX gates into a QuantumCircuit, where each gate's rotation angle is
        calculated by multiplying the corresponding coefficient by twice the time parameter.
    """
    qcir=QCircuit()
    for i in range(len(coef)):
        qcir.insert(RX(qubit_list[i],2*coef[i]*t))
    return qcir

def pauliZjZk(qubit_list,coef,t):
    """
    Constructs a quantum circuit using the Pauli-Z gates on a list of qubits, based on the provided coefficients and time factor.

        Args:
            qubit_list (list): A list of qubit objects representing the qubits in the quantum circuit.
            coef (list of lists): A 2D list where each inner list corresponds to a qubit pair, containing the coefficients for the
                                  RZ gates to be applied between the qubits.
            t (float): The time factor that scales the RZ gate operations.

        Returns:
            QCircuit: A quantum circuit object representing the constructed circuit, which can be executed on a quantum simulator or quantum hardware.

            This function operates within the `pyQPanda` package, specifically in the `Algorithm.QuantumCircuitLearning` module. It applies
            CNOT and RZ gates to create a circuit that models a quantum interaction based on the input coefficients and time factor.
    """
    qcir=QCircuit()
    for i in range(len(coef)):
        for j in range(i):
            qcir.insert(CNOT(qubit_list[i],qubit_list[j])) \
                .insert(RZ(qubit_list[j],coef[i][j]*2*t)) \
                .insert(CNOT(qubit_list[i],qubit_list[j]))
    return qcir

def initial_state(qubit_list,x):
    """
    Constructs an initial quantum circuit by applying a rotation gate on each qubit in the given list.
    
        Args:
            qubit_list (list): A list of qubits to be manipulated in the quantum circuit.
            x (float): The parameter used to define the rotation angles for the RY and RZ gates.
    
        Returns:
            QCircuit: A quantum circuit object representing the initial state with RY and RZ gates applied to each qubit.
    
    The function operates within the `pyQPanda` package, which is designed for programming quantum computers using quantum circuits and gates.
    It is intended to be used with quantum circuit simulators, quantum virtual machines, or quantum cloud services.
    """
    qcir=QCircuit()
    for qubit in qubit_list:
        qcir.insert(RY(qubit,np.arcsin(x))).insert(RZ(qubit,np.arccos(x*x)))
    return qcir


def ising_model_simulation(qubit_list,hamiltonian_coef2d,step,t):
    """
    Simulates the Ising model Hamiltonian using a quantum circuit, incorporating both single-qubit and two-qubit interactions.

    Args:
        qubit_list (list): A list of qubits to be used in the simulation.
        hamiltonian_coef2d (list of list): A 2D list representing the coefficients for the Ising Hamiltonian.
                                            The diagonal elements correspond to the single-qubit terms (Xi coefficients),
                                            while the off-diagonal elements represent the two-qubit terms (ZjZk coefficients).
        step (int): The number of time steps to simulate the Ising model.
        t (float): The total time duration over which the simulation is to be performed.

    Returns:
    QCircuit: A quantum circuit object representing the simulation of the Ising model.
    """
    single_coef=[]
    for i in range(len(qubit_list)):
        single_coef.append(hamiltonian_coef2d[i][i])
    qcir=QCircuit()
    for i in range(step):
        qcir.insert(pauliX(qubit_list,single_coef,t/step)) \
            .insert(pauliZjZk(qubit_list,hamiltonian_coef2d,t/step))
    return qcir

def unitary(qubit,theta):
    """
    Generates a quantum circuit with a sequence of rotation gates on a specified qubit.

        Args:
            qubit (int): The index of the qubit to be manipulated.
            theta (list of float): A list of rotation angles in degrees for the gates. The order is as follows:
            1. RX rotation angle for the qubit.
            2. RZ rotation angle for the qubit.
            3. RX rotation angle for the qubit.

        Returns:
            QCircuit: A quantum circuit object with the specified rotation gates inserted in sequence.

    The function constructs a quantum circuit using the pyQPanda library, which is designed for programming quantum computers. The circuit is composed of three gates:
    1. RX gate applied to the specified qubit with the provided RX rotation angle.
    2. RZ gate applied to the specified qubit with the provided RZ rotation angle.
    3. RX gate applied to the specified qubit with the provided RX rotation angle.
    
    The resulting circuit can be used for simulations or further processing within the quantum computing framework provided by pyQPanda.
    """
    qcir=QCircuit()
    qcir.insert(RX(qubit,theta[2])).insert(RZ(qubit,theta[1])).insert(RX(qubit,theta[0]))
    return qcir

def one_layer(qubit_list,theta2d,hamiltonian_coef2d,step=100,t=10):
    """
    Constructs a quantum circuit for a learning algorithm in the pyQPanda framework.

        Args:
            qubit_list (list of int): A list of integers representing the qubits involved in the circuit.
            theta2d (list of list of float): A 2D list where each sublist contains three floats representing the rotation angles for a single qubit.
            hamiltonian_coef2d (list of list of float): A 2D list representing the coefficients of the Hamiltonian for the system.
            step (int, optional): The number of steps to simulate the Ising model. Defaults to 100.
            t (int, optional): The temperature parameter for the Ising model simulation. Defaults to 10.

        Returns:
            QCircuit: A QuantumCircuit object representing the quantum circuit after applying the specified transformations.

    The function initializes a Quantum Circuit and inserts an Ising model simulation followed by a series of unitary operations defined by the angles in theta2d for each qubit. The resulting circuit can be used for quantum learning algorithms within the pyQPanda package.
    """
    qcir=QCircuit()
    qcir.insert(ising_model_simulation(qubit_list,hamiltonian_coef2d,step,t))
    for i in range(len(qubit_list)):
        qcir.insert(unitary(qubit_list[i],theta2d[i]))
    return qcir



def learning_circuit(qubit_list,layer,theta3d,hamiltonian_coef3d,x,t=10):
    """
    Constructs a quantum circuit for learning the state of a quantum system.

        Args:
            qubit_list (list): A list of integers representing the qubits in the quantum circuit.
            layer (int): The number of layers in the quantum circuit.
            theta3d (list of lists of lists): Parameters to be optimized, structured as [layer][qubit_number][3].
            hamiltonian_coef3d (list of lists of lists): Coefficients of the fully connected transverse Ising model Hamiltonian,
                structured as [layer][qubit_num][qubit_num], where C[i,i] are Pauli X coefficients,
                C[i,j] are Pauli Z and Zj coefficients when i > j, and C[i,j] = 0 when i < j.
            x (float): The initial state or time parameter for the quantum circuit.
            t (int, optional): The number of iterations for the learning process (default is 10).

        Returns:
            QCircuit: The constructed quantum circuit object.
    """
    qcir=QCircuit()
    qcir.insert(initial_state(qubit_list,x))
    for i in range(layer):
        qcir.insert(one_layer(qubit_list,theta3d[i],hamiltonian_coef3d[i],t))
    return qcir
    
def get_expectation(program,qubit_list):
    """
    Calculate the expectation value of a quantum circuit.

    This function evaluates the expectation value by executing a quantum circuit
    defined by `program` and measuring the qubits in the `qubit_list`. It
    computes the expectation by summing the product of the measured probabilities
    and their respective signs based on the parity of their indices.

        Args:
            program (list): A list of tuples defining the quantum circuit. Each tuple
                          contains the gate and its parameters.
            qubit_list (list): A list of integers representing the indices of the qubits
                             to be measured.

        Returns:
            float: The calculated expectation value.

    The function is designed to be used within the pyQPanda package, which
    provides a framework for programming quantum computers using quantum circuits
    and gates. It operates on a quantum circuit simulator or quantum cloud service.
    """
    expect=0
    result=prob_run(program=program,noise=False,select_max=-1,qubit_list=qubit_list,dataType='list')
    for i in range(len(result)):
        if parity(i):
            expect-=result[i]
        else:
            expect+=result[i]
    return expect   

def parity(number):
    """
    Determines the parity of a given integer by counting the number of 1s in its binary representation.
    
        Args:
            number (int): The integer for which to calculate the parity.
    
        Returns:
            int: The parity of the input number. Returns 0 if the number of 1s is even, and 1 if it is odd.
    
    This function is utilized within the quantum computing framework pyQPanda, which simulates quantum circuits and gates for
    quantum computing applications. The parity calculation can be critical in certain quantum algorithms, particularly those
    involving error correction and logical operations in quantum computing.
    """
    check=0
    for i in bin(number):
        if i=='1':
            check+=1
    return check%2

class qcl:
    def __init__(self,
                 qubit_number,
                 layer,
                 coef=1,
                 t=10
    ):
        self.qnum=qubit_number
        self.layer=layer
        self.theta3d=np.random.random_sample((layer,qubit_number,3))*2*pi
        self.hamiltonian_coef3d=1-2*np.random.random_sample((layer,qubit_number,qubit_number))
        self.coef=1
        self.t=10

    def learning_circuit(self,qubit_list,x):
        """Generate a quantum circuit for learning.

        This function constructs a quantum circuit based on the specified qubit list and input data.

            Args:
                qubit_list: The list of qubits to be used in the circuit.
                x: The input data to initialize the quantum circuit.

            Returns:
                A QCircuit object representing the constructed quantum circuit.
        """
        qcir=QCircuit()
        qcir.insert(initial_state(qubit_list,x))
        for i in range(self.layer):
            qcir.insert(one_layer(qubit_list,self.theta3d[i],self.hamiltonian_coef3d[i],self.t))
        return qcir
   
    def get_function_value(self,qubit_list,x):
        """Calculate the function value based on the quantum circuit.

        This function evaluates the quantum circuit and returns the expectation value for the specified qubit.

            Args:
                qubit_list: The list of qubits to be measured.
                x: The input data used to initialize the quantum circuit.

            Returns:
                The expectation value calculated from the quantum circuit, scaled by the coefficient.
        """
        prog=QProg()
        prog.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,x,self.t))
        return get_expectation(prog,[qubit_list[0]])*self.coef
        
    def get_expectation(self,qubit_list,x):
        """Calculate the expectation value from the quantum circuit.

        This function constructs a quantum circuit and computes the expectation value for the specified qubit.

            Args:
                qubit_list: The list of qubits to be measured.
                x: The input data used to initialize the quantum circuit.

            Returns:
                The expectation value calculated from the quantum circuit.
        """
        prog=QProg()
        prog.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,x,self.t))
        return get_expectation(prog,[qubit_list[0]])
    
    def cost_funciton(self,qubit_list,train_data):
        """Calculate the cost based on training data.

        This function computes the cost by comparing the expected values with the predicted function values from the quantum circuit.

            Args:
                qubit_list: The list of qubits used in the calculations.
                train_data: A list of tuples where each tuple contains input data and the corresponding expected output.

            Returns:
                The total cost calculated from the training data.
        """
        cost=0
        for data in train_data:
            cost+=(data[1]-self.get_function_value(qubit_list,data[0]))*(data[1]-self.get_function_value(qubit_list,data[0]))
        return cost
    
    def optimize(self,qubit_list,m_qulist,train_data,velocity=0.01):
        """
        parameter optimization:optimize theta3d and coef

            Args:
                qubit_list : The list of qubit.
                m_qulist   : The list of qubit.
                train_data : list
                velocity   : float
        
            Returns:
                None
        """
        
        new_theta3d=copy.copy(self.theta3d)
        grad1=0
        for data in train_data:
            grad1+=(self.get_function_value(qubit_list,data[0])-data[1])*self.get_expectation(qubit_list,data[0])*2
        
        for i in range(self.layer):
            for j in range(self.qnum):
                for k in range(3):
                    grad=0
                    for data in train_data:
                        
                        self.theta3d[i][j][k]+=pi/2
                        qprog2=QProg()
                        qprog2.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,data[0],10))
                        
                        qprog3=QProg()
                        self.theta3d[i][j][k]-=pi
                        qprog3.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,data[0],10))
                        
                        self.theta3d[i][j][k]+=pi/2
                        
                        result3=get_expectation(qprog2,m_qulist)
                        result4=get_expectation(qprog3,m_qulist)
                        grad+=self.coef*(self.get_function_value(qubit_list,data[0])-data[1])*(result3-result4)
                    new_theta3d[i][j][k]-=grad*velocity
        #print(new_theta3d)
        self.theta3d=copy.copy(new_theta3d)
        self.coef-=grad1*velocity*0.05    
        
    def function_learning(self,qubit_list,train_data,step=1000,velocity=0.01):
        """
        train quantum circuit

            Args:
                qubit_list : The list of qubit.
                m_qulist   : The list of qubit.
                train_data : list
                step       : int
                velocity   : float
        
            Returns:
                float
        """

        for i in range(step):
            self.optimize(qubit_list,[qubit_list[0]],train_data,velocity)
        cost=self.cost_funciton(qubit_list,train_data)
        return cost

        
def generate_train_data(type,range):
    """
    Generate training data for various mathematical functions within specified range.

        Args:
            type (int): Specifies the function type to generate data for.
                - 1: Outputs x squared (x^2).
                - 2: Outputs exponential of x (exp(x)).
                - 3: Outputs sine of x (sin(x)).
                - 4: Outputs the absolute value of x (|x|).
            range (range): A range object defining the interval for the x values.

        Returns:
            list of tuples: A list containing tuples of (x, y) where y is the function of x.

        Note: 
            This function is intended for use within the pyQPanda package for quantum computing
            applications, particularly in the context of quantum circuit simulation and quantum
            cloud services.
    """
    train_data=[]
    for i in range:
        if type==1:
            train_data.append((i,i*i))
        elif type==2:
            train_data.append((i,np.exp(i)))
        elif type==3:
            train_data.append((i,np.sin(i)))
        elif type==4:
            train_data.append((i,np.abs(i)))
        else:
            print('undefined')
    return train_data
        
    