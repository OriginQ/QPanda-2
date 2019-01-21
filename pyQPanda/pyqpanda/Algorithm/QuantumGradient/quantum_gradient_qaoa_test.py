from pyqpanda.Algorithm.QuantumGradient.quantum_gradient import *





step=2
gamma=(1-2*np.random.random_sample(step))*2
beta=(1-2*np.random.random_sample(step))*pi/4
qubit_number=7
hp_7=PauliOperator({'Z0 Z4':0.73,'Z0 Z5':0.33,'Z0 Z6':0.5,'Z1 Z4':0.69,'Z1 Z5':0.36,'Z2 Z5':0.88,'Z2 Z6':0.58,
                 'Z3 Z5':0.67,'Z3 Z6':0.43})
Hd_7=PauliOperator({'X0':1,'X1':1,'X2':1,'X3':1,'X4':1,'X5':1,'X6':1})
Hp=hp_7*0.5
Hp=flatten(Hp)

init()
qlist=qAlloc_many(qubit_number)
qqat=qaoa(qubit_number,step,gamma,beta,Hp,Hd_7)


cost=qqat.get_exp(qlist)
print('cost',cost)

#qqat.optimize(qlist,20,0.01)
qqat.momentum_optimize(qlist,50,0.02,0.9)
print(qqat.beta,qqat.gamma)
exp2=qqat.get_exp(qlist)
print('exp2',exp2)
finalize()


