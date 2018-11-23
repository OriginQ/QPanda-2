from matplotlib.pyplot import *
from pyqpanda.Algorithm.VariationalQuantumEigensolver.vqe import *
from pyqpanda.Hamiltonian import (PauliOperator, 
                                  chem_client)



def H2_vqe_test():
        geometry=[['H',[0,0,0]], ['H',[0,0,0.74]]]
        basis="sto-3g"
        multiplicity=1
        charge=0
        run_mp2=True
        run_cisd=True
        run_ccsd=True
        run_fci=True

        str1=chem_client(
                geometry=geometry,
                basis=basis,
                multiplicity=multiplicity,
                charge=charge,
                run_mp2=run_mp2,
                run_cisd=run_cisd,
                run_ccsd=run_ccsd,
                run_fci=run_fci,
                hamiltonian_type="pauli")

        pauli=convert_operator(str1)
        print(pauli)
        # distance_range=np.linspace(0.25,2.5,50)
        # energy=np.zeros(len(distance_range))
        # for i,distance in enumerate(distance_range):
        #     print('{} / {}'.format(i+1,len(distance_range)))
        #     energy[i]=H2_energy_from_distance(distance)
        # plot(distance_range,energy)
        # show()
        # energy[-1]
        operator=convert_operator(str1)
        n_qubit = operator.get_qubit_count()
        n_electron=get_electron_count(geometry)

        n_param=get_ccs_n_term(n_qubit,n_electron)
        paramlist=np.ones(n_param)*0.5
        op1=get_ccs(n_qubit,n_electron,paramlist)
        #print(op1)

        ucc=cc_to_ucc_hamiltonian(op1)
        #print(ucc)
        ucc=flatten(ucc)
        #print(ucc)
        # component=('Z0 Z1 Z2',0.23)
        # exp= vqe_subroutine(n_qubit,n_electron,convert_operator(str1),unitaryCC_=ucc,shots=1000)
        # print(exp)
        #result=H2_vqe_subprocess(0.64)

        distance_range=np.linspace(0.25,2.5,50)



        energy=H2_vqe(
                distance_range=distance_range,
                initial_guess=paramlist,
                basis='sto-3g',
                multiplicity=1,
                charge=0,
                run_mp2=True,
                run_cisd=True,
                run_ccsd=True,
                run_fci=True,
                method='Powell')

        plot(distance_range,energy)
        show()
        return







