#综合示例——H2基态能量计算
#3.使用封装的ChemiQ计算接口进行实现
import matplotlib.pyplot as plt

from pyqpanda import *

if __name__=="__main__":

    distances = [x * 0.1 for x in range(2, 25)]
    molecule = "H 0 0 0\nH 0 0 {0}"

    molecules = []
    for d in distances:
        molecules.append(molecule.format(d))

    chemiq = ChemiQ()
    chemiq.setMolecules(molecules)
    chemiq.setCharge(0)
    chemiq.setMultiplicity(1)
    chemiq.setBasis("sto-3g")
    chemiq.setUccType(UccType.UCCS)
    chemiq.setTransformType(TransFormType.Jordan_Wigner)
    chemiq.setOptimizerType(OptimizerType.NELDER_MEAD)
    chemiq.setOptimizerIterNum(200)
    chemiq.setOptimizerFatol(200)
    chemiq.setOptimizerDisp(True)
    chemiq.exec()
    
    value = chemiq.getEnergies()
    print(value)

    plt.plot(distances , value, 'r')
    plt.xlabel('distance')
    plt.ylabel('energy')
    plt.title('VQE PLOT')
    plt.show()