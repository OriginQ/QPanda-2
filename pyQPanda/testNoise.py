
from pyqpanda import *

if __name__ == "__main__":
	dict = {"gates":[["RX","RY"],["CNOT"]],
            "noisemodel":{"RX":[NoiseModel.DECOHERENCE_KRAUS_OPERATOR,10.0,2.0,0.03],
                          "RY":[NoiseModel.DECOHERENCE_KRAUS_OPERATOR,10.0,2.0,0.03]
						  }
		   }              
	qvm = NoiseQVM() 
	qvm.initQVM(dict)


    

    