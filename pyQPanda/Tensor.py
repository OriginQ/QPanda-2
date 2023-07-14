import pyqpanda as pq
import numpy as np

#默认不合并同类项
operator = pq.PauliOperator({"X0 Y2" : -0.044750,
                            "Z0 Z1" : 0.189766,
                            "Z1 Z0" : 0.270597,
                            "Z3" : -0.242743})

print(operator)

#合并同类项
operator = pq.PauliOperator({"X0 Y2" : -0.044750,
                            "Z0 Z1" : 0.189766,
                            "Z1 Z0" : 0.270597,
                            "Z3" : -0.242743},True)

print(operator)

#手动合并
operator.reduce_duplicates()