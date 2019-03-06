## optimizerTest

import pyQPanda as pq
import numpy as np


train_x = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27,3.1])
train_y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94,1.3])


x = np.zeros((len(train_x),2))
y = np.zeros((len(train_y),2))
for i in range(len(train_x)):
    x[i][0] = train_x[i]
    y[i][0] = train_y[i]

X = pq.var(x)
Y = pq.var(y)
W = pq.var(1.0, True)
b = pq.var(1.0, True)

Y_ = W*X+b

v = pq.var

loss = pq.sum(pq.poly(Y - Y_, v(2.0)) / v(17.0))

print(loss)
optimizer = pq.VanillaGradientDescentOptimizer.minimize(loss, 0.01, 1.e-6)
#optimizer = pq.MomentumOptimizer.minimize(loss, 0.01, 1.e-6)
#optimizer = pq.AdaGradOptimizer.minimize(loss,0.01,0.0, 1e-10)
#optimizer = pq.RMSPropOptimizer.minimize(loss, 0.001, 0.9, 1e-10)
#optimizer = pq.AdamOptimizer.minimize(loss, 0.001,0.9,0.999, 1e-8)
print("====optimizer=====")
leaves = optimizer.get_variables()
print("====leaves=====")
print(leaves)
print("====get_loss=====")
print(optimizer.get_loss())

it = 1000

for i in range(it):
    optimizer.run(leaves,0)
    oloss = optimizer.get_loss()
    print("i:",i," loss:",oloss," W:",pq.eval(W,True)," b:",pq.eval(b,True) )
    

