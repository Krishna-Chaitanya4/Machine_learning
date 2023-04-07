import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
class batch_gradient_descent:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.b = [0 for i in range(X.shape[0])]
        self.costs = []
    def predict(self):
        z = np.dot(self.b,self.X)
        return z
    def predict_difference(self,j):
        pd = self.predict()
        sum = 0
        for i in range(len(pd)):
            sum += (pd[i]-self.Y[i])*self.X[j,i]
        return sum/len(pd)
    def update(self,rate):
        for i in range(len(self.b)):
            diff = self.predict_difference(i)
            self.b[i] -= rate*diff
    def cost_fun(self):
        pd = self.predict()
        sum = 0
        for i in range(len(pd)):
            sum += (pd[i]-self.Y[i])**2
        return sum/(2*len(pd))
    def BGD(self,rate,iterations):
        i=0
        while i<iterations:
            i+=1
            cost = self.cost_fun()
            self.costs.append(cost)
            self.update(rate)       
df = pd.read_csv("train.csv")
k = df.to_numpy()
x = k[:,:6].T
on = [1 for i in range(x.shape[1])]
x = np.append(x,np.array([on]),axis=0)
x = x/10**4
y = k[:,6]
y = y/10**4
reg = batch_gradient_descent(x,y)
reg.BGD(0.8,50)
z = reg.predict()
print(reg.b)
#plt.plot(z)
#plt.plot(y)
plt.plot(reg.costs)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
# print(reg.costs[len(reg.costs)-1])
class stochastic_gradient_descent:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.b = [0 for i in range(X.shape[0])]
        self.costs = []
    def predict(self):
        #print(self.b)
        #print("done")
        z = np.dot(self.b,self.X)
        return z
    def predict_difference(self,j):
        pd = self.predict()
        sum = 0
        for i in range(len(pd)):
            sum += (pd[i]-self.Y[i])*self.X[j,i]
        return sum/len(pd)
    def update(self,rate):
        for i in range(len(self.b)):
            diff = self.predict_difference(i)
            self.b[i] -= rate*diff
        #print(self.b)
    def cost_fun(self):
        pd = self.predict()
        sum = 0
        for i in range(len(pd)):
            sum += (pd[i]-self.Y[i])**2
        return sum/(2*len(pd))
    def SGD(self,rate,iterations,batch_size):
        i=0
        while i<iterations:
            a = range(self.X.shape[1])
            
            a = rand.sample(a,batch_size)
            tmp_x=self.X[:,a]
            tmp_y=self.Y[a]
            # print(a)
            # i+=1
            # cost = self.cost_fun()
            # self.costs.append(cost)
            # for j in range(len(self.b)):
            #     sum = 0
            #     for sa in a:
            #         pre = np.dot(self.b,self.X[:,sa])
            #         sum += (pre-self.Y[sa])*self.X[j,sa]
            #     self.b -= rate*sum/batch_size
            pd = np.dot(self.b,tmp_x)
            i+=1
            cost = self.cost_fun()
            self.costs.append(cost)
            diff = np.dot(tmp_x,(pd-tmp_y).T)
            for j in range(len(self.b)):
                self.b[j] -= rate*diff[j]/batch_size
from cProfile import label
# df = pd.read_csv("train.csv")
# k = df.to_numpy()
# x = k[:,:6].T
# on = [1 for i in range(x.shape[1])]
# x = np.append(x,np.array([on]),axis=0)
# x = x.T
# x_norm=x/x.max(axis=0)
# x= x_norm.T
# # print(x.shape)
# #print(x.shape[0])
# y = k[:,6]
# y =y.T
# y_norm=y/y.max(axis=0)
# y=y_norm.T
re3 = stochastic_gradient_descent(x,y)
re3.SGD(0.03,100,1)
plt.plot(re3.costs,label='Batch size-1')
print("weights for Batch size 1:")
print(re3.b)
re2 = stochastic_gradient_descent(x,y)
re2.SGD(0.03,100,2)
plt.plot(re2.costs,label='Batch size-2')
print("weights for Batch size 2:")
print(re2.b)
re1 = stochastic_gradient_descent(x,y)
re1.SGD(0.03,100,5)
plt.plot(re1.costs,label='Batch size-5')
print("weights for Batch size 5:")
print(re1.b)
re4 = stochastic_gradient_descent(x,y)
re4.SGD(0.03,100,30)
plt.plot(re4.costs,label='Batch size-30')
print("weights for Batch size 30:")
print(re4.b)
re5 = stochastic_gradient_descent(x,y)
re5.SGD(0.03,100,1000)
plt.plot(re5.costs,label='Batch size-1000')
print("weights for Batch size 1000:")
print(re5.b)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
# plt.plot(re2.costs)