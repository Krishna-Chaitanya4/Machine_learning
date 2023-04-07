import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
X = np.load("train_images_subset.npy")
X = X.T
X = X
Y = np.load("train_labels_subset.npy")
# k = []
# for i in Y:
#     if i==9:
#         k.append(9)
#     else:
#         k.append(2)
# Y = k
Y = np.array(Y)
on = [1 for i in range(X.shape[1])]
X = np.append(X,np.array([on]),axis=0)

# ks = [1,2]
#print(X[:,20])
#print(y)
class batch_logistic_regression:
    def __init__(self,x,y):
        self.x = x.copy()
        self.y = y.copy()
        self.costs = []
        self.w = [0 for i in range(x.shape[0])]
    def predict(self,y):
        pre = []
        for i in y:
            pre.append(self.sigmoid(i))
        y_pre = []
        for i in pre:
            if i>=0.5:
                y_pre.append(2)
            else:
                y_pre.append(9)
        y_pre = np.array(y_pre)
        return y_pre
    def cost_fun(self):
        y = np.dot(self.w,self.x)
        y_pre = self.predict(y)
        sum = 0
        for i in range(len(y_pre)):
            sum += (y_pre[i]-self.y[i])**2
        return sum/self.x.shape[1]
    def sigmoid(self,x):
        #print(x)
        k = 1.0/(1+np.exp(-x))
        return k
    def BGD(self,rate,iterations):
        i=0
        while i<iterations:
            i+=1
            cost = self.cost_fun()
            self.costs.append(cost)
            y = np.dot(self.w,self.x)
            y_pre = self.predict(y)
            dw = np.dot(self.x,(y_pre-self.y).T)
            for j in range(self.x.shape[0]):
                self.w[j] += rate*dw[j]/self.x.shape[1]
class stochastic_logistic_regression:
    def __init__(self,x,y):
        self.x = x.copy()
        self.y = y.copy()
        self.costs = []
        self.w = [0 for i in range(x.shape[0])]
    def predict(self,y):
        pre = []
        for i in y:
            pre.append(self.sigmoid(i))
        y_pre = []
        for i in pre:
            if i>=0.5:
                y_pre.append(2)
            else:
                y_pre.append(9)
        y_pre = np.array(y_pre)
        return y_pre
    def cost_fun(self):
        y = np.dot(self.w,self.x)
        y_pre = self.predict(y)
        sum = 0
        for i in range(len(y_pre)):
            sum += (y_pre[i]-self.y[i])**2
        return sum/self.x.shape[1]
    def sigmoid(self,x):
        #print(x)
        k = 1.0/(1+np.exp(-x))
        return k
    def BGD(self,rate,iterations,bs):
        i=0
        while i<iterations:
            i+=1
            cost = self.cost_fun()
            self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            y = np.dot(self.w,newx)
            y_pre = self.predict(y)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w[j] += rate*dw[j]/bs
re = stochastic_logistic_regression(X,Y)
re.BGD(0.1,30,20)
pre = re.predict(np.dot(re.w,re.x))
plt.plot(re.costs)
print(re.costs[-1])
plt.show()
plt.plot(pre[:50],'.')
plt.plot(re.y[:50],'.')