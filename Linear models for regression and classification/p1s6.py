import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
class ridge_regression:
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
    def wgt(self,rate):
        for i in range(len(self.b)):
            diff = self.predict_difference(i)
            self.b[i] -= rate*diff
        return self.b
    def cost_fun(self,lam):
        pd = self.predict()
        reg_term=0
        for i in range(len(self.b)):
            reg_term+=(self.b[i])**2
        sum = 0
        for i in range(len(pd)):
            sum += (pd[i]-self.Y[i])**2
        return ((lam*reg_term)+sum)/(2*len(pd))
    def RG(self,rate,iterations,batch_size,lam):
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
            cost = self.cost_fun(lam)
            self.costs.append(cost)
            diff = np.dot(tmp_x,(pd-tmp_y).T)
            sa = self.b.copy()
            for j in range(len(sa)):
                sa[j] = sa[j]*lam
            diff = (diff+sa)/batch_size
            for j in range(len(self.b)):
                self.b[j] -= rate*diff[j]
df = pd.read_csv("train.csv")
k = df.to_numpy()
x = k[:,:6].T
on = [1 for i in range(x.shape[1])]
x = np.append(x,np.array([on]),axis=0)
x = x.T
x_norm=x/10000
x= x_norm.T
# print(x.shape)
#print(x.shape[0])
y = k[:,6]
y =y.T
y_norm=y/10000
y=y_norm.T
#print(y)
df1 = pd.read_csv("test.csv")
k1 = df1.to_numpy()
x1 = k1[:,:6].T
on1 = [1 for i in range(x1.shape[1])]
x1 = np.append(x1,np.array([on1]),axis=0)
x1 = x1.T
x1_norm=x1/10000
x1= x1_norm.T
# print(x.shape)
#print(x.shape[0])
y1 = k1[:,6]
y1 =y1.T
y1_norm=y1/10000
y1=y1_norm.T
re = ridge_regression(x,y)
re.RG(0.12,200,300,0.044483055205949085)
yppre = np.dot(re.b,x1)
stand = (yppre-y1).std()
print("variance  :")
print(stand**2)