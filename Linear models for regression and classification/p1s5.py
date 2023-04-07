import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
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
class test:
    def __init__(self,X,Y,b):
            self.X = X
            self.Y = Y
            self.b = b
    def predict(self):
        #print(self.b)
        #print("done")
        z = np.dot(self.b,self.X)
        return z
    def err(self):
        pt = self.predict()
        sum = 0
        for i in range(len(pt)):
            sum += (pt[i]-self.Y[i])**2
        return sum
df = pd.read_csv("train.csv")
k = df.to_numpy()
x = k[:,:6].T
on = [1 for i in range(x.shape[1])]
x = np.append(x,np.array([on]),axis=0)
x = x.T
x_norm=x/10000
x= x_norm.T
dg = pd.read_csv("test.csv")
kr = dg.to_numpy()
xr = kr[:,:6].T
onr = [1 for i in range(xr.shape[1])]
xr = np.append(xr,np.array([onr]),axis=0)
xr = xr.T
xr_norm=xr/10000
xr= xr_norm.T
# print(x.shape)
#print(x.shape[0])
yr = kr[:,6]
yr =yr.T
yr_norm=yr/10000
yr=yr_norm.T
#print(y)
p=np.array([120,240,360,480,600])
s=np.array(['120','240','360','480','600'])
i=0
while i<p.size :
    a = range(x.shape[1])            
    a = rand.sample(a,p[i])
    t_x=x[:,a]
    # print(i)
# print(x.shape)
#print(x.shape[0])
    y = k[:,6]
    y =y.T
    y_norm=y/10000
    y=y_norm.T
#print(y)
    re = stochastic_gradient_descent(t_x,y)
    re.SGD(0.3,100,100)
    rer = test(xr,yr,re.b)
    plt.plot(rer.err(),'.',label='datasize '+s[i])
    i+=1

plt.legend()
# plt.plot(re.costs)
# plt.show()
# plt.plot(z)
# plt.plot(re.Y)