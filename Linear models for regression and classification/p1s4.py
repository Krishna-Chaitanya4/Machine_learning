import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
class lasso_regression:
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
            reg_term+=abs(self.b[i])
        sum = 0
        for i in range(len(pd)):
            sum += (pd[i]-self.Y[i])**2
        return ((lam*reg_term)+sum)/len(pd)
    def LG(self,rate,iterations,batch_size,lam):
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
        return sum/len(pt)
df = pd.read_csv("train.csv")
k_i = df.to_numpy()
k_i=k_i.T
k_i=np.delete(k_i,1,0)
k_i=np.delete(k_i,5,0)
k=k_i.T
x = k[:,:4].T
on = [1 for i in range(x.shape[1])]
x = np.append(x,np.array([on]),axis=0)
x = x.T
x_norm=x/10000
x= x_norm.T
# print(x.shape)
#print(x.shape[0])
y = k[:,4]
y =y.T
y_norm=y/10000
y=y_norm.T
#print(y)
re = lasso_regression(x,y)
z = re.predict()
dg = pd.read_csv("test.csv")
kr_i = dg.to_numpy()
kr_i=kr_i.T
kr_i=np.delete(kr_i,1,0)
kr_i=np.delete(kr_i,5,0)
kr=kr_i.T
xr = kr[:,:4].T
onr = [1 for i in range(xr.shape[1])]
xr = np.append(xr,np.array([onr]),axis=0)
xr = xr.T
xr_norm=xr/10000
xr= xr_norm.T
# print(x.shape)
#print(x.shape[0])
yr = kr[:,4]
yr =yr.T
yr_norm=yr/10000
yr=yr_norm.T
#print(y)
rer = test(xr,yr,re.b)
i=0
lambd_min = 0
er = 10000
while i<100 :
    p=rand.uniform(0,1)
    re.LG(0.1,100,1,p)
    k = rer.err()
    if k<er:
        er = k
        lambd_min = p
    plt.plot(p,k,'.')
    i+=1
print(lambd_min)
print(re.b)
#plt.plot(re.b,'.')
