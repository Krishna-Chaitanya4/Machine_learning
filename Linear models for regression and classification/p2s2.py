import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
class lasso_logistic_regression:
    def __init__(self,x,y):
        self.x = x
        self.y = y
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
    def cost_fun(self,lam):
        y = np.dot(self.w,self.x)
        y_pre = self.predict(y)
        sum = 0
        reg_term=0
        for i in range(len(self.w)):
            reg_term+=abs(self.w[i])
        for i in range(len(y_pre)):
            sum += (y_pre[i]-self.y[i])**2
        return (lam*reg_term+sum)/self.x.shape[1]
    def sigmoid(self,x):
        #print(x)
        k = 1.0/(1+np.exp(-x))
        return k
    def SGD(self,rate,iterations,bs,lam):
        i=0
        while i<iterations:
            i+=1
            cost = self.cost_fun(lam)
            self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            y = np.dot(self.w,newx)
            y_pre = self.predict(y)
            dw = np.dot(newx,(y_pre-newy).T)
            sa = self.w.copy()
            for j in range(len(sa)):
                sa[j] = sa[j]*lam
            dw = (dw-sa)/bs
            for j in range(self.x.shape[0]):
                self.w[j] += rate*dw[j]
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
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
class ridge_logistic_regression:
    def __init__(self,x,y):
        self.x = x
        self.y = y
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
    def cost_fun(self,lam):
        y = np.dot(self.w,self.x)
        y_pre = self.predict(y)
        sum = 0
        reg_term=0
        for i in range(len(self.w)):
            reg_term+=(self.w[i])**2
        for i in range(len(y_pre)):
            sum += (y_pre[i]-self.y[i])**2
        return (lam*reg_term+sum)/self.x.shape[1]
    def sigmoid(self,x):
        #print(x)
        k = 1.0/(1+np.exp(-x))
        return k
    def SGD(self,rate,iterations,bs,lam):
        i=0
        while i<iterations:
            i+=1
            cost = self.cost_fun(lam)
            self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            y = np.dot(self.w,newx)
            y_pre = self.predict(y)
            dw = np.dot(newx,(y_pre-newy).T)
            sa = self.w.copy()
            for j in range(len(sa)):
                sa[j] = sa[j]*lam
            dw = (dw-sa)/bs
            for j in range(self.x.shape[0]):
                self.w[j] += rate*dw[j]
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
X_t = np.load("train_images_subset.npy")
Y_t = np.load("train_labels_subset.npy")
X=X_t[:5000,:].T
Y=Y_t[:5000]
on = [1 for i in range(X.shape[1])]
X = np.append(X,np.array([on]),axis=0)
ks = [1,2]

# pre = re.predict(np.dot(re.w,re.x))
#print(len(re.w))
d=X_t[5000:,:].T
on = [1 for i in range(d.shape[1])]
d = np.append(d,np.array([on]),axis=0)
f=Y_t[5000:]
i=0
lambd_min = 0
er = 100000000000000
while i<20:
    # print(i)
    re = lasso_logistic_regression(X,Y)
    p=rand.uniform(0,1)
    re.SGD(0.01,100,20,p)
    rer =test(d,f,re.w)
    k = rer.err()
    if k<er:
        er = k
        lambd_min = p
    # plt.plot(p,k,'.')
    i+=1
# print(lambd_min)
# print(er)
# plt.plot(re.costs)
# plt.show()
# plt.plot(pre[:20],'.')
# plt.plot(re.y[:20],'.')