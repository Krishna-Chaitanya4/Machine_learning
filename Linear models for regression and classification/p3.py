import numpy as np
import pandas as pd
import random as rand
import sys
import matplotlib.pyplot as plt
class Bonus:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w0 = [0 for i in range(self.x.shape[0])]
        self.w1 = [0 for i in range(self.x.shape[0])]
        self.w2 = [0 for i in range(self.x.shape[0])]
        self.w3 = [0 for i in range(self.x.shape[0])]
        self.w4 = [0 for i in range(self.x.shape[0])]
        self.w5 = [0 for i in range(self.x.shape[0])]
        self.w6 = [0 for i in range(self.x.shape[0])]
        self.w7 = [0 for i in range(self.x.shape[0])]
        self.w8 = [0 for i in range(self.x.shape[0])]
        self.w9 = [0 for i in range(self.x.shape[0])]
    def sigmoid(self,x):
        k = 1.0/(1+np.exp(-x))
        return k
    def predict(self,y):
        pre = []
        for i in y:
            pre.append(self.sigmoid(i))
        y_pre = []
        for i in pre:
            if i>=0.5:
                y_pre.append(1)
            else:
                y_pre.append(0)
        y_pre = np.array(y_pre)
        return y_pre
    def train0(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 0:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w0,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w0[j] -= rate*dw[j]/bs
    def train1(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 1:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w1,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w1[j] -= rate*dw[j]/bs
    def train2(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 2:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w2,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w2[j] -= rate*dw[j]/bs
    def train3(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 3:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w3,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w3[j] -= rate*dw[j]/bs
    def train4(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 4:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w4,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w4[j] -= rate*dw[j]/bs
    def train5(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 5:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w5,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w5[j] -= rate*dw[j]/bs
    def train6(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 6:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w6,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w6[j] -= rate*dw[j]/bs
    def train7(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 7:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w7,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w7[j] -= rate*dw[j]/bs
    def train8(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 8:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w8,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w8[j] -= rate*dw[j]/bs
    def train9(self,rate,iterations,bs):
        for i in range(iterations):
            # print(i)
            # cost = self.cost_fun()
            # self.costs.append(cost)
            a = range(self.x.shape[1])
            a = rand.sample(a,bs)
            newx = self.x[:,a]
            newy = self.y[a]
            yk = []
            for i in range(len(newy)):
                if newy[i] == 9:
                    yk.append(1)
                else:
                    yk.append(0)
            newy = np.array(yk)
            y = np.dot(self.w9,newx)
            y_pre = self.predict(y)
            # if i==iterations-1:
            #     print(y_pre)
            dw = np.dot(newx,(y_pre-newy).T)
            for j in range(self.x.shape[0]):
                self.w9[j] -= rate*dw[j]/bs
    def train(self,rate,iterations,bs):
        self.train0(rate,iterations,bs)
        self.train1(rate,iterations,bs)
        self.train2(rate,iterations,bs)
        self.train3(rate,iterations,bs)
        self.train4(rate,iterations,bs)
        self.train5(rate,iterations,bs)
        self.train6(rate,iterations,bs)
        self.train7(rate,iterations,bs)
        self.train8(rate,iterations,bs)
        self.train9(rate,iterations,bs)
    def pred(self,newx):
        y0 = np.dot(self.w0,newx)
        y1 = np.dot(self.w1,newx)
        y2 = np.dot(self.w2,newx)
        y3 = np.dot(self.w3,newx)
        y4 = np.dot(self.w4,newx)
        y5 = np.dot(self.w5,newx)
        y6 = np.dot(self.w6,newx)
        y7 = np.dot(self.w7,newx)
        y8 = np.dot(self.w8,newx)
        y9 = np.dot(self.w9,newx)
        y = []
        for i in range(len(y0)):
            an = 0
            ma = -10000000000
            if y0[i]>ma:
                ma = y0[i]
                an = 0
            if y1[i]>ma:
                ma = y1[i]
                an = 1
            if y2[i]>ma:
                ma = y2[i]
                an = 2
            if y3[i]>ma:
                ma = y3[i]
                an = 3
            if y4[i]>ma:
                ma = y4[i]
                an = 4
            if y5[i]>ma:
                ma = y5[i]
                an = 5
            if y6[i]>ma:
                ma = y6[i]
                an = 6
            if y7[i]>ma:
                ma = y7[i]
                an = 7
            if y8[i]>ma:
                ma = y8[i]
                an = 8
            if y9[i]>ma:
                ma = y9[i]
                an = 9
            y.append(an)
        return y
    def error(self,newx):
        y_pre = self.pred(newx)
        sum =0
        for i in range(len(y_pre)):
            sum+=(y_pre[i]-self.y[i])**2
        return sum/len(y_pre)
lis = sys.argv
terx = lis[0]+"/"+lis[1]
tery = lis[2]+"/"+lis[3]
tes = lis[4]+"/"+lis[5]
ou = lis[6]+"/"+lis[7]
X = np.load(terx)
Y = np.load(tery)
X = X.T
# k = []
# for i in Y:
#     if i%2==1:
#         k.append(1)
#     else:
#         k.append(0)
# Y = k
# Y = np.array(Y)
on = [1 for i in range(X.shape[1])]
X = np.append(X,np.array([on]),axis=0)
re = Bonus(X,Y)
re.train(0.048,80,100)
X1 = np.load(tes)
X1 = X1.T
on1 = [1 for i in range(X1.shape[1])]
X1 = np.append(X1,np.array([on1]),axis=0)
yppre = re.pred(X1)

# y_pre = re.pred()
# plt.plot(y_pre[:20],'.')
# plt.plot(re.y[:20],'.')
# print(re.error())
f = open(ou,'w')
# print(np.array_str(yppre))
for ik in yppre:
    # print(i)
    f.write(np.str_(ik)+"\n")