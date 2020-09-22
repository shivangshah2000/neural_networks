import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

inp_df = pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\training_feature_matrix.xlsx", header=None).to_numpy()
y= pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\training_output.xlsx",header=None).to_numpy()

def cost(X,y,theta,lamb=100):
    m=len(y)
    predict= np.dot(X,theta)
    cost1 =  np.sum(np.square(predict-y))+ lamb*np.sum(np.square(theta))
    return cost1/2

def mse(X,y,theta):
    m = len(y)
    predict = np.dot(X, theta)
    cost1 = np.sum(np.square(predict - y))
    return cost1 / m

def grad_descent(X , y, n_iter, alpha,theta,lamb=10):
    m=len(y)
    cost_history = np.zeros(n_iter)
    theta_hist= np.zeros((n_iter,3))
    for iter1 in range(n_iter) :
        predict= np.dot(X,theta)
        theta =(1-alpha*lamb)* theta -  alpha*(X.T.dot((predict-y)))
        theta_hist[iter1 , :]= theta.T
        cost_history[iter1]= cost(X,y,theta)
    return theta, cost_history, theta_hist

def stochastic_grad_descent(X , y, n_iter, alpha,theta,lamb=10):
    m=len(y)
    cost_history= np.zeros(n_iter)
    for it in range(n_iter):
        cost_sum=0
        for i in range(m):
            rand_int= np.random.randint(0,m)
            X_i= X[rand_int,:].reshape(1,X.shape[1])
            y_i= y[rand_int].reshape(1,1)
            predict= np.dot(X_i,theta)
            theta=(1-alpha*lamb)*theta- alpha*(X_i.T.dot((predict-y_i)))
            cost_sum+= cost(X_i,y_i,theta)
        cost_history[it]=cost_sum
    return theta,cost_history

def mini_batch_grad_descent(X,y,theta_mini,alpha,n_iter,batch_size=32,lamb=10):
    m=len(y)
    cost_history = np.zeros(n_iter)
    n_batches= int(m/batch_size)
    for it in range(n_iter):
        cost1=0
        index= np.random.permutation(m)
        X=X[index]
        y=y[index]
        for i in range(0,m,batch_size):
            X_i= X[i:i+batch_size]
            y_i= y[i:i+batch_size]
            prediction=np.dot(X_i,theta_mini)
            theta_mini= (1-alpha*lamb)*theta_mini - alpha* (X_i.T.dot((prediction-y_i)))
            cost1 += cost(X_i,y_i,theta_mini)
        cost_history[it]=cost1
    return theta_mini,cost_history

m=y.shape[0]
X0= np.ones((m,1))
Xnew=np.hstack((X0,inp_df))
ncol= Xnew.shape[1]   # cal. no. of cols

alpha = 0.0001
theta= np.random.randn(ncol,1)
n_iter= 100

ym = np.mean(y)
ystd = np.std(y)
Xm1 = np.mean(Xnew[:,1])
Xstd1 = np.std(Xnew[:,1])
Xm2 = np.mean(Xnew[:,2])
Xstd2 = np.std(Xnew[:,2])

Xnew[:,1] = (Xnew[:,1]-np.mean(Xnew[:,1]))/np.std(Xnew[:,1])
Xnew[:,2] = (Xnew[:,2]-np.mean(Xnew[:,2]))/np.std(Xnew[:,2])
y=(y-ym)/ystd

n_iter=200
theta , cost_history , theta_hist = grad_descent(Xnew , y , n_iter, alpha, theta)
plt.plot(cost_history)
plt.show()

n_iter=40
theta_stoc=np.random.randn(ncol,1)
theta_stoc , cost_history_stoc = stochastic_grad_descent(Xnew , y , n_iter, alpha, theta_stoc)
plt.plot(cost_history_stoc)
plt.show()

theta_mini= np.random.randn(ncol,1)
n_iter=200
theta_mini, cost_history_mini = mini_batch_grad_descent(Xnew, y, theta_mini, alpha, n_iter)
plt.plot(cost_history_mini)
plt.show()

out_df = pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\test_feature_matrix.xlsx", header=None).to_numpy()
y1= pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\test_output.xlsx",header=None).to_numpy()

m1=y1.shape[0]
X= np.ones((m1,1))
Xt=np.hstack((X,out_df))

Xt[:,1] = (Xt[:,1]-Xm1)/Xstd1  #normalize all X and y
Xt[:,2] = (Xt[:,2]-Xm2)/Xstd2
y1=(y1-ym)/ystd

print(mse(Xt,y1,theta))
print(mse(Xt,y1,theta_stoc))
print(mse(Xt,y1,theta_mini))


