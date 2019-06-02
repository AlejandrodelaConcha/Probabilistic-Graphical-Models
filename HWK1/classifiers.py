# -*- coding: utf-8 -*-
"""
Created on Fri May 10 04:28:51 2019

@author: Fraintendetore
"""

import numpy as np
import matplotlib.pyplot as plt

class classifier():
    def __init__(self,title):
        self.title=title
      
        
    def fit(self,x,y):
        self.x=x
        self.y=y
        pass
    
    def predict(self,x):
        pass
    
    def predictfit(self,xtrain,y,xtest):
        fit(xtrain,y)
        return(predict(xtest))
        
    def plot(self,x,y):
        x_min, x_max = np.min([(x[:, 0].min()),(self.x[:, 0].min())]) - 1, np.max([(x[:, 0].max()),(self.x[:, 0].max())]) + 1
        y_min, y_max = np.min([(x[:, 1].min()),(self.x[:, 1].min())]) - 1, np.max([(x[:, 1].max()),(self.x[:, 1].max())]) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr.contourf(xx, yy, Z, alpha=0.4)
        axarr.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
        axarr.set_title(self.title,fontsize=20)
        plt.show()
        return f,axarr
    
class LDA(classifier):
    ##### Function implementing the LDA algorithm for binary classification 
    def __init__(self):
        super(LDA, self).__init__("LDA")
 
    def fit(self,x,y):
        self.x=x
        self.y=y
        N=len(y)
        index=np.where(y==1,True,False)
        self.m0=np.sum(x[index],axis=0)/np.sum(y)
        self.m1=np.sum(x[~ index],axis=0)/np.sum(1-y)
        self.sigma=0
        
        for i in range(N):
            self.sigma+=np.outer(x[i],x[i])
            
        self.sigma/=N
        self.precision=np.linalg.inv(self.sigma)
    
    def predict(self,x):
        N=x.shape[0]
        score0=np.zeros(N)
        score1=np.zeros(N)
        for i in range(N):
            score0[i]=np.exp(-(x[i]-self.m0).dot((self.precision).dot(x[i]-self.m0))/2)
            score1[i]=np.exp(-(x[i]-self.m1).dot((self.precision).dot(x[i]-self.m1))/2)
        
        prediction=np.where(score0/(score0+score1)>=0.5,1,0)
        return(prediction)
        
        

        

class QDA(classifier):
    ##### Function implementing the QDA algorithm for classification
    def __init__(self):
        super(QDA, self).__init__("QDA")
        
    def fit(self,x,y):
        self.x=x
        self.y=y
        N=len(y)
        index=np.where(y==1,True,False)
        self.m0=np.sum(x[index],axis=0)/np.sum(y)
        self.m1=np.sum(x[~ index],axis=0)/np.sum(1-y)
        self.sigma0=0
        self.sigma1=0
        
        for i in range(N):
            if y[i]==1:
                self.sigma0+=np.outer(x[i]-self.m0,x[i]-self.m0)
            else:
                self.sigma1+=np.outer(x[i]-self.m1,x[i]-self.m1)
        
        self.sigma0=self.sigma0/np.sum(y)
        self.sigma1=self.sigma1/np.sum(1-y)
        self.precision0=np.linalg.inv(self.sigma0)
        self.precision1=np.linalg.inv(self.sigma1)
        
    def predict(self,x):
        N=x.shape[0]
        score0=np.zeros(N)
        score1=np.zeros(N)
        for i in range(N):
             score0[i]=np.exp(-(x[i]-self.m0).dot((self.precision0).dot(x[i]-self.m0))/2)/(np.sqrt(np.linalg.det(self.sigma0)))
             score1[i]=np.exp(-(x[i]-self.m1).dot((self.precision1).dot(x[i]-self.m1))/2)/(np.sqrt(np.linalg.det(self.sigma1)))
        prediction=np.where(score0/(score0+score1)>=0.5,1,0)
        return(prediction)
        
       
   
        
class linear_regression(classifier):
    ##### FUnction performing linear regression
    def __init__(self):
        super(linear_regression, self).__init__("Linear Regression")
        
    def fit(self,x,y):
        self.x=x
        self.y=y
        N=len(y)
        design=np.hstack([np.ones((N,1)),x])
        beta=np.linalg.inv(design.transpose().dot(design)).dot(design.transpose().dot(y))
        self.intercept=beta[0]
        self.w=beta[1:]
    
    def predict(self,x):
        prediction=np.round(np.clip(x.dot(self.w)+self.intercept,0,1))
        return(prediction)
        


        
class logistic_regression(classifier):
    ### Function implementing logistic regression
    def __init__(self):
        super(logistic_regression, self).__init__("Logistic Regression")
      
    def fit(self,x,y,w_ini,tol=1e-6):
        self.x=x
        self.y=y
        N=len(y)
        design=np.hstack([np.ones((N,1)),x])
        w_old=w_ini
        
        #### IRLS
        eta=self.sigmoid(design.dot(w_old))
        gradient=design.transpose().dot(y-eta)        
        hessian=-1*design.transpose().dot(np.diag(eta*(1-eta))).dot(design)
        w_new=w_old-np.linalg.inv(hessian).dot(gradient)
        while(np.sqrt(np.sum(gradient**2))>tol):
            w_old=w_new
            eta=self.sigmoid(design.dot(w_old))
            gradient=design.transpose().dot(y-eta)
            hessian=-1*design.transpose().dot(np.diag(eta*(1-eta))).dot(design)
            w_new=w_old-np.linalg.inv(hessian).dot(gradient)
        
        self.intercept=w_new[0]
        self.w=w_new[1:]
          
    def predict(self,x):
        prediction=np.where(self.sigmoid(x.dot(self.w)+self.intercept)>=0.5,1,0)
        return(prediction)
        
    def sigmoid(self,z):
        return(1/(1+np.exp(-1.*z)))
        
          