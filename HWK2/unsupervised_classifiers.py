# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import multivariate_normal

class unsupervised_classifier():
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
        pass

class K_means(unsupervised_classifier):
    
    #### Algorithm implementing the k++ algorithm using the euclidean distance.
    def __init__(self,K):
        super(K_means, self).__init__("K-means")
        self.K=K
    
    def compute_distance(self,x,centroids):
        return np.sum((centroids-x)**2,axis=1)

    
    def fit(self,x):
        N=x.shape[0]
        label_old=np.zeros(N)
        label_new=np.zeros(N)
        
        index_ini=np.random.choice(N,size=self.K,replace=False)
        centroids=x[index_ini]
       
        for i in range(N):
           
            label_new[i]=np.argmin(self.compute_distance(x[i],centroids))
        
        for k in range(self.K):
            index_K=np.where(label_new==k)[0]
            centroids[k]=np.mean(x[index_K],axis=0)
            
        while(np.sum(label_new-label_old)>0):
            label_old=label_new
            
            for i in range(N):
                label_new[i]=np.argmin(self.compute_distance(x[i],centroids))
        
            for k in range(self.K):
                index_K=np.where(label_new==k)[0]
                centroids[k]=np.mean(x[index_K])
        
        self.centroids=centroids
        return(label_new)
    
    def predict(self,x):
        N=x.shape[0]
        label_new=np.zeros(N)
        for i in range(N):
            label_new[i]=np.argmin(self.compute_distance(x[i],self.centroids))
        return(label_new)
        
        
        
    def plot(self,x,y):
        N=x.shape[0]
        r=np.zeros(self.K)
        
        fig, ax = plt.subplots()
        ax.scatter(x[:,0],x[:,1],c=y)
        
        for k in range(self.K):
            index_K=np.where(y==k)[0]
            r[k]=np.max(self.compute_distance(self.centroids[k],x[index_K]))
        ax.set_title(self.title,fontsize=20)
        plt.show()

        
    
class Gaussian_mixture(unsupervised_classifier):
    def __init__(self,K,isotropic=False):
        super(Gaussian_mixture, self).__init__("Gaussian mixture")
        self.K=K
        self.mu=[]
        self.sigma=[]
        self.alpha=np.zeros(self.K)
        self.isotropic=isotropic
    
    def loglikelihood(self,x):
        aux_loglike=np.zeros((x.shape[0],len(self.alpha)))
        if self.isotropic:
            for j in range(len(self.alpha)):
                    aux_loglike[:,j]=self.alpha[j]*multivariate_normal.pdf(x,self.mu[j],self.sigma[j]*np.eye(x.shape[1]))
        else:
            for j in range(len(self.alpha)):
                    aux_loglike[:,j]=self.alpha[j]*multivariate_normal.pdf(x,self.mu[j],self.sigma[j])
       
            
        aux_loglike=np.apply_along_axis(lambda x: np.log(np.sum(x)),1,aux_loglike)
        loglike=np.sum(aux_loglike)
        return(loglike)
        
    def E_step(self,x):
        tau=np.zeros((x.shape[0],self.K))
        if self.isotropic:
            for j in range(self.K):
                tau[:,j]=self.alpha[j]*multivariate_normal.pdf(x,self.mu[j],self.sigma[j]*np.eye(x.shape[1]))
        else:
            for j in range(self.K):
                tau[:,j]=self.alpha[j]*multivariate_normal.pdf(x,self.mu[j],self.sigma[j])
       
            
        
        denominator=np.apply_along_axis(np.sum,1,tau)
        for j in range(self.K):
            tau[:,j]=tau[:,j]/denominator
        return tau
            
    def M_step(self,x,tau):
        N=x.shape[0]
        self.mu=[]
        self.sigma=[]
        self.alpha=np.zeros(self.K)
        for j in range(self.K):
            self.alpha[j]=np.mean(tau[:,j])
            self.mu.append(np.apply_along_axis(lambda x: np.average(x,weights=tau[:,j]),0,x))     
            W=np.identity(x.shape[0])
            np.fill_diagonal(W,tau[:,j])
            data_centered=x-self.mu[j]
            
            if self.isotropic:
                self.sigma.append(np.trace(np.transpose(data_centered).dot(W).dot(data_centered))/(2*np.sum(tau[:,j])))
            
            else:  
                sigma=0
                for i in range(N):
                    sigma+=tau[i,j]*np.outer(data_centered[i],data_centered[i])
                self.sigma.append(sigma/np.sum(tau[:,j]))
        
               
    
    def fit(self,x,max_iter=100):
        N=x.shape[0]
       
        ### Initialization
        initialize=K_means(self.K)
        label_old=initialize.fit(x)
        
        loglike_seq=np.zeros(max_iter+1)
        
        for k in range(self.K):
            index_K=np.where(label_old==k)[0]
            self.mu.append(np.apply_along_axis(np.mean,0,x[index_K,:]))
            if self.isotropic:
                self.sigma.append(np.var(x[index_K,:]))
            
            else:
                self.sigma.append(np.cov(x[index_K,0],x[index_K,1]))
            self.alpha[k]=(1.*len(index_K))/(1.*N)
           
        n_iter=0
        loglike_seq[n_iter]=self.loglikelihood(x)
        
        while(n_iter<max_iter):
        ## E-step
            tau=self.E_step(x)
           
        ## M-step
            self.M_step(x,tau)
            n_iter=n_iter+1
            loglike_seq[n_iter]=self.loglikelihood(x)
            
        label=np.apply_along_axis(np.argmax,1,tau)
        return label
       
    
    def predict(self,x):
       
        tau=E_step(x)
        label=np.apply_along_axis(np.argmax,1,self.tau)
        return label
    
    def plot(self,x,y):
    ## Function to draw the Gaussian contour at 90%.
   
    ## Input
    # data=data set 
    # max iter= maximum number of iteractions.
    # y= label of the data
    # mu= list with mean values
    # sigma= list with covariate matrices
    # k= number of clusters to find

        plt.figure()
        plt.scatter(x[:,0],x[:,1],c=y)
        s=4.6057
    
        for j in range(self.K):  
            if self.isotropic:
                [V, D] = np.linalg.eig(self.sigma[j] *np.eye(x.shape[1])*s)
            else:
                [V, D] = np.linalg.eig(self.sigma[j] * s)
            t = np.linspace(0,2*np.pi)
            vec = np.array([np.cos(t),np.sin(t)])
            VD = D@np.sqrt(np.diag(V))
            z=VD@vec+self.mu[j].reshape(-1,1)
            plt.plot(z[0,:],z[1,:])    
  
        
