import numpy as np

class Probability(object):

     def __init__(self,P,X):
          self.P = P
          self.X = X
          self.mu = 0.0
          self.sigmaSquared = 0.0

     def mean(self):

          n = len(P)
          m = len(X)

          if n != m:
               raise Exception('Number of Xi values do not match the number of Pi values!')

          for i in range(n):
               self.mu += self.P[i,0]*self.X[i,0]

     def variance(self):

          n = len(P)
          m = len(X)

          if n != m:
               raise Exception('Number of Xi values do not match the number of Pi values!')

          for i in range(n):
               self.sigmaSquared += self.P[i,0]*((self.X[i,0]-self.mu)**2)

class ProbabilityDistribution(object):

     def __init__(self,X,mu,sigma,distributionType):
          self.P = None
          self.X = X
          self.mu = mu
          self.sigma = sigma
          self.distributionType = distributionType

     def uniform(self):
          P = np.linspace(0,1,1000)
          P.reshape((len(P),1))

          for i in range(n):
               P[i,0] = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((self.X[i,0]-self.mu)**2)/(2*self.sigma**2))
               
          self.P = P
          
     def normal(self):
          P = np.linspace(0,1,1000)
          P.reshape((len(P),1))

          for i in range(n):
               P[i,0] = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((self.X[i,0]-self.mu)**2)/(2*self.sigma**2))
               
          self.P = P

     
     

     

