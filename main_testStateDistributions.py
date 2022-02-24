# from MyHMMscaled import myHMM
from MyHMMStateDistribution import myHMM
from matplotlib import pyplot as plt
from numpy import array,set_printoptions
import pandas
import time

set_printoptions(precision=5)


YsDF=pandas.read_csv('BW/observations_short.csv')

Ys=[]
for cRow in YsDF.iterrows():
    Ys.append(array(cRow[1]))

# emMat=array([[1/6,1/6,1/6,1/6,1/6,1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
emMat=[array([1/6,1/6,1/6,1/6,1/6,1/6]),array([1/8, 1/8, 1/8, 1/8, 1/8, 3/8])]

YsT=[[y] for y in Ys]
pi0=array([0.2, 0.8])
A=array([[0.70, 0.30],
        [0.05, 0.95]])


HMM1=myHMM(numStates=2,A=A,pi0=pi0)
HMM1.addEmission(emType='discrete',emDists=['Empirical'],params=emMat)
fitness=[]
fit=HMM1.train_pool(YsT,iterations=200,method=None,FracOrNum=5000,printHMM=['A','pi0','emission'],numcore=8)
fitness=fitness+fit
plt.plot(fitness)

