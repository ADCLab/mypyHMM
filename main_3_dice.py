# from MyHMMscaled import myHMM
from MyHMMlog import myHMM
from matplotlib import pyplot as plt
from numpy import array,set_printoptions
import pandas
import time

set_printoptions(precision=5)


YsDF=pandas.read_csv('BW/observations_short.csv')

Ys=[]
for cRow in YsDF.iterrows():
    Ys.append(array(cRow[1]))

emMat=array([[1/6,1/6,1/6,1/6,1/6,1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
YsT=[[y] for y in Ys]
pi0=array([0.25, 0.75])
T=array([[0.75, 0.25],
        [0.1, 0.9]])


# HMM=myHMM(numStates=2,T=T,pi0=pi0)
# HMM.addEmission('discrete',numOutputsPerFeature=5,emMat=emMat)

HMM=myHMM(numStates=2)
HMM.addEmission('discrete',numOutputsPerFeature=6)


sTime=time.time()
fitness=HMM.train_pool(YsT,iterations=200,Ttrue=T,method='log',FracOrNum=500)
eTime=time.time()
rtimes=eTime-sTime
print('Pool Runtime: %f'%(eTime-sTime))

sTime=time.time()
fitness=HMM.train_pool(YsT,iterations=5,Ttrue=T,method='log',FracOrNum=1)
eTime=time.time()
rtimes=eTime-sTime
print('Pool Runtime: %f'%(eTime-sTime))


# sTime=time.time()
# fitness=HMM.train(Ys=YsT,iterations=10,Ttrue=T,method='log')
# eTime=time.time()
# rtimes=eTime-sTime
# print('Pool Runtime: %f'%(eTime-sTime))


plt.plot(fitness)
