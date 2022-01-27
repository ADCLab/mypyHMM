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

# emMat=array([[1/6,1/6,1/6,1/6,1/6,1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
emMat=array([[1/6,1/6,1/6,1/6,1/6,1/6],[1/8, 1/8, 1/8, 1/8, 1/8, 3/8]])

YsT=[[y] for y in Ys]
pi0=array([0.2, 0.8])
A=array([[0.70, 0.30],
        [0.05, 0.95]])


HMM1=myHMM(numStates=2,A=A,pi0=pi0)
HMM1.addEmission('discrete',numOutputsPerFeature=5,emMat=emMat)

HMM2=myHMM(numStates=2,A=A,pi0=pi0)
# HMM2.addEmission('discrete',numOutputsPerFeature=5,emMat=emMat)

# fitness1=HMM1.train_pool(YsT,iterations=50,method=None,FracOrNum=1)
# f1=HMM1.train_pool(YsT,iterations=100,method=None,FracOrNum=100)
# f2=HMM1.train_pool(YsT,iterations=100,method=None,FracOrNum=500)
# f3=HMM1.train_pool(YsT,iterations=100,method=None,FracOrNum=1000)
# f4=HMM1.train_pool(YsT,iterations=100,method=None,FracOrNum=2000)
# f5=HMM1.train_pool(YsT,iterations=100,method=None,FracOrNum=1)
fitness=[]
fit=HMM1.train_pool(YsT,iterations=1000,method=None,FracOrNum=1,printHMM=['A','pi0','emission'])
fitness=fitness+fit
# fitness2=HMM1.train(YsT,iterations=2,method='log')



# HMM=myHMM(numStates=2)
# HMM.addEmission('discrete',numOutputsPerFeature=6)

# sTime=time.time()
# fitness=HMM.train_pool(YsT,iterations=1,method='log',FracOrNum=1000)
# eTime=time.time()
# rtimes=eTime-sTime
# print('Pool Runtime: %f'%(eTime-sTime))
# plt.plot(fitness)


# sTime=time.time()
# fitness=HMM.train_pool(YsT,iterations=10,Ttrue=T,method=None,FracOrNum=1)
# eTime=time.time()
# rtimes=eTime-sTime
# print('Pool Runtime: %f'%(eTime-sTime))


# sTime=time.time()
# fitness=HMM.train(Ys=YsT,iterations=10,Ttrue=T,method='log')
# eTime=time.time()
# rtimes=eTime-sTime
# print('Pool Runtime: %f'%(eTime-sTime))

plt.plot(fitness)
