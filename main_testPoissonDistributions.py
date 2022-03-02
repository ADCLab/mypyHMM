# from MyHMMscaled import myHMM
from MyHMMStateDistribution import myHMM, loadSequences
from matplotlib import pyplot as plt
from numpy import array,set_printoptions
import pandas
import time

set_printoptions(precision=5)


# stateSeqs,emissionSeqs=loadSequences('testingData/poissonTest.njson')
stateSeqs,emissionSeqs=loadSequences('testingData/poissonTest_3FeatsShort.njson')


pi0=array([0.15, 0.6, 0.4])
A=array([[0.7, 0.2, .1],
        [0.15, 0.7, .15],
        [.1, .2, .7]     
        ])


HMM=myHMM(numStates=3,A=A,pi0=pi0)
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[5, 2.5, 1])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[2, 2, 2])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[1, 1, 3])


fit=HMM.train_pool(emissionSeqs,iterations=200,method=None,FracOrNum=500,printHMM=['A','pi0','emission'],numcore=8)
# fit=HMM.train(emissionSeqs,iterations=200,method=None,FracOrNum=500,printHMM=['A','pi0','emission'])

# fitness=fitness+fit
# plt.plot(fitness)

