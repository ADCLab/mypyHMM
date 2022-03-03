# from MyHMMscaled import myHMM
from MyHMMStateDistribution import myHMM, loadSequences
from matplotlib import pyplot as plt
from numpy import array,set_printoptions, save
import pandas
import time
set_printoptions(precision=5)


stateSeqs,emissionSeqs=loadSequences('testingData/poissonTest_3Feats.njson')


pi0=array([0.15, 0.6, 0.4])
A=array([[0.7, 0.2, .1],
        [0.15, 0.7, .15],
        [.1, .2, .7]     
        ])


HMM=myHMM(numStates=3,A=A,pi0=pi0)
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[4, 2, 1])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[1, 3, 2])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[.25, .5, 3])

cStart=time.time()
optPaths=HMM.viterbi_pool(emissionSeqs)
cEnd=time.time()
print(cEnd-cStart)

# fit=HMM.train_pool(emissionSeqs,iterations=200,method=None,FracOrNum=1,printHMM=['A','pi0','emission'],numCores=8)
# fit=HMM.train(emissionSeqs,iterations=200,method=None,FracOrNum=500,printHMM=['A','pi0','emission'])

# # fitness=fitness+fit
# # plt.plot(fitness)

