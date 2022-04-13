import sys
sys.path.append('../')

from MyHMMStateDistribution import myHMM, loadSequences
from matplotlib import pyplot as plt
from numpy import array



stateSeqs,emissionSeqs=loadSequences('data/HMM_Poisson_2Feats.njson')


pi0=array([0.15, 0.6, 0.4])
A=array([[0.7, 0.2, .1],
        [0.15, 0.7, .15],
        [.1, .2, .7]     
        ])
HMM=myHMM(numStates=3,A=A,pi0=pi0)

params=[{'lamb':1},{'lamb':5},{'lamb':10}]
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=params)

params=[{'lamb':10},{'lamb':5},{'lamb':1}]
HMM.addEmission(emType='discrete',emDists=['Poisson','Poisson','Poisson'],params=params)

fit=HMM.train_pool(emissionSeqs,iterations=200,method='normalize',FracOrNum=.1,printHMM=['A','pi0','emission'],numCores=3)
# fit=HMM.train(emissionSeqs,iterations=200,method=None,FracOrNum=500,printHMM=['A','pi0','emission'])

# plt.plot(fit)

