import sys
sys.path.append('../')

from MyHMMStateDistribution import myHMM, loadSequences
from matplotlib import pyplot as plt
from numpy import array



stateSeqs,emissionSeqs=loadSequences('data/HMM_Categorical_2Feats.njson')

pi0=array([0.15, 0.6, 0.4])
A=array([[0.7, 0.2, .1],
        [0.15, 0.7, .15],
        [.1, .2, .7]     
        ])


HMM=myHMM(numStates=3,A=A,pi0=pi0)

params=[{'distDim':5}]
HMM.addEmission(emType='discrete',emDists=['Categorical'],params=params)

params=[{'distMat':array([.1,.3,.6])}, {'distMat':array([.1,.8,.1])}, {'distMat':array([.8,.1,.1])}]
HMM.addEmission(emType='discrete',emDists=['Categorical'],params=params)

fit=HMM.train_pool(emissionSeqs,iterations=200,method='normalize',FracOrNum=1,printHMM=['A','pi0','emission'],numCores=3)
# fit=HMM.train(emissionSeqs,iterations=200,method='normalize',FracOrNum=500,printHMM=['A','pi0','emission'])

# plt.plot(fit)

