import sys
sys.path.append('../')

from MyHMMStateDistribution import myHMM, saveSequences
from numpy import array

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

Y=HMM.genSequences(NumSequences=5000,maxLength=20,method='iter',asList=True)

saveSequences(fout='data/HMM_Poisson_2Feats.njson',StateSeqs=Y[0],EmissionSeqs=Y[1])
    