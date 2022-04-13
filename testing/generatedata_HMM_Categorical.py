import sys
sys.path.append('../')

from MyHMMStateDistribution import myHMM, saveSequences
from numpy import array,set_printoptions, save


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

Y=HMM.genSequences(NumSequences=5000,maxLength=20,method='iter',asList=True)

saveSequences(fout='data/HMM_Categorical_2Feats.njson',StateSeqs=Y[0],EmissionSeqs=Y[1])
    