# from MyHMMscaled import myHMM
from MyHMMStateDistribution import myHMM, saveSequences
from matplotlib import pyplot as plt
from numpy import array,set_printoptions, save
import pandas
import time
set_printoptions(precision=5)


lambs=array([1, 5, 10])
pi0=array([0.15, 0.6, 0.4])
A=array([[0.7, 0.2, .1],
        [0.15, 0.7, .15],
        [.1, .2, .7]     
        ])


HMM=myHMM(numStates=3,A=A,pi0=pi0)
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[4, 2, 1])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[1, 3, 2])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=[.25, .5, 3])


Y=HMM.genSequences(NumSequences=5000,maxLength=8,method='iter',asList=True)

saveSequences(fout='testingData/poissonTest_3FeatsShort.njson',StateSeqs=Y[0],EmissionSeqs=Y[1])
    