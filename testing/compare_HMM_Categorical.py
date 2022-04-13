import sys
sys.path.append('../')

from MyHMMStateDistribution import myHMM, saveSequences, loadSequences
from numpy import array,set_printoptions, save
from hmmlearn.hmm import MultinomialHMM


maxIter=40
#----------------------------------------    
#----------------------------------------    
pi0=array([0.15, 0.6, 0.4])
A=array([[0.7, 0.2, .1],
        [0.15, 0.7, .15],
        [.1, .2, .7]     
        ])


HMM=myHMM(numStates=3,A=A,pi0=pi0)

params=[{'distMat':array([0.0838,  0.21388, 0.18264, 0.35745, 0.16222])},
         {'distMat':array([0.05298, 0.226,   0.27031, 0.2585,  0.19221])},
         {'distMat':array([0.2862,  0.11876, 0.28604, 0.19402, 0.11498])}]
HMM.addEmission(emType='discrete',emDists=['Categorical'],params=params)

Y=HMM.genSequences(NumSequences=5000,maxLength=20,method='iter',asList=True)

saveSequences(fout='data/HMM_Categorical_Compare.njson',StateSeqs=Y[0],EmissionSeqs=Y[1])
#----------------------------------------    
#----------------------------------------    

stateSeqs,emissionSeqs=loadSequences('data/HMM_Categorical_Compare.njson')
fit=HMM.train_pool(emissionSeqs,iterations=maxIter,method='normalize',FracOrNum=1,printHMM=['A','pi0','emission'],numCores=3)


HMM2=MultinomialHMM(n_components=3,implementation='scaling',n_iter=maxIter) 
HMM2.transmat_= A
HMM2.startprob_ = pi0