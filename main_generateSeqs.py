# from MyHMMscaled import myHMM
from MyHMMlog import myHMM
from matplotlib import pyplot as plt
from numpy import array,set_printoptions
import pandas
import time
set_printoptions(precision=5)


emMat=array([[1/6,1/6,1/6,1/6,1/6,1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
pi0=array([0.25, 0.75])
T=array([[0.75, 0.25],
        [0.1, 0.9]])


HMM=myHMM(numStates=2,T=T,pi0=pi0)
HMM.addEmission('discrete',numOutputsPerFeature=5,emMat=emMat)


NumLoops=15
NumSequences=1000
sTime=time.time()
for i in range(NumLoops):
    Y=HMM.genSequences(NumSequences=NumSequences,method='iter')
eTime=time.time()
rtimes=eTime-sTime
print('Itar Runtime: %f'%(eTime-sTime))


sTime=time.time()
for i in range(NumLoops):
    Y=HMM.genSequences(NumSequences=NumSequences,method='map')
eTime=time.time()
rtimes=eTime-sTime
print('Map Runtime: %f'%(eTime-sTime))


sTime=time.time()
for i in range(NumLoops):
    Y=HMM.genSequences(NumSequences=NumSequences,method='pool',numcore=6)
eTime=time.time()
rtimes=eTime-sTime
print('Pool Runtime: %f'%(eTime-sTime))