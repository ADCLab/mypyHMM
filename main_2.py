from MyHMM import myHMM
from numpy import array

mod=myHMM(numStates=3)
T=mod.T
pi0=mod.pi0
mod.addEmission(emType='discrete',numOutputsPerFeature=2)
# mod.addEmission(emType='discrete',numOutputsPerFeature=2)
X,Y=mod.genSequences(NumSequences=1,maxLength=20)

Y[0][0]=array([0,0,0,0,0,1,1,0,0,0])
HMM=myHMM(numStates=3,T=array([[.5,.5],[.3, .7]]),pi0=array([.2,.8]))
HMM.addEmission('discrete',numOutputsPerFeature=5,emMat=array([[.3,.7],[.8,.2]]))
HMM.train(Ys=Y,iterations=1,Ttrue=T)

# aa=myHMM(numStates=5)
# aa.addEmission('discrete',numOutputsPerFeature=4)