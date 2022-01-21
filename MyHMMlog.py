import numpy
from numpy import array, prod, empty, multiply, dot, ones, fliplr, matmul
from numpy import log, exp, sum
from numpy import array, random, diag, einsum, zeros, log, inf
from numpy.linalg import eigh, inv, norm
from scipy.special import logsumexp
import os

#Log-sum-exp trick
def logsumexptrick(x):
    c = x.max()
    return c + log(sum(exp(x - c)))


def calc_logalpha(log_T,log_pi0,log_probObsState):
    Tsteps=log_probObsState.shape[1]
    numStates=log_T.shape[0]
    log_alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    log_alpha[:,0]=log_pi0+log_probObsState[:,0]
    for t in range(1,Tsteps):
        for i in range(numStates):
            terms=log_probObsState[i,t]+log_alpha[:,t-1]+log_T[:,i]
            log_alpha[i,t]=logsumexptrick(terms)
    return log_alpha



def calc_alpha_scale1(T,pi0,probObsState,rescale=True):
    Tsteps=probObsState.shape[1]
    numStates=T.shape[0]
    alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    alpha[:,0]=einsum('i,i->i',pi0,probObsState[:,0])
    if rescale:
        alpha[:,0]=alpha[:,0]/alpha[:,0].sum()
    for t in range(1,Tsteps):
        alpha[:,t]=einsum('j,ji,i->i',alpha[:,t-1],T,probObsState[:,t])
        if rescale:
            alpha[:,t]=alpha[:,t]/alpha[:,t].sum()
    return alpha

def calc_logbeta(log_T,log_probObsState):
    Tsteps=log_probObsState.shape[1]
    numStates=log_T.shape[0]
    log_beta=zeros((numStates,Tsteps))
    log_probObsState=fliplr(log_probObsState)
    for tb in range(1,Tsteps):
        for i in range(numStates):
            terms=log_probObsState[:,tb-1]+log_beta[:,tb-1]+log_T[i,:]
            log_beta[i,tb]=logsumexptrick(terms)
    return fliplr(log_beta)

def calc_beta_scale1(T,probObsState,rescale=True):
    Tsteps=probObsState.shape[1]
    numStates=T.shape[0]
    beta=empty((numStates,Tsteps))
    beta[:,0]=numStates*[1]
    if rescale:
        beta[:,0]=beta[:,0]/beta[:,0].sum()
    probObsState=fliplr(probObsState)
    for tb in range(1,Tsteps):
        beta[:,tb]=einsum('j,ij,j->i',beta[:,tb-1],T,probObsState[:,tb-1])
        if rescale:
            beta[:,tb]=beta[:,tb]/beta[:,tb].sum()
    return fliplr(beta)

def calc_gamma(alpha,beta,method=None):
    if method=='log':
        topPart=alpha+beta
        bottomPart=[logsumexptrick(cCol) for cCol in topPart.T]
        gamma=exp(topPart-bottomPart)
        gamma=gamma/gamma.sum(axis=0)
    else:
        topPart=einsum('it,it->it',alpha,beta)
        gamma=einsum('it,t->it',topPart,1/topPart.sum(axis=0))
    return gamma

def calc_eta(T,alpha,beta,probStateObs,method=None):
    Tsteps=probStateObs.shape[1]
    numStates=T.shape[0]
    if method=='log':
        topPart=array([[[alpha[i,t]+T[i,j]+beta[j,t+1]+probStateObs[j,t+1] for t in range(Tsteps-1)] for j in range(numStates)] for i in range(numStates)])
        bottomPart=array([logsumexptrick(cMat.reshape(-1)) for cMat in topPart.T])
        eta=exp(topPart-bottomPart)
        eta=array([cMat/cMat.sum() for cMat in eta.T]).T
    else:
        topPart=einsum('it,ij,jt,jt->ijt',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
        # bottomPart=einsum('kt,kw,wt,wt->t',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
        # eta=einsum('ijt,t->ijt',topPart,1/bottomPart)
        eta=einsum('ijt,t->ijt', topPart,1/topPart.sum(axis=(0,1)))
    return eta

def viterbi(obs,HMM):
    T=len(obs)
    delta=empty((HMM.numStates,T))
    psi=empty((HMM.numStates,T))
    delta[:,0]=multiply(HMM.pi0,[HMM.emission.probStateObs(cState,obs[:,0]) for cState in range(HMM.numStates)])
    psi[:,0]=HMM.numStates*[-1]
    for t in range(T-1):
        bestPathCostSoFar=array([multiply(delta[:,t], cCol) for cCol in HMM.T.T]).T.max(axis=0)
        probStateObs=[HMM.emission.probStateObs(cState,obs[:,t+1]) for cState in range(HMM.numStates)]
        delta[:,t+1]=multiply(bestPathCostSoFar,probStateObs)
        psi[:,t+1]=array([multiply(delta[:,t], cCol) for cCol in HMM.T.T]).T.argmax(axis=0)
    optPath=(T+1)*[None]
    optPath[T]=delta[:,T-1].argmax()
    for t in range(T-1,0,-1):
        print(t)
        print(optPath)
        optPath[t]=int(psi[optPath[t+1],t])
    return delta,psi,optPath

    
            
            
def calc_ProbObs(obs,HMM):
    return sum(calc_alpha(obs,HMM)[:,-1])


class MyValidationError(Exception):
    pass

class MarkovSeqLibrary():
    def __init__(self):
        self.__stateSeqs=[]
        self.__outputSeqs=[]

    def addSeq(self,stateSeq,outputSeq):
        self.__stateSeqs.append(stateSeq)
        self.__outputSeqs.append(outputSeq)
        
        
        

class MarkovSeq():
    numSeqs=None
    stateSeq=None
    outputSeq=None
    def __init__(self,stateSeq,outputSeq):
        self.__stateSeq=stateSeq
        self.__outputSeq=outputSeq
        
    @property
    def state(self):
        return self.__stateSeq
    
    @property
    def output(self):
        return self.__outputSeq
    

class Emission():
    def __init__(self):
        self.properties=None
        self.emType=None

class discreteEmission(Emission):


    
    def __init__(self,numStates,emMat=None,numOutputsPerFeature=None):
        self.emMat=None
        self.numOutputsPerFeature=None
                                  
        if emMat is None:
            if numOutputsPerFeature is None:
                raise MyValidationError("Must provide Emission matrix or numOutputFeatures and numOutputsPerFeature")
            else:
                A=random.rand(numStates,numOutputsPerFeature)
                self.emMat=(A.T/A.sum(axis=1)).T
        elif isinstance(emMat,(numpy.ndarray, numpy.generic)) and (emMat.ndim==2) and (emMat.shape[0]==numStates):
            self.emMat=emMat
        else:
            raise MyValidationError("Emission matrix not valid type or shape")
        self.emType='discrete'
        self.numOutputsPerFeature=self.emMat.shape[1]
        self.topPart=zeros(self.emMat.shape) 
    
    def generateEmission(self,stateSeq):
        return numpy.array([numpy.random.choice(self.numOutputsPerFeature,p=self.emMat[cState]) for cState in stateSeq])

    def probObs(self,Obs):
        return array([self.emMat[:,cObs] for cObs in Obs]).T
    
    def probStateObs(self,cState,cObs):
        return self.emMat[cState,cObs]
        
    def calcTopPart(self,obs,gamma):
        Tsteps=obs.shape[0]
        Indicator=zeros((self.numOutputsPerFeature, Tsteps))
        Indicator[obs,range(Tsteps)]=1
        topPart=einsum('kt,it->ik',Indicator,gamma)
        self.topPart=self.topPart+topPart

    def updateEmission(self,bottomPart):
        self.emMat = (self.topPart.T/self.topPart.sum(axis=1)).T
        self.topPart=zeros(self.emMat.shape) 

    
        
        
            
class myHMM():
    def __init__(self, T=None,numStates=None,pi0=None):
        self.T=None
        self.log_T=None
        self.numStates=None
        self.numOutputFeatures=0
        self.emission=[]
        self.pi0=None
        self.log_pi0=None
        if isinstance(numStates,int) and (T is None):
            tmpMat=random.rand(numStates,numStates)
            T=(tmpMat/tmpMat.sum(axis=0)).T
        elif T is not None:
            pass
        else:
            raise MyValidationError("Must provide Transition Matrix or number of states")
        self.updateT(T)
        self.updatePi0(pi0)
                          
    def updateT(self,T):
        if isinstance(T,(numpy.ndarray, numpy.generic)) and T.shape[0]==T.shape[1]:
            self.T=T 
            self.log_T=log(T)
            self.numStates=self.T.shape[0]  
        else:
            raise MyValidationError("Unexpected Transition Matrix")
                   
    def addEmission(self, emType=None,**kargs):
        if self.numStates is None:
            raise MyValidationError("Must first define transition matrix or number of states")
            
        if emType=='discrete':
            newEmission=discreteEmission(self.numStates,**kargs)
            self.emission.append(newEmission)
        else:
            raise MyValidationError("Must provide valid emission type")
        self.numOutputFeatures=self.numOutputFeatures+1

    def updatePi0(self, pi0=None,useSteadyState=True):
        if pi0 is None:
            # if useSteadyState:
                
            #     else
            tmpMat=random.rand(self.numStates)
            pi0=tmpMat/tmpMat.sum()
         

            self.pi0=pi0
        elif isinstance(pi0,(numpy.ndarray, numpy.generic)) and self.T.shape[0]==self.numStates:
            pi0=pi0/pi0.sum()
            self.pi0=pi0
        else:
            raise MyValidationError("Something wrong with pi0 input")
        self.log_pi0=log(self.pi0)
        


            
    def genSequences(self,NumSequences=100,maxLength=100,CheckAbsorbing=False):
        stateSeqs=[self.genStateSequence(maxLength,CheckAbsorbing) for x in range(NumSequences)]
        outputSeqs=[self.genOutputSequence(stateSeq) for stateSeq in stateSeqs]
        return stateSeqs, outputSeqs
        # return([[stateSeq,outputSeq] for stateSeq,outputSeq in zip(stateSeqs,outputSeqs)])
    
    def genStateSequence(self,maxLength=100,CheckAbsorbing=False):
        stateSeq=[numpy.random.choice(self.numStates,p=self.pi0)]
        for cState in range(maxLength-1):
            if CheckAbsorbing and self.T[stateSeq[-1],stateSeq[-1]]==1:
                break
            stateSeq.append(numpy.random.choice(self.numStates,p=self.T[stateSeq[-1]]))
        return stateSeq
        
    def genOutputSequence(self,stateSeq):
        outputSeq=[]
        for cEmission in self.emission:
            outputSeq.append(cEmission.generateEmission(stateSeq))
        return outputSeq


    


    def train(self,Ys,iterations=20,Ttrue=None,pi0true=None,method='log'):
        lastLogProb=-inf
        fitness=[]
        for iter in range(iterations):
            pi0_topPart=zeros((self.numStates))
            T_topPart=zeros((self.numStates,self.numStates))
            T_bottomPart=zeros((self.numStates))
            b_bottomPart=zeros((self.numStates))
            b_topParts=[zeros((cEmission.emMat.shape)) for cEmission in self.emission]
            logProb=0
            for obs in Ys:
                log_probObsState=log(array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)])).sum(axis=0)    
                log_alpha=calc_logalpha(self.log_T,self.log_pi0,log_probObsState)
                if method=='log':
                    log_beta=calc_logbeta(self.log_T,log_probObsState)
                    log_gamma=calc_gamma(log_alpha, log_beta,method='log')
                    log_eta=calc_eta(self.log_T, log_alpha, log_beta, log_probObsState,method='log')
                    gamma=log_gamma
                    eta=log_eta
                else:
                    probObsState=array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)]).prod(axis=0)
                    alpha=calc_alpha_scale1(self.T,self.pi0,probObsState,rescale=True)
                    beta=calc_beta_scale1(self.T,probObsState,rescale=True)
                    gamma=calc_gamma(alpha, beta)
                    eta=calc_eta(self.T, alpha, beta, probObsState)
                                
                pi0_topPart=pi0_topPart+gamma[:,0]
                T_topPart=T_topPart+eta.sum(axis=2)
                T_bottomPart=T_bottomPart+gamma[:,0:-1].sum(axis=1)
                
                
                b_bottomPart=b_bottomPart+gamma.sum(axis=1)
                logProb=logProb+log_alpha[:,-1].sum()
                for cFeat in range(len(obs)):
                    self.emission[cFeat].calcTopPart(obs[cFeat],gamma)

            fitness.append(logProb)
            R=len(Ys)
            pi0=pi0_topPart/R
            T=einsum('ij,i->ij',T_topPart,1/T_bottomPart)
            
            # emMat=einsum('ik,i->ik',b_topPart,1/b_bottomPart)
            self.updateT(T)
            self.updatePi0(pi0)
            for cFeat in range(len(obs)):
                self.emission[cFeat].updateEmission(b_bottomPart)
            # # for cEmission in self.emission:
            # #     cEmission.updateEmission(b_bottomPart)
            # # print(norm(self.pi0-pi00))
            # # if Ttrue is not None:
            # #     print(norm(self.T-Ttrue))
            # # if pi0true is not None:
            # #     print(norm(self.pi0-pi0true))            
            # # print(logProb)
            # print(self.T)
            # Some logging is always good :)
            os.system('clear')
            print('= EPOCH #{} ='.format(iter))
            print('Transition Matrix:')
            print(self.T)
            print('Initial Dice Probability:', self.pi0)
            print('First Dice:', self.emission[0].emMat[0,:])
            print('Second Dice:', self.emission[0].emMat[1,:])
            print('Fitness:', logProb)
            print()  
        return fitness
        

        
                   
                 


def createRandomEmission(numStates,numOutputFeatures,NumOutputsPerFeature):
    if not isinstance(NumOutputsPerFeature,list):
        NumOutputsPerFeature=numStates*[NumOutputsPerFeature]
        
    for cState in range(numStates):
        random.randint(0,100,(numOutputFeatures,NumOutputsPerFeature))
        