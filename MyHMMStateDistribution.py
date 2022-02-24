import numpy
from numpy import array, prod, empty, multiply, dot, ones, fliplr, matmul
from numpy import log, exp, sum, array2string
from numpy import array, random, diag, einsum, zeros, log, inf
from numpy.linalg import eigh, inv, norm
from scipy.special import logsumexp
import os
from multiprocessing import Pool
from itertools import repeat
from random import shuffle

#Log-sum-exp trick
def logsumexptrick(x):
    c = x.max()
    return c + log(sum(exp(x - c)))




def calc_logalpha(log_A,log_pi0,log_probObsState):
    Tsteps=log_probObsState.shape[1]
    numStates=log_A.shape[0]
    log_alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    log_alpha[:,0]=log_pi0+log_probObsState[:,0]
    for t in range(1,Tsteps):
        for i in range(numStates):
            terms=log_probObsState[i,t]+log_alpha[:,t-1]+log_A[:,i]
            log_alpha[i,t]=logsumexptrick(terms)
    return log_alpha



def calc_alpha_scale1(A,pi0,probObsState,rescale=True):
    Tsteps=probObsState.shape[1]
    numStates=A.shape[0]
    alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    alpha[:,0]=einsum('i,i->i',pi0,probObsState[:,0])
    if rescale:
        alpha[:,0]=alpha[:,0]/alpha[:,0].sum()
    for t in range(1,Tsteps):
        alpha[:,t]=einsum('j,ji,i->i',alpha[:,t-1],A,probObsState[:,t])
        if rescale:
            alpha[:,t]=alpha[:,t]/alpha[:,t].sum()
    return alpha

def calc_logbeta(log_A,log_probObsState):
    Tsteps=log_probObsState.shape[1]
    numStates=log_A.shape[0]
    log_beta=zeros((numStates,Tsteps))
    log_probObsState=fliplr(log_probObsState)
    for tb in range(1,Tsteps):
        for i in range(numStates):
            terms=log_probObsState[:,tb-1]+log_beta[:,tb-1]+log_A[i,:]
            log_beta[i,tb]=logsumexptrick(terms)
    return fliplr(log_beta)

def calc_beta_scale1(A,probObsState,rescale=True):
    Tsteps=probObsState.shape[1]
    numStates=A.shape[0]
    beta=empty((numStates,Tsteps))
    beta[:,0]=numStates*[1]
    if rescale:
        beta[:,0]=beta[:,0]/beta[:,0].sum()
    probObsState=fliplr(probObsState)
    for tb in range(1,Tsteps):
        beta[:,tb]=einsum('j,ij,j->i',beta[:,tb-1],A,probObsState[:,tb-1])
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

def calc_zeta(A,alpha,beta,probStateObs,method=None):
    Tsteps=probStateObs.shape[1]
    numStates=A.shape[0]
    if method=='log':
        topPart=array([[[alpha[i,t]+A[i,j]+beta[j,t+1]+probStateObs[j,t+1] for t in range(Tsteps-1)] for j in range(numStates)] for i in range(numStates)])
        bottomPart=array([logsumexptrick(cMat.reshape(-1)) for cMat in topPart.T])
        eta=exp(topPart-bottomPart)
        eta=array([cMat/cMat.sum() for cMat in eta.T]).T
    else:
        topPart=einsum('it,ij,jt,jt->ijt',alpha[:,0:-1],A,beta[:,1:],probStateObs[:,1:])
        # bottomPart=einsum('kt,kw,wt,wt->t',alpha[:,0:-1],A,beta[:,1:],probStateObs[:,1:])
        # eta=einsum('ijt,t->ijt',topPart,1/bottomPart)
        eta=einsum('ijt,t->ijt', topPart,1/topPart.sum(axis=(0,1)))
    return eta

def viterbi(obs,HMM):
    Tsteps=len(obs)
    delta=empty((HMM.numStates,Tsteps))
    psi=empty((HMM.numStates,Tsteps))
    delta[:,0]=multiply(HMM.pi0,[HMM.emission.probStateObs(cState,obs[:,0]) for cState in range(HMM.numStates)])
    psi[:,0]=HMM.numStates*[-1]
    for t in range(Tsteps-1):
        bestPathCostSoFar=array([multiply(delta[:,t], cCol) for cCol in HMM.A.T]).T.max(axis=0)
        probStateObs=[HMM.emission.probStateObs(cState,obs[:,t+1]) for cState in range(HMM.numStates)]
        delta[:,t+1]=multiply(bestPathCostSoFar,probStateObs)
        psi[:,t+1]=array([multiply(delta[:,t], cCol) for cCol in HMM.A.T]).T.argmax(axis=0)
    optPath=(Tsteps+1)*[None]
    optPath[Tsteps]=delta[:,Tsteps-1].argmax()
    for t in range(Tsteps-1,0,-1):
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
    



class discreteEmpiricalEmission():
    emType='empirical'
    def __init__(self,distMat=None,distDim=None):
        self.distMat=None
        self.distDim=None
                                  
        if distMat is None:
            if distDim is None:
                raise MyValidationError("Must provide Emission distribution or dimension of output distribution associated with state")
            else:
                A=random.rand(distDim)
                self.distMat=(A/A.sum())
        elif isinstance(distMat,(numpy.ndarray, numpy.generic)) and (distMat.ndim==1):
            self.distMat=distMat
        else:
            raise MyValidationError("Emission distribution not valid type or shape")
        self.distDim=self.distMat.shape[0]
    
    def generateEmission(self):
        return numpy.random.choice(self.distDim,1,p=self.distMat)[0]

    def probObs(self,Obs):
        return array([self.distMat[cObs] for cObs in Obs]).T

    def calcSingleTopPart(self,obs,state_gamma):
        Tsteps=obs.shape[0]
        Indicator=zeros((self.distDim, Tsteps))
        Indicator[obs,range(Tsteps)]=1
        topPart=einsum('kt,t->k',Indicator,state_gamma)
        return topPart

    def updateEmissionDist(self,topParts_unpool2state):
        topPartSum=array(topParts_unpool2state).sum(axis=0)
        self.distMat=topPartSum.T/topPartSum.sum()
        
    def printEmission(self,stateNum=None):
        stateStr=''
        if stateNum is not None:
            stateStr='State %d: '%stateNum
        print(stateStr+array2string(self.distMat,precision=5))   

# class Emission():
#     def __init__(self):
#         self.properties=None
#         self.emType=None

class discreteEmission():
    def __init__(self,numStates,emDists=['Empirical'],params=[None],emDims=[None]):                                  
        self.numStates=numStates
        self.state={}
        if numStates!=len(emDists):
            emDists=numStates*emDists
        if numStates!=len(params):
            params=numStates*params
        if numStates!=len(emDims):
            emDims=numStates*emDims
        
        for numState,emDist,parameter,enDim in zip(range(numStates),emDists,params,emDims):
            self.state[numState]=self.addStateEmission(emDist,parameter,enDim)
        
    
    def addStateEmission(self,emDist,parameter,enDim):
        if emDist=='Empirical':
            return discreteEmpiricalEmission(parameter,enDim)
    
    def generateEmission(self,stateSeq):
        return array( [self.state[cState].generateEmission() for cState in stateSeq] )

    def probObs(self,Obs):
        return array([self.probStateObs(cState,Obs) for cState in range(self.numStates)])
            
    def probStateObs(self,cState,Obs):
        return self.state[cState].probObs(Obs)

    def calcSingleTopPart(self,obs,gamma):
        topParts=[self.state[cState].calcSingleTopPart(obs,gamma[cState,:]) for cState in range(self.numStates)]
        return topParts

    def updateEmissionDist(self,topParts_unpool2state):
        [self.state[cState].updateEmissionDist(topParts_unpool2state[cState]) for cState in range(self.numStates)]
        
    def printEmission(self):
        [self.state[cState].printEmission(cState) for cState in range(self.numStates)] 
        
        
            
class myHMM():
    def __init__(self, A=None,numStates=None,pi0=None):
        self.A=None
        self.log_A=None
        self.numStates=None
        self.numOutputFeatures=0
        self.emission=[]
        self.pi0=None
        self.log_pi0=None
        if isinstance(numStates,int) and (A is None):
            tmpMat=random.rand(numStates,numStates)
            A=(tmpMat/tmpMat.sum(axis=0)).T
        elif A is not None:
            pass
        else:
            raise MyValidationError("Must provide Transition Matrix or number of states")
        self.updateA(A)
        self.updatePi0(pi0)
                          
    def updateA(self,A):
        if isinstance(A,(numpy.ndarray, numpy.generic)) and A.shape[0]==A.shape[1]:
            self.A=A 
            self.log_A=log(A)
            self.numStates=self.A.shape[0]  
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
        elif isinstance(pi0,(numpy.ndarray, numpy.generic)) and self.A.shape[0]==self.numStates:
            pi0=pi0/pi0.sum()
            self.pi0=pi0
        else:
            raise MyValidationError("Something wrong with pi0 input")
        self.log_pi0=log(self.pi0)
            
    def genSequences(self,NumSequences=100,maxLength=100,CheckAbsorbing=False,method='iter',numcore=1):
        if method=='iter':
            stateSeqs=[self.genStateSequence(maxLength,CheckAbsorbing) for x in range(NumSequences)]
            outputSeqs=[self.genOutputSequence(stateSeq) for stateSeq in stateSeqs]
        elif method=='map':
            stateSeqs=list(map(self.genStateSequence,NumSequences*[maxLength],NumSequences*[CheckAbsorbing]))
            outputSeqs=list(map(self.genOutputSequence,stateSeqs))
        elif method=='pool':
            with Pool(numcore) as pool:
                stateSeqs = pool.starmap(self.genStateSequence, zip(NumSequences*[maxLength],NumSequences*[CheckAbsorbing]))
                outputSeqs=pool.map(self.genOutputSequence,stateSeqs)      
        return stateSeqs, outputSeqs
    
    def genStateSequence(self,maxLength=100,CheckAbsorbing=False):
        stateSeq=[numpy.random.choice(self.numStates,p=self.pi0)]
        for cState in range(maxLength-1):
            if CheckAbsorbing and self.A[stateSeq[-1],stateSeq[-1]]==1:
                break
            stateSeq.append(numpy.random.choice(self.numStates,p=self.A[stateSeq[-1]]))
        return stateSeq
        
    def genOutputSequence(self,stateSeq):
        outputSeq=[]
        for cEmission in self.emission:
            outputSeq.append(cEmission.generateEmission(stateSeq))
        return outputSeq


    def calcLogProbObsState(self,obs,method='log'):
        if method=='log':
            return log(array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)])).sum(axis=0)  
        elif method==None:
            return array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)]).prod(axis=0)

    def train_pool(self,allYs,iterations=20,method='log',numcore=6,FracOrNum=None,printHMM=[]):
        lastLogProb=-inf
        fitness=[]
        for iter in range(iterations):
            Number=len(allYs)
            if FracOrNum<=1:
                Number=round(FracOrNum*Number)
            elif FracOrNum is not None:
                Number=FracOrNum
            
            shuffle(allYs)
            Ys=allYs[0:Number]
            b_topPart_pool=[]
            logProb=0

            with Pool(numcore) as pool:                
                log_probObsState_all = pool.map(self.calcLogProbObsState,allYs)
                log_alpha_all = pool.starmap(calc_logalpha,zip(repeat(self.log_A),repeat(self.log_pi0),log_probObsState_all))
                log_probObsState_pool = log_probObsState_all[0:Number]
                log_alpha_pool = log_alpha_all[0:Number]
                # log_probObsState_pool = pool.map(self.calcLogProbObsState,Ys)
                # log_alpha_pool = pool.starmap(calc_logalpha,zip(repeat(self.log_A),repeat(self.log_pi0),log_probObsState_pool))
                if method=='log':
                    log_beta_pool=pool.starmap(calc_logbeta,zip(repeat(self.log_A),log_probObsState_pool))
                    log_gamma_pool=pool.starmap(calc_gamma,zip(log_alpha_pool, log_beta_pool,repeat('log')))
                    log_zeta_pool=pool.starmap(calc_zeta,zip(repeat(self.log_A), log_alpha_pool, log_beta_pool, log_probObsState_pool,repeat('log')))
                    gamma_pool=log_gamma_pool
                    zeta_pool=log_zeta_pool
                else:
                    probObsState_pool= pool.starmap(self.calcLogProbObsState,zip(Ys,repeat(None)))
                    alpha_pool=pool.starmap(calc_alpha_scale1,zip(repeat(self.A),repeat(self.pi0),probObsState_pool,repeat(True)))
                    beta_pool=pool.starmap(calc_beta_scale1,zip(repeat(self.A),probObsState_pool,repeat(True)))
                    gamma_pool=pool.starmap(calc_gamma,zip(alpha_pool, beta_pool))
                    zeta_pool=pool.starmap(calc_zeta,zip(repeat(self.A), alpha_pool, beta_pool, probObsState_pool))   

                for cFeat in range(len(Ys[0])):
                    b_topPart_pool=pool.starmap(self.emission[cFeat].calcSingleTopPart,zip([obs[cFeat] for obs in Ys],gamma_pool))
                    b_topPart_unpool2state=list(map(list, zip(*b_topPart_pool)))
                    self.emission[cFeat].updateEmissionDist(b_topPart_unpool2state)

                    
            pi0_topPart=array([gamma[:,0] for gamma in gamma_pool]).sum(axis=0)
            A_topPart=array([zeta.sum(axis=2) for zeta in zeta_pool]).sum(axis=0)
            


                
                
                # b_bottomPart=b_bottomPart+gamma.sum(axis=1)
            logProb=logProb+sum([log_alpha[:,-1].sum() for log_alpha in log_alpha_all])
            fitness.append(logProb)
            ## Normally this should be pi0_topPart/len(Ys)
            ## This ensures pi0 sumers to 0.  
            pi0=pi0_topPart/pi0_topPart.sum()
            A=(A_topPart.T/A_topPart.sum(axis=1)).T               
            self.updateA(A)
            self.updatePi0(pi0)
            os.system('clear')
            print('= EPOCH #{} ='.format(iter))
            self.printHMM(printHMM)
            print('Fitness:', logProb)

        return fitness
        
    def printHMM(self,printHMM):
        if 'A' in printHMM:
            print('Transition Matrix:')
            print(self.A)
            print()

        if 'pi0' in printHMM:
            print('Initial Probability:')
            print(self.pi0)
            print()

        if 'emission' in printHMM:
            print('Emission Paramters:')
            for cFeat in range(len(self.emission)):
                self.emission[cFeat].printEmission()
            print()


    def train(self,Ys,iterations=20,method='log'):
        lastLogProb=-inf
        fitness=[]
        for iter in range(iterations):
            pi0_topPart=zeros((self.numStates))
            A_topPart=zeros((self.numStates,self.numStates))
            A_bottomPart=zeros((self.numStates))
            b_bottomPart=zeros((self.numStates))
            b_topParts=[zeros((cEmission.emMat.shape)) for cEmission in self.emission]
            logProb=0
            for obs in Ys:
                log_probObsState=log(array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)])).sum(axis=0)    
                log_alpha=calc_logalpha(self.log_A,self.log_pi0,log_probObsState)
                if method=='log':
                    log_beta=calc_logbeta(self.log_A,log_probObsState)
                    log_gamma=calc_gamma(log_alpha, log_beta,method='log')
                    log_zeta=calc_zeta(self.log_A, log_alpha, log_beta, log_probObsState,method='log')
                    gamma=log_gamma
                    eta=log_zeta
                else:
                    probObsState=array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)]).prod(axis=0)
                    alpha=calc_alpha_scale1(self.A,self.pi0,probObsState,rescale=True)
                    beta=calc_beta_scale1(self.A,probObsState,rescale=True)
                    gamma=calc_gamma(alpha, beta)
                    eta=calc_zeta(self.A, alpha, beta, probObsState)
                                
                pi0_topPart=pi0_topPart+gamma[:,0]
                A_topPart=A_topPart+eta.sum(axis=2)
                A_bottomPart=A_bottomPart+gamma[:,0:-1].sum(axis=1)
                
                
                b_bottomPart=b_bottomPart+gamma.sum(axis=1)
                logProb=logProb+log_alpha[:,-1].sum()
                for cFeat in range(len(obs)):
                    self.emission[cFeat].calcTopPart(obs[cFeat],gamma)

            fitness.append(logProb)
            R=len(Ys)
            pi0=pi0_topPart/R
            A=einsum('ij,i->ij',A_topPart,1/A_bottomPart)
            
            # emMat=einsum('ik,i->ik',b_topPart,1/b_bottomPart)
            self.updateA(A)
            self.updatePi0(pi0)
            for cFeat in range(len(obs)):
                self.emission[cFeat].updateEmission(b_bottomPart)
            # # for cEmission in self.emission:
            # #     cEmission.updateEmission(b_bottomPart)
            # # print(norm(self.pi0-pi00))
            # # if Ttrue is not None:
            # #     print(norm(self.A-Ttrue))
            # # if pi0true is not None:
            # #     print(norm(self.pi0-pi0true))            
            # # print(logProb)
            # print(self.A)
            # Some logging is always good :)
            os.system('clear')
            print('= EPOCH #{} ='.format(iter))
            print('Transition Matrix:')
            print(self.A)
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
        