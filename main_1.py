#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:39:15 2021

@author: aev
"""

import numpy
from numpy import array, prod, empty, multiply, dot, ones, fliplr, matmul

from numpy import array, random, diag, einsum, zeros
from numpy.linalg import eigh, inv




def calc_alpha(obs,T,pi0,probObsState):
    Tsteps=obs[0].shape[0]
    numStates=T.shape[0]
    alpha=empty((numStates,Tsteps))
    alphaTransition=pi0
    # equivilent of elment-by-element multiplication
    alpha[:,0]=einsum('i,i->i',alphaTransition,probObsState[:,0])
    for t in range(1,Tsteps):
        alphaTransition=einsum('j,ji->j',alpha[:,0],T)
        alpha[:,t]=einsum('i,i->i',alphaTransition,probObsState[:,t])
    return alpha

def calc_beta(obs,T,probObsState):
    Tsteps=obs[0].shape[0]
    numStates=T.shape[0]
    beta=empty((numStates,Tsteps))
    beta[:,-1]=numStates*[1]
    for tb in range(-1,-Tsteps,-1):
        beta[:,tb-1]=einsum('j,ij,j->i',beta[:,tb],T,probObsState[:,-tb])
    return beta

def calc_gamma(alpha,beta):
    topPart=einsum('ij,ij->ij',alpha,beta)
    gamma=topPart/topPart.sum(axis=0)
    return gamma

def calc_eta(obs,T,alpha,beta,probStateObs):
    topPart=einsum('it,ij,jt,jt->ijt',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
    bottomPart=einsum('kt,kw,wt,wt->t',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
    eta=topPart/bottomPart
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
    properties=None
    emType=None
    # def __init__(self):

class discreteEmission(Emission):
    emMat=None
    numOutputsPerFeature=None

    
    def __init__(self,numStates,emMat=None,numOutputsPerFeature=None):
                                  
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
    
    def generateEmission(self,stateSeq):
        return numpy.array([numpy.random.choice(self.numOutputsPerFeature,p=self.emMat[cState]) for cState in stateSeq])

    def probObs(self,Obs):
        return array([self.emMat[:,cObs] for cObs in Obs]).T
    
    def probStateObs(self,cState,cObs):
        return self.emMat[cState,cObs]
        
    def fitTopEmission(self,obs,gamma):
        Tsteps=obs[0].shape[0]
        Indicator=zeros((self.numOutputFeatures,self.numOutputsPerFeature,T))
        for cFeat in range(self.numOutputFeatures):
            Indicator[[cFeat]*T,obs[cFeat,:],range(T)]=1
        topPart=einsum('jkt,it->ijk',Indicator,gamma)
        return topPart
        
        
            
class myHMM():
    T=None
    numStates=None
    numOutputFeatures=0
    emission=[]
    pi0=None
    def __init__(self, T=None,numStates=None,pi0=None):
        if isinstance(numStates,int) and (T is None):
            tmpMat=random.rand(numStates,numStates)
            T=(tmpMat/numpy.sum(tmpMat,axis=0)).T 
        else:
            raise MyValidationError("Must provide Transition Matrix or number of states")
        self.addT(T)
        self.addPi0(pi0)
                          
    def addT(self,T):
        if isinstance(T,(numpy.ndarray, numpy.generic)) and T.shape[0]==T.shape[1]:
            self.T=T 
            self.numStates=self.T.shape[0]
        else:
            raise MyValidationError("Unexpected Transition Matrix")
                   
    def addEmission(self, emType=None,**kargs):
        if self.numStates is None:
            raise MyValidationError("Must first define transition matrix or number of states")
            
        if emType=='discrete':
            self.emission.append(discreteEmission(self.numStates,**kargs))
        else:
            raise MyValidationError("Must provide valid emission type")
        self.numOutputFeatures=self.numOutputFeatures+1

    def addPi0(self, pi0=None,useSteadyState=True):
        if pi0 is None:
            # if useSteadyState:
                
            #     else
            tmpMat=random.rand(self.numStates)
            pi0=tmpMat/sum(tmpMat)
            pi0[-1]=1-sum(pi0[:-1])
            pi0=pi0/pi0.sum()

            self.pi0=pi0
        elif isinstance(pi0,(numpy.ndarray, numpy.generic)) and self.T.shape[0]==self.numStates:
            pi0=pi0/pi0.sum()
            self.pi0=pi0
        else:
            raise MyValidationError("Something wrong with pi0 input")
            
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

    def fitTopEmission(self,obs,gamma):
        Tsteps=obs[0].shape[0]
        Indicator=zeros((self.numOutputFeatures,self.numOutputsPerFeature,T))
        for cFeat in range(self.numOutputFeatures):
            Indicator[[cFeat]*T,obs[cFeat,:],range(T)]=1
        topPart=einsum('jkt,it->ijk',Indicator,gamma)
        return topPart

    
    def train(self,Ys):
        
        
        gammas=[]
        etas=[]
        pi0_topPart=zeros((self.numStates))
        T_topPart=zeros((self.numStates,self.numStates))
        T_bottomPart=zeros((self.numStates))
        b_bottomPart=zeros((self.numStates))
        # b_topPart=zeros((self.emission.emMat.shape))
        for obs in Ys:
            probObsState=array([cEmission.probObs(cObs) for cEmission,cObs in zip(self.emission,obs)]).prod(axis=0)

            alpha=calc_alpha(obs,self.T,self.pi0,probObsState)
            beta=calc_beta(obs,self.T,probObsState)
            gamma=calc_gamma(alpha, beta)
            eta=calc_eta(obs, self.T, alpha, beta, probObsState)
            
        #     # gammas.append(calc_gamma(alpha, beta))
        #     # etas.append(calc_eta(obs, self, alpha, beta))
            pi0_topPart=pi0_topPart+gamma[:,0]
            T_topPart=T_topPart+eta.sum(axis=2)
            T_bottomPart=T_bottomPart+gamma[:,0:-1].sum(axis=1)
            b_bottomPart=b_bottomPart+gamma.sum(axis=1)
            b_topPart=b_topPart+self.emission.fitTopEmission(obs,gamma)

        # R=len(Ys)
        # pi0=pi0_topPart/R
        # T=T_topPart/T_bottomPart
        # b=einsum('ijk,ij->ijk',b_topPart,1/b_bottomPart)
        
            a=5
            
        

        
                   
                 


def createRandomEmission(numStates,numOutputFeatures,NumOutputsPerFeature):
    if not isinstance(NumOutputsPerFeature,list):
        NumOutputsPerFeature=numStates*[NumOutputsPerFeature]
        
    for cState in range(numStates):
        random.randint(0,100,(numOutputFeatures,NumOutputsPerFeature))
        


numStates=3
NumOutputsPerFeature=10




mod=myHMM(numStates=3)
mod.addEmission(emMat=None,emType='discrete',numOutputsPerFeature=NumOutputsPerFeature)

# obs=[[3],[4],[2],[3],[4],[2]]
# alpha=calc_alpha(obs, mod)
# beta=calc_beta(obs, mod)
# delta,psi,path=viterbi(obs, mod)

X,Y=mod.genSequences(NumSequences=5,maxLength=20)


mod.train(Y)

# mod.addT()

# A=array([[.9,.05,.05],[.1,.85,.05],[0, .25, .75]])
# pi_o=array([.25,.45,.3])

# createRandomEmission(numStates,numOutputFeatures,NumOutputsPerFeature)

# emmissionDist=random.random_intergers(0,maxNumOutputsPerFeature,[numStates,numOutputFeature,maxNumOutputsPerFeature])