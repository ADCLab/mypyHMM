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
        # alphaTransition=einsum('j,ji->i',alpha[:,t-1],T)
        # alpha[:,t]=einsum('i,i->i',alphaTransition,probObsState[:,t])
        alpha[:,t]=einsum('i,ij,j->j',alpha[:,t-1],T,probObsState[:,t])
    return alpha

def calc_beta(obs,T,probObsState):
    Tsteps=obs[0].shape[0]
    numStates=T.shape[0]
    beta=empty((numStates,Tsteps))
    beta[:,0]=numStates*[1]
    probObsState=fliplr(probObsState)
    for tb in range(1,Tsteps):
        beta[:,tb]=einsum('j,ij,j->i',beta[:,tb-1],T,probObsState[:,tb-1])
    return fliplr(beta)

def calc_gamma(alpha,beta):
    topPart=einsum('it,it->it',alpha,beta)
    gamma=einsum('it,t->it',topPart,1/topPart.sum(axis=0))
    return gamma

def calc_eta(obs,T,alpha,beta,probStateObs):
    topPart=einsum('it,ij,jt,jt->ijt',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
    bottomPart=einsum('kt,kw,wt,wt->t',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
    eta=einsum('ijt,t->ijt',topPart,1/bottomPart)
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
        Tsteps=obs.shape[0]
        Indicator=zeros((self.numOutputsPerFeature, Tsteps))
        Indicator[obs,range(Tsteps)]=1
        topPart=einsum('kt,it->ik',Indicator,gamma)
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
            T=(tmpMat/tmpMat.sum(axis=0)).T
        elif T is not None:
            self.addT(T)
            
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


    
    def train(self,Ys,iterations=20):
        for iter in range(iterations):
            pi0_topPart=zeros((self.numStates))
            T_topPart=zeros((self.numStates,self.numStates))
            T_bottomPart=zeros((self.numStates))
            b_bottomPart=zeros((self.numStates))
            b_topParts=[zeros((cEmission.emMat.shape)) for cEmission in self.emission]
            for obs in Ys:
                probObsState=array([cEmission.probObs(cObs) for cEmission,cObs in zip(self.emission,obs)]).prod(axis=0)
    
                alpha=calc_alpha(obs,self.T,self.pi0,probObsState)
                beta=calc_beta(obs,self.T,probObsState)
                gamma=calc_gamma(alpha, beta)
                eta=calc_eta(obs, self.T, alpha, beta, probObsState)
                pi0_topPart=pi0_topPart+gamma[:,0]
                T_topPart=T_topPart+eta.sum(axis=2)
                T_bottomPart=T_bottomPart+gamma[:,0:-1].sum(axis=1)
                
                
                b_bottomPart=b_bottomPart+gamma.sum(axis=1)
                b_topParts=[cTopPart+cEmission.fitTopEmission(feat_obs,gamma) for  feat_obs,cEmission, cTopPart in zip(obs,self.emission, b_topParts.copy())]
                # b_topPart=b_topPart+self.emission[0].fitTopEmission(obs[0],gamma)

            R=len(Ys)
            pi0=pi0_topPart/R
            T=einsum('ij,i->ij',T_topPart,1/T_bottomPart)
            # emMat=einsum('ik,i->ik',b_topPart,1/b_bottomPart)
            self.T=T
            self.pi0=pi0
            # self.emission[0].emMat=emMat
            print(self.pi0)

            
        

        
                   
                 


def createRandomEmission(numStates,numOutputFeatures,NumOutputsPerFeature):
    if not isinstance(NumOutputsPerFeature,list):
        NumOutputsPerFeature=numStates*[NumOutputsPerFeature]
        
    for cState in range(numStates):
        random.randint(0,100,(numOutputFeatures,NumOutputsPerFeature))
        


numStates=2
NumOutputsPerFeature=3


# T = array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
# B = array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
# pi = array([0.2,0.4,0.4])
# O = [[array([0,1,0,1])]]
# mod=myHMM(T=T,numStates=3,pi0=pi)
# mod.addEmission(emMat=B,emType='discrete')
# mod.train(O)
# print(mod.T)


# #transition probabilities
# T = array([[0.8,0.1],
#                        [0.1,0.8]])
# #Emission probabilities
# B = array([[0.1,0.2,0.7],
#                      [0.7,0.2,0.1]])
# #test sequence
# test_sequence = '331122313'
# O = [[array([3,3,1,1,2,2,3,1,3])-1]]
# pi = array([0.5,.5])

# #probabilities of going to end state
# end_probs = [0.1, 0.1]
# #probabilities of going from start state
# start_probs = [0.5, 0.5]
# mod=myHMM(T=T,numStates=3,pi0=pi)
# mod.addEmission(emMat=B,emType='discrete')
# mod.train(O)

mod=myHMM(numStates=3)
T=mod.T
pi0=mod.pi0
mod.addEmission(emType='discrete',numOutputsPerFeature=5)
# mod.addEmission(emType='discrete',numOutputsPerFeature=2)


X,Y=mod.genSequences(NumSequences=1000,maxLength=10)
mod.train(Ys=Y,iterations=1)
# print(T-mod.T)
# print(pi0-mod.pi0)



# mod.addT()

# A=array([[.9,.05,.05],[.1,.85,.05],[0, .25, .75]])
# pi_o=array([.25,.45,.3])

# createRandomEmission(numStates,numOutputFeatures,NumOutputsPerFeature)

# emmissionDist=random.random_intergers(0,maxNumOutputsPerFeature,[numStates,numOutputFeature,maxNumOutputsPerFeature])