# for obs in Ys:
#     log_probObsState=log(array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)])).sum(axis=0)    
import time      
from numpy import log

def add(b):
    return log(5+b)

class test(object):
    def __init__(self):
        self.a=5
        
    def addition(self,b):
        return log(self.a + b)

    def mapadd(self,mylist):
        # res=list(map(self.addition,mylist))
        res=[self.addition(x) for x in mylist]
        return res


mylist=range(1000)
a=test()
sTime=time.time()
for i in range(1000):
    # for val in mylist:
    #     add(val)
    res0=a.mapadd(mylist)
eTime=time.time()
print(eTime-sTime)

# res1=map(addition,mylist)


# def addition(n):
#     return n + n
  
# # We double all numbers using map()
# numbers = (1, 2, 3, 4)
# result = map(addition, numbers)
# print(list(result))

