import numpy as np
from numpy import array, zeros
import algorithms
import croupier
import dice
import pandas

np.set_printoptions(precision=5)

OBSERVATION_LENGTH = 20
OBSERVATIONS = 5000

# Prepare environment for our simulation
fair_dice = dice.Dice(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
loaded_dice = dice.Dice(np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/2]))
# fair_dice = dice.Dice(np.array([1/5, 1/5, 1/5, 1/5, 1/5, 0]))
# loaded_dice = dice.Dice(np.array([0, 0, 0, 0, 0, 1]))
initial_dice_probability = np.array([0.75, 0.25])  # Croupier can use either of dices initially
transition_matrix = np.array([
    [0.75, 0.25],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
    [0.10, 0.90],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
])
my_croupier = croupier.Croupier(fair_dice, loaded_dice, initial_dice_probability, transition_matrix)
observation_states = [my_croupier.get_observations(OBSERVATION_LENGTH)  for _ in range(OBSERVATIONS)]
allObservations=[x[0] for x in observation_states]
allTrueStates=[x[1] for x in observation_states]

# observation_df=pandas.DataFrame(allObservations)
# observation_df.to_csv('observations.csv',index=False)

pi0_Numerical=sum([x[0] for x in allTrueStates])/len(allTrueStates)
T_Numerical=zeros((2,2))
for cStateChain in allTrueStates:
    for i in range(len(cStateChain)-1):
        T_Numerical[cStateChain[i],cStateChain[i+1]] += 1
        

T_Numerical=(T_Numerical.T/T_Numerical.sum(axis=1)).T   
print(T_Numerical)
# [[0.74925 0.25075]
#  [0.09781 0.90219]]
print([1-pi0_Numerical, pi0_Numerical])
# [0.285,0.715]