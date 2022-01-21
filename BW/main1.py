import numpy as np
from numpy import array
import algorithms
import croupier
import dice
import pandas

np.set_printoptions(precision=5)

# OBSERVATION_LENGTH = 20
# OBSERVATIONS = 50

# # Prepare environment for our simulation
# # fair_dice = dice.Dice(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
# # loaded_dice = dice.Dice(np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/2]))
# fair_dice = dice.Dice(np.array([1/5, 1/5, 1/5, 1/5, 1/5, 0]))
# loaded_dice = dice.Dice(np.array([0, 0, 0, 0, 0, 1]))
# initial_dice_probability = np.array([0.5, 0.5])  # Croupier can use either of dices initially
# transition_matrix = np.array([
#     [0.75, 0.25],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
#     [0.20, 0.80],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
# ])
# my_croupier = croupier.Croupier(fair_dice, loaded_dice, initial_dice_probability, transition_matrix)


YsDF=pandas.read_csv('observations.csv')

Ys=[]
for cRow in YsDF.iterrows():
    Ys.append(array(cRow[1]))


emMat=array([[1/6,1/6,1/6,1/6,1/6,1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
YsT=[[y] for y in Ys]
pi0=array([0.75, 0.25])
T=array([[0.75, 0.25],
        [0.1, 0.9]])


first_dice, second_dice, initial_dice_probability, transition_matrix = algorithms.baum_welch(Ys[0:500],T=T,emMat=emMat,pi0=pi0,epochs=200)

