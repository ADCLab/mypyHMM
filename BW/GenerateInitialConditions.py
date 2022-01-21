import numpy as np

first_dice_probabilities = np.random.rand(6)
first_dice_probabilities /= first_dice_probabilities.sum()  # Make probabilities sum up to 1.0
# first_dice = dice.Dice(first_dice_probabilities)

# Initialize second dice parameters
second_dice_probabilities = np.random.rand(6)
second_dice_probabilities /= second_dice_probabilities.sum()  # Make probabilities sum up to 1.0
# second_dice = dice.Dice(second_dice_probabilities)

# Initialize probabilities which were used to randomize first dice
initial_dice_probability = np.random.rand(2)
initial_dice_probability /= initial_dice_probability.sum()  # Make probabilities sum up to 1.0

# Initialize transition probabilities
transition_matrix = np.random.rand(2, 2)
transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]  # Make probabilities sum up to 1.0 for each dice