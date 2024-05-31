import numpy as np
import cv2
import random
from sklearn.preprocessing import minmax_scale
from pandas import DataFrame as df

# Define the fitness function
def _compute_otsu_criteria(img: np.ndarray, threshold: int) -> float:
    # Compute the Otsu criteria for a given threshold
    foreground = img > threshold
    background = img <= threshold

    weight_foreground = np.sum(foreground) / img.size
    weight_background = np.sum(background) / img.size

    mean_foreground = np.mean(img[foreground]) if weight_foreground > 0 else 0
    mean_background = np.mean(img[background]) if weight_background > 0 else 0

    return weight_foreground * weight_background * (mean_foreground - mean_background) ** 2

# Genetic Algorithm for finding the best threshold
def genetic_algorithm_otsu(img: np.ndarray, pop_size=12, generations=30, mutation_rate=0.02):
    # Initialize population with valid threshold values
    population = np.random.randint(0, 256, pop_size)

    for gen in range(generations):
        # Evaluate fitness for each individual in the population
        fitness = np.array([_compute_otsu_criteria(img, individual) for individual in population])

        # Handle the case where all fitness values are zero
        if np.sum(fitness) == 0:
            fitness += 1e-10

        # Selection (Roulette wheel selection)
        scaled_fitness = minmax_scale(fitness)

        # Ensure scaled_fitness does not result in NaN or Inf
        if np.sum(scaled_fitness) == 0:
            scaled_fitness += 1e-10

        selection_probs = scaled_fitness / np.sum(scaled_fitness)

        # Ensure probabilities are valid
        selection_probs = np.nan_to_num(selection_probs, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(selection_probs) == 0:
            selection_probs = np.ones_like(selection_probs) / len(selection_probs)

        selected_indices = np.random.choice(population, size=pop_size, p=selection_probs)

        # Crossover (Single point crossover)
        new_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_indices[i], selected_indices[i + 1]
            crossover_point = np.random.randint(1, 8)
            mask = (1 << crossover_point) - 1
            child1 = (parent1 & mask) | (parent2 & ~mask)
            child2 = (parent2 & mask) | (parent1 & ~mask)
            new_population.extend([child1, child2])

        # Mutation
        for i in range(pop_size):
            if random.random() < mutation_rate:
                mutation_point = np.random.randint(8)
                new_population[i] ^= 1 << mutation_point

        population = np.array(new_population)

    # Get the best threshold
    best_threshold = population[np.argmin([_compute_otsu_criteria(img, individual) for individual in population])]
    return best_threshold, population

# Otsu Thresholding with Genetic Algorithm
def otsu_thresholding_ga(img: np.ndarray) -> np.ndarray:
    best_threshold,population = genetic_algorithm_otsu(img)
    print(f"Best threshold found by GA: {best_threshold}")

    # binary, population = np.where(img > best_threshold, 255, 0).astype(np.uint8)
    return best_threshold, population

# Load image
image = cv2.imread('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/Praise THE SUN BG.jpg', 0)

# # Apply Otsu's thresholding using Genetic Algorithm
# binary_image, show_threshold, population = otsu_thresholding_ga(image)

# # Display the result
# import matplotlib.pyplot as plt

# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(binary_image, cmap='gray')
# plt.title(f'Best GA threshold = {show_threshold}')
# plt.axis('off')

# plt.savefig('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/Sun_Rani.png', bbox_inches='tight')
# plt.show()

data = []
last_pop = []
row = []
for i in range(30):
    best_th, last_pop = otsu_thresholding_ga(image)
    print(f'Roll - {i}')
    print(last_pop)
    print("Best threshold:",best_th)
    # thresholded_im = threshold_image(im,best_th)
    row = np.hstack([best_th,last_pop.T])
    print(row)
    
    data = np.vstack((data, row.T)) if (i > 0) else row.T
    print(data)
    
data = df(data)
topcer = np.count_nonzero((data == 160))
accuracy = topcer/30
print(f'Count = {topcer} || Acc = {accuracy}')
data.to_csv('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/Stresstest_Holin.csv')