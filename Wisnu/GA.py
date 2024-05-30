import numpy as np  # Import the numpy library for numerical operations
import random  # Import the random library for random number generation
from Otsu_Threshold import my_otsu  # Import the my_otsu function from Otsu_Threshold module
from PIL import Image  # Import the Python Imaging Library for image manipulation

class GeneticAlgorithm:
    def __init__(self, image, N, max_interation):
        self.image = np.array(image)  # Convert the image to a numpy array
        self.N = N  # Number of chromosomes in the population
        self.population = self.init_chrome(self.N)  # Initialize the population
        self.max_interation = max_interation  # Maximum number of iterations

    def bin_to_oct(self, chrom):
        result = 0  # Initialize the result to 0
        for i in range(len(chrom) - 1, -1, -1):  # Loop through the chromosome in reverse order
            if chrom[i] == 1:  # If the current bit is 1
                result += pow(2, len(chrom) - 1 - i)  # Add the corresponding power of 2 to the result
        return result  # Return the result as the decimal equivalent of the binary number

    def init_chrome(self, N):
        population = []  # Initialize an empty list to store the population
        for i in range(0, N):  # Loop N times to create N chromosomes
            population.append(np.random.randint(0, 2, 8).tolist())  # Create a random chromosome of 8 bits
        return population  # Return the initialized population

    def get_fitness(self):
        test_nums = []  # Initialize an empty list to store the decimal equivalents of the chromosomes
        for pop in self.population:  # Loop through each chromosome in the population
            test_nums.append(self.bin_to_oct(pop))  # Convert the chromosome to its decimal equivalent
        fitness = [my_otsu(self.image, i) for i in test_nums]  # Calculate the fitness for each chromosome using my_otsu

        return fitness  # Return the list of fitness values

    def select(self):
        new_population = []  # Initialize an empty list to store the new population
        fitness = self.get_fitness()  # Get the fitness values of the current population
        sum_fitness = np.sum(fitness)  # Calculate the sum of the fitness values
        probability = fitness / sum_fitness  # Calculate the selection probability for each chromosome
        accu_probability = [0]  # Initialize the accumulated probability list
        for i in range(0, len(probability), 1):  # Loop through the selection probabilities
            accu_probability.append(sum(probability[0:i+1]))  # Calculate the accumulated probabilities
        random_num = np.random.random(self.N)  # Generate N random numbers between 0 and 1
        for num in random_num:  # Loop through each random number
            for i in range(0, len(accu_probability), 1):  # Loop through the accumulated probabilities
                if accu_probability[i] <= num <= accu_probability[i + 1]:  # Check if the random number falls within the current range
                    new_population.append(self.population[i])  # Select the corresponding chromosome
                else:
                    pass

        if len(new_population) < self.N:  # If the new population is smaller than N
            for i in range(0, self.N - len(new_population)):  # Loop to fill the remaining slots
                new_population.append(np.random.randint(0, 2, 8).tolist())  # Add random chromosomes
            self.population = new_population[:]  # Update the population with the new population
        elif len(new_population) > self.N:  # If the new population is larger than N
            self.population = new_population[:self.N]  # Trim the population to size N

    def cross(self):
        num1, num2 = random.randint(0, self.N - 2), random.randint(0, self.N - 2)  # Select two random chromosomes
        cross_bits_num = 4  # Define the crossover point
        self.population[num1][cross_bits_num:], self.population[num2][cross_bits_num:] = \
            self.population[num2][cross_bits_num:], self.population[num1][cross_bits_num:]  # Perform crossover

    def mutate(self):
        mutate_num = 0.04 * self.N * 8  # Calculate the number of mutations (4% of the total bits in the population)
        mutate_num = int(mutate_num)  # Convert to an integer
        for i in range(mutate_num):  # Loop through the number of mutations
            x = random.randint(0, 7)  # Select a random chromosome
            y = random.randint(0, 7)  # Select a random bit in the chromosome
            pre_fitness = my_otsu(self.image, self.bin_to_oct(self.population[x]))  # Calculate the fitness of the chromosome
            if self.population[x][y] == 1:  # If the selected bit is 1
                self.population[x][y] = 0  # Flip it to 0
            else:
                self.population[x][y] = 1  # Flip it to 1

    def get_threshold(self):
        best_thresholds, best_fitnesss = [], []  # Initialize lists to store the best thresholds and fitness values
        inter_count = 0  # Initialize the iteration counter
        fitness = self.get_fitness()  # Get the fitness values of the initial population
        best_fitness = np.max(fitness)  # Find the maximum fitness value
        print(best_fitness)  # Print the best fitness value
        fiteness_max, sustain_num, cata_count = 0, 0, 0  # Initialize variables for tracking fitness and iteration count
        best_threshold = self.bin_to_oct(self.population[np.argmax(fitness)])  # Get the threshold corresponding to the best fitness value
        while True:  # Loop until the stopping condition is met
            self.select()  # Select the next generation of chromosomes
            self.cross()  # Perform crossover
            self.mutate()  # Perform mutation
            fitness = self.get_fitness()  # Get the fitness values of the new population
            if best_fitness < np.max(fitness):  # If the best fitness has improved
                best_fitness = np.max(fitness)  # Update the best fitness
                best_threshold = self.bin_to_oct(self.population[np.argmax(fitness)])  # Update the best threshold
                inter_count += 1  # Increment the iteration counter
                sustain_num = 0  # Reset the sustain count
            else:
                sustain_num += 1  # Increment the sustain count
                inter_count += 1  # Increment the iteration counter
                if sustain_num >= 5:  # If the best fitness has not improved for 5 iterations
                    fitness_max = max(fitness)  # Find the maximum fitness value
                    for i in range(0, len(self.population)):  # Loop through the population
                        if my_otsu(self.image, self.bin_to_oct(self.population[i])) == fiteness_max:  # If the fitness matches the maximum fitness
                            self.population[i] = self.init_chrome(1)  # Reinitialize the chromosome
                            cata_count += 1  # Increment the catastrophe count
                            if cata_count == 5:  # If 5 catastrophes have occurred
                                sustain_num, cata_count = 0, 0  # Reset the sustain and catastrophe counts
                        else:
                            pass

            if inter_count > self.max_interation:  # If the maximum number of iterations is reached
                break  # Exit the loop
            print("Max Fitness： {}".format(best_threshold))  # Print the best threshold
            best_thresholds.append(best_threshold)  # Append the best threshold to the list
            print("Best Fitness： {}".format(best_fitness))  # Print the best fitness value
            best_fitnesss.append(best_fitness)  # Append the best fitness value to the list
            print()
        return best_threshold, best_thresholds, best_fitnesss, inter_count  # Return the results
