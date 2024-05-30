import numpy as np
import cv2
import random

def threshold_image(im,th):
    threshold_im = np.zeros(im.shape)
    threshold_im[im>th] = 1
    return threshold_im

def compute_otsu_criteria(im:np.ndarray,th):
    thresholded_im = threshold_image(im,th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1/nb_pixels
    weight0 = 1-weight1
    if weight1 == 0 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_im==1]
    val_pixels0 = im[thresholded_im==0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0*var0 + weight1*var1

# # if using iteration then find the best threshold using minimizing within class variance
# def find_best_threshold(im:np.ndarray):
#     threshold_range = range(np.max(im)+1)
#     criterias = [compute_otsu_criteria(im,th) for th in threshold_range]
#     best_th = threshold_range[np.argmin(criterias)]
#     return best_th

# if using GA then find the best threshold using maximizing within class variance
def find_threshold_ga(im:np.ndarray,population_size=12,generations=30,mutation_rate=0.2):

    # initialize random population
    population = [random.randint(0, 255) for _ in range(population_size)]
    # iterate over generations
    for gen in range(generations):
        # sort population based on fitness value
        fitness = [compute_otsu_criteria(im,t) for t in population]
        population = sorted(population, key=lambda t: fitness[population.index(t)])
        # select the best half of the population
        next_population = population[:population_size//2]
        # generate offspring
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(next_population[:population_size//2], 2)
            crossover_point = random.randint(0, 255)
            child = (parent1 & crossover_point) | (parent2 & ~crossover_point) # crossover using bitwise operation
            if random.random() < mutation_rate:
                child = random.randint(0, 255)
            next_population.append(child)
        population = next_population
        # for visualization purpose
        print("\n##################")
        print("##Generation %2d" %(gen+1),"##")
        print("##################")
        zipped = zip(population,fitness)
        print("pop (threshold val),\t fitness value")
        for z in zipped:
            print("\t",z[0],",\t\t",z[1])
        #
    # return the best threshold, 0 because the population is sorted in ascending order
    # and the best threshold is the first element because it has the lowest within class variance
    best_threshold = population[0]
    return best_threshold


path = "D:\SKRIPSI related\PY\gogogo\otsu_test.jpeg"
im = cv2.imread(path,0)
cv2.imshow("original",im)
cv2.waitKey(0)

# best_th = find_best_threshold(im)
# print("Best threshold:",best_th)
# thresholded_im = threshold_image(im,best_th)
# cv2.imshow("thresholded",thresholded_im*255)
# cv2.waitKey(0)

best_th = find_threshold_ga(im)
print("Best threshold:",best_th)
thresholded_im = threshold_image(im,best_th)
cv2.imshow("thresholded",thresholded_im*255)
cv2.waitKey(0)
