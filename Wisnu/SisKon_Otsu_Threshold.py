import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from GA import GeneticAlgorithm
import Otsu_Threshold


def apply_threshold(t, image):
    image_tmp = np.asarray(image)  # Convert the image to a numpy array
    background = Image.fromarray(np.where(image_tmp < t, image_tmp, 255).astype(np.uint8))  # Create the background image
    foreground = Image.fromarray(np.where(image_tmp >= t, image_tmp, 0).astype(np.uint8))  # Create the foreground image
    return background, foreground  # Return the background and foreground images


def main():
    thresholds, fitnesss = [], []  # Initialize lists to store thresholds and fitness values
    im = Image.open(r'C:\Users\wisnu\OneDrive\Pictures\PCB_Test.jpeg')  # Open the image file
    im.load()  # Load the image
    im.show()  # Display the image
    im_gray = im.convert('L')  # Convert the image to grayscale
    im_gray.show()  # Display the grayscale image
    im_gray.save('PCB_Test.jpeg')  # Save the grayscale image
    g = GeneticAlgorithm(im_gray, 9, 101)  # Initialize the genetic algorithm with the grayscale image, population size 8, and 100 iterations
    best_threshold = Otsu_Threshold.get_best_threshold(np.array(im_gray))  # Get the best threshold using Otsu's method
    print(f"Best Threshold from Otsu's Method: {best_threshold}")  # Print the best threshold from Otsu's method
    best_threshold, thresholds, fitnesss, cur_iteration = g.get_threshold()  # Get the best threshold, thresholds, fitness values, and current iteration using the genetic algorithm
    Otsu_Threshold.histogramify(im_gray)  # Generate and display the histogram of the grayscale image
    
    background, foreground = apply_threshold(best_threshold, im_gray)  # Apply the best threshold to the image to get the background and foreground images
    
    # Display the original, background, and foreground images
    plt.figure(figsize=(12, 6))  # Create a figure with a specified size
    
    plt.subplot(1, 3, 1)  # Create the first subplot
    plt.title('Original Image')  # Set the title of the first subplot
    plt.imshow(im_gray, cmap='gray')  # Display the original grayscale image
    plt.axis('off')  # Hide the axis
    
    plt.subplot(1, 3, 2)  # Create the second subplot
    plt.title('Background')  # Set the title of the second subplot
    plt.imshow(background, cmap='gray')  # Display the background image
    plt.axis('off')  # Hide the axis
    
    plt.subplot(1, 3, 3)  # Create the third subplot
    plt.title('Foreground')  # Set the title of the third subplot
    plt.imshow(foreground, cmap='gray')  # Display the foreground image
    plt.axis('off')  # Hide the axis
    
    plt.show()  # Show the plot
    
    # Plot the thresholds over iterations
    plt.figure()  # Create a new figure
    plt.title("Threshold (N={} iteration={})".format(g.N, g.max_interation), fontsize=12)  # Set the title of the plot
    plt.plot(thresholds, linewidth=1)  # Plot the thresholds
    plt.show()  # Show the plot
    
    # Plot the fitness values over iterations
    plt.figure()  # Create a new figure
    plt.title("Fitness (N={} iteration={})".format(g.N, g.max_interation), fontsize=12)  # Set the title of the plot
    plt.plot(fitnesss, linewidth=1)  # Plot the fitness values
    plt.show()  # Show the plot
    
    print(fitnesss)  # Print the fitness values

if __name__ == '__main__':
    main() # Call the main function
