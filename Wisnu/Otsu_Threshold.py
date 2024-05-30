import numpy as np  # Import the numpy library for numerical operations
import matplotlib.pyplot as plt  # Import the matplotlib library for plotting
from PIL import Image  # Import the Python Imaging Library for image manipulation

def total_pix(image):
    size = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    return size  # Return the total number of pixels

def histogramify(image):
    grayscale_array = []  # Initialize an empty list to store grayscale intensity values
    for w in range(0, image.size[0]):  # Loop through the width of the image
        for h in range(0, image.size[1]):  # Loop through the height of the image
            intensity = image.getpixel((w, h))  # Get the pixel intensity at position (w, h)
            grayscale_array.append(intensity)  # Append the intensity to the list
    bins = range(0, 256)  # Define the range of bins for the histogram (0-255 for grayscale)
    grayscale_array = np.array(grayscale_array)  # Convert the list to a numpy array
    plt.hist(grayscale_array, bins=bins, density=False, facecolor="blue", edgecolor="blue", alpha=0.7)  # Plot the histogram
    plt.title("Grayscale")  # Set the title of the histogram
    plt.show()  # Display the histogram

def get_best_threshold(img_array):
    height = img_array.shape[0]  # Get the height of the image array
    width = img_array.shape[1]  # Get the width of the image array
    count_pixel = np.zeros(256)  # Initialize a numpy array to count pixel intensities

    for i in range(height):  # Loop through the height of the image
        for j in range(width):  # Loop through the width of the image
            count_pixel[int(img_array[i][j])] += 1  # Increment the count for the pixel intensity

    max_variance = 0.0  # Initialize the maximum variance
    best_threshold = 0  # Initialize the best threshold
    for threshold in range(256):  # Loop through all possible thresholds (0-255)
        n0 = count_pixel[:threshold].sum()  # Sum of pixel counts for the foreground
        n1 = count_pixel[threshold:].sum()  # Sum of pixel counts for the background
        w0 = n0 / (height * width)  # Weight of the foreground
        w1 = n1 / (height * width)  # Weight of the background
        u0 = 0.0  # Mean intensity of the foreground
        u1 = 0.0  # Mean intensity of the background

        for i in range(threshold):  # Loop through the foreground pixel intensities
            u0 += i * count_pixel[i]
        for j in range(threshold, 256):  # Loop through the background pixel intensities
            u1 += j * count_pixel[j]

        u = u0 * w0 + u1 * w1  # Overall mean intensity
        tmp_var = w0 * np.power((u - u0), 2) + w1 * np.power((u - u1), 2)  # Between-class variance

        if tmp_var > max_variance:  # If the current variance is greater than the max variance
            best_threshold = threshold  # Update the best threshold
            max_variance = tmp_var  # Update the maximum variance
    return best_threshold  # Return the best threshold

def my_otsu(image, threshold):
    image = np.transpose(np.asarray(image))  # Convert the image to a numpy array and transpose it
    total = total_pix(image)  # Get the total number of pixels in the image
    bin_image = image < threshold  # Binarize the image using the threshold
    sumT = np.sum(image)  # Sum of all pixel intensities in the image
    w0 = np.sum(bin_image)  # Sum of binarized image pixels
    sum0 = np.sum(bin_image * image)  # Sum of the foreground pixels
    w1 = total - w0  # Sum of the background pixels
    if w1 == 0:  # If there are no background pixels
        return 0  # Return 0 to avoid division by zero
    sum1 = sumT - sum0  # Sum of the background pixels
    mean0 = sum0 / (w0 * 1.0)  # Mean intensity of the foreground
    mean1 = sum1 / (w1 * 1.0)  # Mean intensity of the background
    varBetween = w0 / (total * 1.0) * w1 / (total * 1.0) * (mean0 - mean1) * (mean0 - mean1)  # Calculate the between-class variance
    return varBetween  # Return the between-class variance
