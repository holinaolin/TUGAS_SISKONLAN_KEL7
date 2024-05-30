import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from GA import GeneticAlgorithm
import Otsu_Threshold


def apply_threshold(t, image):
    image_tmp = np.asarray(image)
    background = Image.fromarray(np.where(image_tmp < t, image_tmp, 255).astype(np.uint8))
    foreground = Image.fromarray(np.where(image_tmp >= t, image_tmp, 0).astype(np.uint8))
    return background, foreground


def main():
    thresholds, fitnesss = [], []
    im = Image.open(r'C:\Users\wisnu\OneDrive\Pictures\PCB_Test.jpeg')
    im.load()
    im.show()
    im_gray = im.convert('L')
    im_gray.show()
    im_gray.save('PCB_Test.jpeg')
    g = GeneticAlgorithm(im_gray, 8, 100)
    best_threshold = Otsu_Threshold.get_best_threshold(np.array(im_gray))
    print(f"Best Threshold from Otsu's Method: {best_threshold}")
    best_threshold, thresholds, fitnesss, cur_iteration = g.get_threshold()
    Otsu_Threshold.histogramify(im_gray)
    
    background, foreground = apply_threshold(best_threshold, im_gray)
    
    # Display the images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(im_gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Background')
    plt.imshow(background, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Foreground')
    plt.imshow(foreground, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    # Plot the thresholds and fitness values
    plt.figure()
    plt.title("Threshold (N={} iteration={})".format(g.N, g.max_interation), fontsize=12)
    plt.plot(thresholds, linewidth=1)
    plt.show()
    
    plt.figure()
    plt.title("Fitness (N={} iteration={})".format(g.N, g.max_interation), fontsize=12)
    plt.plot(fitnesss, linewidth=1)
    plt.show()
    
    print(fitnesss)


if __name__ == '__main__':
    main()
