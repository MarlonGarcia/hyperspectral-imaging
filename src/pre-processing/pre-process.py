# -*- coding: utf-8 -*-
"""
First Steps with Hyperspectral Images

This code is dedicated to first steps processing a hyperspectarl image.

"""

# Importing the important libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class hipercube:
    
    def __init__(self, image_dir):
        self.dir = image_dir
    
    def load(self):
        image_names = os.listdir(self.dir)
        os.chdir(self.dir)
        I = []
        for name in image_names:
            I.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))
        return np.asarray(I)
    
    def get_rgb(self, I, w = [5, 15, 23]):
        # Considering RGB not BGR (opencv)
        Ib = np.expand_dims(I[w[0]], axis = 2)    #450
        Ig = np.expand_dims(I[w[1]], axis = 2)    #550
        Ir = np.expand_dims(I[w[2]], axis = 2)    #630
        return np.concatenate((Ir, Ig, Ib), axis = 2)
    
    def hist(self, I, pd = 256):
        # This code consider a pixel detph of 8 bits
        # Preparing the variable
        hist = np.zeros([len(I), pd])
        for n in range(len(I)):
            temp = np.histogram(I[n], bins = np.linspace(0, 256, 257))
            hist[n, :] = temp[0]
        return hist
    





#%% Testing the Code


if __name__ == "__main__":
    # Defining directory. Use 'r' before the string
    im_dir = r'C:\Users\marlo\My Drive\College\Biophotonics Lab\Research\Data\Imageamento Multiespectral\16) 15.12.20 - Teste 01 Café bom'
    # Defining a function to plot images
    
    def press(image):
        plt.subplots()
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
    
    # Instanciating the hipercube class
    cube = hipercube(im_dir)
    I = cube.load()
    Irgb = cube.get_rgb(I)
    press(Irgb)
    
    hist = cube.hist(I)
    plt.subplots()
    for n in range(len(hist)):
        plt.plot(hist[n, :])
    plt.show()

