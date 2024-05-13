# -*- coding: utf-8 -*-
"""
First Steps with Hyperspectral Images

This code is dedicated to first steps processing a hyperspectarl image.

"""

# Importing the important libraries
import numpy as np
import os
import cv2
import imfun
import matplotlib.pyplot as plt

class hipercube:
    
    def __init__(self, image_dir):
        '''
        Initializing function

        Parameters
        ----------
        image_dir : 'string'
            Directory to the images.
        '''
        self.dir = image_dir
    
    def load(self):
        '''
        Loading hyperspectral image

        Returns
        -------
        I : 'numpy.ndarray'
            Hyperspectral image in (C, H, W) format, with C channels (images),
            with a hight H and a width W.
        '''
        # First, listing all the names of files in the image directory
        image_names = os.listdir(self.dir)
        # Changing to the images directory
        os.chdir(self.dir)
        # Creating the variable to contain the images
        I = []
        # Adding all images with name 'name' from 'self.dir' directory to 'I'
        for name in image_names:
            I.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))
        # Returning the hiperspectral cube 'I'
        return np.asarray(I)
    
    def get_rgb(self, I, w = [5, 15, 23]):
        '''
        Getting an RGB image from the hypercube

        Parameters
        ----------
        I : 'numpy.ndarray'
            Hyperspectral cube in (C, H, W) format, with C channels (images),
            with a hight H and a width W.
        w : 'list', optional
            A list with the position of the wavelengths to be considered as
            red, green and blue, respectivelly. The default is [5, 15, 23].

        Returns
        -------
        'numpy.ndarray'
            RGB image from the hypercube.
        '''
        # Getting a (C, H, W) format (full RGB) from a (H, W) image format
        # (each color or wavelength) using 'expand_dims'
        Ib = np.expand_dims(I[w[0]], axis = 2)    #450
        Ig = np.expand_dims(I[w[1]], axis = 2)    #550
        Ir = np.expand_dims(I[w[2]], axis = 2)    #630
        # Returning the full RGB image
        return np.concatenate((Ir, Ig, Ib), axis = 2)
    
    def hist(self, I, pd = 8):
        '''
        Getting the histogram for each channel

        Parameters
        ----------
        I : 'numpy.ndarray'
            Hyperspectral cube in (C, H, W) format, with C channels (images),
            with a hight H and a width W.
        pd : 'int', optional
            Camera's pixel depth. The default is 8 bits.

        Returns
        -------
        hist : 'numpy.ndarray'
            A (C, V) variable with the histogram for pixels' intensities in the
            V axis, for each channel C.
        '''
        # Defining a null variable
        hist = np.zeros([len(I), 2**pd])
        # Adding histograms to 'hist', channel by channel
        for n in range(len(I)):
            # Using 'np.linspace' to create the intensity vector of a histogram
            # and 'np.histogram' to calculate it for a given channel 'n'
            # OBS: The total number of points is 2**pd+1
            temp = np.histogram(I[n], bins = np.linspace(0, 2**pd, 2**pd+1))
            # from 'temp'  variable to 'hist' variable, in channel 'n'
            hist[n, :] = temp[0]
        # Returning full histogram
        return hist
    
    def points_spec(self, I):
        '''
        Choose points in image to print their spectrum

        Parameters
        ----------
        I : 'numpy.ndarray'
            Hyperspectral cube in (C, H, W) format, with C channels (images),
            with a hight H and a width W.

        Returns
        -------
        None.

        '''
        Irgb = self.get_rgb(I)
        _, points = imfun.choos_points(Irgb[:,:,::-1], show=True)
        
        points_spec = np.ones((len(Irgb[:,...]), len(points[:,0])))
        for n in range(0,len(points[:,0])):
            # 
            points_spec[:,n] = Irgb[:,points[n,1], points[n,0]]
        # 
        points_spec = np.asarray(points_spec)

        
        return 'True'
        
        
    





#%% Testing the Code


if __name__ == "__main__":
    # Defining directory. Use 'r' before the string
    im_dir = r'H:\Shared drives\Imageamento Hiperespectral\Imagens\Imagens Multiespectrais\2021.06.01 - Lâmina Histlógica\Image1'
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

