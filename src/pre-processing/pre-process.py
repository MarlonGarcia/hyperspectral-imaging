# -*- coding: utf-8 -*-
"""
First Steps with Hyperspectral Images

This code is dedicated to the first steps of processing a hyperspectral image.

"""

# Importing the important libraries
import numpy as np
import os
import cv2
import imfun
import matplotlib.pyplot as plt

class hypercube:
    
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
            with a height of H and a width of W.
        '''
        # First, listing all the names of files in the image directory
        image_names = os.listdir(self.dir)
        # Changing to the images directory
        os.chdir(self.dir)
        # Creating the variable to contain the images
        I = []
        # Adding all images from the 'self.dir' directory to 'I'
        for name in image_names:
            I.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))
        # Returning the hyperspectral cube 'I'
        return np.asarray(I)
    
    def get_wavelengths(self, init=400, end=720, num=33):
        '''
        Getting the wavelengths of the hyperspectral image

        Parameters
        ----------
        init : int, optional
            The initial value of the wavelengths. The default is 400 (nm).
        end : int, optional
            The final value of the wavelengths. The default is 720 (nm).
        num : int, optional
            Number of wavelengths in the hypercube. The default is 33.

        Returns
        -------
        'numpy.ndarray'
            Vector with all wavelengths of the hyperspectral image.
        '''
        return np.linspace(init, end, num)
    
    def normalize(self, I, Irgb):
        '''
        Normalizing image by a 'white' region.

        Parameters
        ----------
        I : 'numpy.ndarray'
            Hyperspectral cube in (C, H, W) format, with C channels (images),
            with a height of H and a width of W.
        Irgb : 'numpy.ndarray'
            RGB image from the hypercube.

        Returns
        -------
        I : 'numpy.ndarray'
            Normalized hyperspectral cube, in (C, H, W) format.
        '''
        # Creating a null 'white' vector to add the ref. values to normalize.
        white = np.zeros(len(I))
        window_name = 'Choose the white region to normalize the image'
        [Imask, points] = imfun.polyroi(Irgb, window_name = window_name)
        Imask = Imask[:,:,0]
        
        for n in range(len(I)):
            temp = np.mean(I[n][Imask>0])
            white[n] = temp.astype('uint8')
            Itemp = I[n].astype('float')*(255/white[n])
            I[n] = Itemp.astype('uint8')
        
        return I
    
    
    def get_rgb(self, I, w = [5, 15, 23]):
        '''
        Getting an RGB image from the hypercube

        Parameters
        ----------
        I : 'numpy.ndarray'
            Hyperspectral cube in (C, H, W) format, with C channels (images),
            with a height of H and a width of W.
        w : 'list', optional
            A list with the position of the wavelengths to be considered as
            red, green, and blue, respectively. The default is [5, 15, 23].

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
            with a height H and a width W.
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
        Choose points in the image to print their spectrum

        Parameters
        ----------
        I : 'numpy.ndarray'
            Hyperspectral cube in (C, H, W) format, with C channels (images),
            with a height of H and a width of W.

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



#%% Testing Code

if __name__ == "__main__":
    # Defining directory. Use 'r' before the string
    im_dir = r'G:\Drives compartilhados\Imageamento Hiperespectral\Imagens\Imagens Multiespectrais\2020.06.08 - R3 PPIX PS - Fígado camundongo'
    
    # Closing all the images
    plt.close('all')
    
    # Defining a function to print images
    def print_image(image):
        plt.subplots()
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
    
    # Instanciating the hypercube class
    cube = hypercube(im_dir)
    
    # Loading the hypercube
    I = cube.load()
    
    # Getting the wavelengths
    w = cube.get_wavelengths()
    
    # Getting the RGB from the hypercube image
    Irgb = cube.get_rgb(I)
    print_image(Irgb)
    
    # Getting the histogram for each image channel
    hist = cube.hist(I)
    # Printing histogram to each channel
    plt.subplots()
    for n in range(len(hist)):
        plt.plot(hist[n, :], label=str(w[n])+' nm')
    plt.legend()
    plt.show()




#%%

I = np.zeros([150, 150], dtype='uint8')
Irgb = np.zeros([150, 150, 3], dtype='uint8')

top_left = (50, 50)
bottom_right = (100, 100)

cv2.rectangle(I, top_left, bottom_right, 201, -1)

I = np.array([I,I,I,I,I])

np.shape(I)

cv2.rectangle(Irgb, top_left, bottom_right, (0, 201, 0), -1)

plt.subplots()
plt.imshow(Irgb)


I = cube.normalize(I, Irgb)

plt.subplots()
plt.imshow(I[0,:,:])