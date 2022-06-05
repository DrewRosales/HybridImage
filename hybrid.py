import cv2 as cv
import sys
import numpy as np
import os


#referenced Cornell's CV course (CS5670)
def cross_correlation_2d(img, kernel):
    '''
    INPUT:
        img:    RBG image (height x width x 3) or grayscale image (height x width) as a numpy array
        kernel: numpy array to specify the transformation
    
    OUTPUT:
        Returns an image with the same initial dimensions as the input
    '''

    u,v = kernel.shape
    cross_img = np.zeros(img.shape)

    u_pad = (u-1)/2
    v_pad = (v-1)/2

    #RGB
    if len(img.shape) > 2:
        x,y, colors = img.shape

        pad_img = np.pad(img, pad_width=((u_pad, u_pad), (v_pad, v_pad), (0,0)), mode='constant', constant_values = 0)

        cross_img = 0
        for i in range(x):
            for j in range(y):
                for color in range(colors):
                    cross_img = cross_img +  kernel* pad_img[i:i+u, j:j+v, color]

    #grayscale
    else:
        pad_img = np.pad(img, pad_width=((u_pad, u_pad), (v_pad, v_pad)), mode='constant', constant_values = 0)

        cross_img = 0
        for i in range(x):
            for j in range(y):
                    cross_img = cross_img +  kernel* pad_img[i:i+u, j:j+v]
    return cross_img

def conv_2d(img, kernel):
    '''
    INPUT:
        img: RGB image (height x width x 3) or grayscale image (height x width) as a numpy array
        kernel: numpy array to specify the transformation
    OUTPUT:
        Returns an image with the same initial dimesnions as  the input
    '''
    return conv_img

def gauss_blur_2d(height, width, sigma):
    '''
    INPUT:
        height + width: describes the size of the kernel
        sigma: the width of the gaussian blur
    OUTPUT:
        Returns a kernel to create the resulting blur given the appropriate parameters
    '''

    gauss = np.zeros((width, height))

    x = np.linspace(-np.floor(width/2), np.floor(width/2), width)
    y = np.linspace(-np.floor(height/2), np.floor(height/2), height)
    
    for i in range(0, width):
        for j in range (0, height):
            gauss[i,j] = 1/(2*np.pi*sigma**2) * np.exp(- (x[i]**2 + y[j]**2)/(2* sigma**2))

    return gauss

def lowpass(img, size, sigma):
    '''
    INPUT:
        img: RGB or grayscale image
        size: Sware size of the kernel
        sigma: width of the gaussian blur
    OUTPUT:
        Returns an image with the lowpass filter applied to it
    '''
    return conv_2d(img, gauss_blur_2d(size, size, sigma))

def highpass(img, size, sigma):
    '''
    OUTPUT:
        Returns an the high frequencies (edges) of the image utilizing 
        the complement of a lowpass filter
    '''
    return img - lowpass(img, size, sigma)

def create_img(img1, img2, size1, size2, sigma1, sigma2):
    print("placeholder")