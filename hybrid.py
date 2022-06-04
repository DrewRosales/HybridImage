import cv2 as cv
import sys
import numpy as np
import os

def cross_correlation_2d(img, kernel):
    '''
    INPUT:
        img:    RBG image (height x width x 3) or grayscale image (height x width) as a numpy array
        kernel: numpy array to specify the transformation
    
    OUTPUT:
        Returns an image with the same initial dimensions as the input
    '''
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