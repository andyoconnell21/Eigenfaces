###############################################################################
## Semester:         CS 540 Spring 2020
##
## This File:        pca.py
## Author:           Andy O'Connell
## Email:            ajoconnell2@wisc.edu
## CS Login:         o-connell
##
###############################################################################
##                   fully acknowledge and credit all sources of help,
##                   other than Instructors and TAs.
##
## Persons:          N/A
##
## Online sources:   Lecture Notes and Piazza
##
###############################################################################

from scipy.io import loadmat
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image 


def load_and_center_dataset(filename):
    #Dataset of image
    dataset = loadmat(filename)

    #'Fea' array from the dataset
    x = dataset['fea']
    n = len(x)
    d = len(x[0])
    x = np.array(x)

    #Recenters the dataset around the origin
    x = x - np.mean(x, axis = 0)

    return x


def get_covariance(dataset):
    #Creates an array of the dataset
    x = np.array(dataset)
    x = np.array(x)

    #Takes transpose of dataset
    x = np.transpose(x)

    #Takes dot product of transpose of dataset and dataset
    x = np.dot(x, np.transpose(x))

    #Divides by the length-1 of the dataset
    x = x / (len(dataset) - 1)

    return x

def get_eig(S, m):
    #Eigen decomposition
    value, r = scipy.linalg.eigh(S, eigvals = (len(S) - m, len(S) - 1))
    eigen_vectors = np.dot(r, np.identity(m))

    #Array of eigen vectors
    eigen_vectors = np.array(eigen_vectors)
    
    #Flips the diagonal array into decending order
    eigen_vectors = np.fliplr(eigen_vectors)

    np.array(value)
    value = np.diag(value)

    #Flips the vector so it is diagonal and in desending order
    value = np.fliplr(value)
    value = np.flipud(value)
    
    return value, eigen_vectors

def project_image(image, U):
    #Takes the dot product of the image vector and eigenvector 
    product = np.dot(image, U)
    product = np.array(product)
    
    #Gets transpose of eigen_vectors
    transpose = np.transpose(U)

    #Takes the dot product of product vector and transpose of eigenvectors
    product_transpose = np.dot(product, transpose)
    product_transpose = np.array(product_transpose)

    return product_transpose

def display_image(orig, proj):
    #Reshapes both images to 32x32 vectors
    reshaped_orig = np.reshape(orig, (32,32))
    reshaped_proj = np.reshape(proj, (32,32))
    
    #Figure with 1 row of 2 and sets the corresponding titles
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Original")
    ax2.set_title("Projection")

    #Projection of original image onto axis1
    ax1_projection = ax1.imshow(np.transpose(reshaped_orig), aspect = 'equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(ax1_projection, cax=cax,fraction=0.046)

    #Projection of the projected image onto axis2
    ax2_projection = ax2.imshow(np.transpose(reshaped_proj), aspect = 'equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.04)
    fig.colorbar(ax2_projection, cax=cax, fraction=0.046)

    plt.show(fig)

    return fig


