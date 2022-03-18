import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from scipy.stats import multivariate_normal


def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """ 
    # Random seed initialisation
    rng = np.random.default_rng(seed)

    # Image vectorisation
    Ivec = np.reshape(image, (-1, 3))
    centers = Ivec[rng.integers(low = 0, high = Ivec.shape[0], size = K)]  # Cluster initialisation
    segmentation = np.zeros(Ivec.shape[0]) # Null binary vect image 
    
    # Distance matrix computation
    distances = distance_matrix(Ivec, centers)
    
    for ite in range(L):
        segmentation = np.argmin(distances, axis=1)
        prev_centers= np.copy(centers)
        for k in range(K):
            centers[k] = np.mean(Ivec[np.where(segmentation == k)], axis = 0)

        distances = distance_matrix(Ivec, centers)
        
        evo = np.linalg.norm(prev_centers-centers)
        # print(ite, ':', evo)
        
    if len(image.shape) == 3:
        segmentation = np.reshape(segmentation, (image.shape[:2]))
    else:
        segmentation = np.reshape(segmentation, (image.shape[:1]))
    return segmentation, centers


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """ 
    
    
    image = image / 255
        
    image = image / 255
    Ivec = np.reshape(image, (-1, 3)).astype(np.float32)
    used_pixels = image[mask == 1]
    
        
    # Plotting image and cropped image
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    
    # plt.figure()
    # plt.imshow(np.reshape(used_pixels, (X_masked, Y_masked, 3)))
    # plt.show()

    # ---  Init of K components using K-means
    # Mean init
    segmentation, Mu = kmeans_segm(used_pixels, K, L=25, seed = np.random.randint(1))
    
    W = np.zeros(K)
    N = used_pixels.shape[0]
    
    # Weights Init
    for k in range(K):
        W[k] = np.count_nonzero( segmentation == k ) / N
    
    Cov = np.array([np.eye(3) for i in range(K)])
    
    # Components init
    components = np.ones((N, K))
    prob_mask = np.zeros((N, K))

    for ite in range(L):
        # Expectation: Compute probabilities P_ik using masked pixels
        for k in range(K):
            components[:, k] = W[k] * multivariate_normal(mean=Mu[k], cov=Cov[k]).pdf(used_pixels)
            
        for k in range(K):
            prob_mask[:, k] = components[:, k] / np.sum(components, axis=1)
        # Maximization: Update weights, means and covariances using masked pixels

        for j in range(K):
            W[j] = np.mean(prob_mask[:,j])
            Mu[j,:] = prob_mask[:, j].T @ used_pixels / np.sum(prob_mask[:, j])
            centered = used_pixels - Mu[j,:]
            Cov[j] = centered.T @ (centered * np.reshape(prob_mask[:, j], (-1, 1))) / np.sum(prob_mask[:, j])

    prob = np.zeros((np.shape(Ivec)[0], K))
    
    for i in range(K):
        prob[:, i] = W[i] * multivariate_normal(mean=Mu[i], cov=Cov[i]).pdf(Ivec)
        prob[:, i] = prob[:, i] / np.sum(prob[:, i])
        
    prob = np.reshape(np.sum(prob, axis=1), (image.shape[0], image.shape[1]))
    return prob
