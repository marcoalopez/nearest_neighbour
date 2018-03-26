# -*- coding: utf-8 -*-
# ============================================================================ #
#                                                                              #
#    Nearest_neighbour Script                                                  #
#    A Python script to estimate the nearest neighbour distance and perform    #
#    and nearest neighbour Monte Carlo simulations                             #
#                                                                              #
#    Copyright (c) 2017-present   Marco A. Lopez-Sanchez                       #
#                                                                              #
#    Licensed under the Apache License, Version 2.0 (the "License");           #
#    you may not use this file except in compliance with the License.          #
#    You may obtain a copy of the License at                                   #
#                                                                              #
#        http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                              #
#    Unless required by applicable law or agreed to in writing, software       #
#    distributed under the License is distributed on an "AS IS" BASIS,         #
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#    See the License for the specific language governing permissions and       #
#    limitations under the License.                                            #
#                                                                              #
#    Version 1.0                                                               #
#    For details see: http://marcoalopez.github.io/nearest_neighbour           #
#    download at https://github.com/marcoalopez/nearest_neighbour/releases     #
#                                                                              #
#    Requirements:                                                             #
#        Python version 3.5.x or higher                                        #
#        Numpy version 1.11 or higher                                          #
#        Matplotlib version 1.5.3 or higher                                    #
#        Scipy version 0.13 or higher                                          #
#                                                                              #
# ============================================================================ #

# import required libraries
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')


def MCarlo_mean_dist(trials, sample_size, x_size, y_size):
    """ Apply a Monte Carlo simulation to estimate mean nearest neighbour
    distance in a finite 2D space considering: (1) a defined sample size
    and (2) a random spatial distribution.

    Parameters
    ----------
    trials: positive integer
        the number of trials

    sample_size: positive integer
       the sample size

    x_size: integer or float
        x-axis size

    y_size: integer or float
        y-axis size

    Call functions
    --------------
    - generate_rand_coordinates
    - nearest_neighbor_dist

    Return
    ------
    The mean and standard deviation at a 2-sigma level of the nearest neighbour
    distances (k=1), and a plot with their distribution.
    """

    nn_distances = np.zeros(trials)

    for i in range(trials):
        datapoints = generate_rand_coordinates(sample_size, x_size, y_size)
        distances = nearest_neighbor_dist(datapoints)
        nn_distances[i] = np.mean(distances)

    mu = round(np.mean(nn_distances), 2)
    sigma = round(np.std(nn_distances), 2)

    print(' ')
    print('Mean =', mu)
    print('Standard Deviation (2-sigma) =', sigma * 2)
    print(' ')

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    textstr = 'mean = {a}; Std (1-sigma) = {b}' .format(a=mu, b=sigma)

    plt.figure(tight_layout=True)
    plt.hist(nn_distances, density=True, color='#108ED2', alpha=0.75)
    plt.plot(x, norm.pdf(x, mu, sigma), color='#1F1F1F', linewidth=2)
    plt.xlabel('nearest neighbour distance')
    plt.ylabel('frequency')
    plt.title(textstr)

    return plt.show()


def generate_rand_coordinates(sample_size, x_size, y_size, gen_plot=False):
    """Generate a defined number of random coordinates in a predefined 2D space
    and plot them. Origin is set at (0, 0).

    Parameters
    ----------
    sample_size: positive integer
        number of random points
    x_size: integer
        the size of the grid in the x-axis.
    y_size: integer
        the size of the grid in the y-axis.
    gen_plot: bool
        whether the user wants to generate a plot. Set to False by default.

    Call function
    --------------
    - generate_plot
    """

    x_coord = np.matrix(np.random.uniform(0, x_size, sample_size))
    y_coord = np.matrix(np.random.uniform(0, y_size, sample_size))
    coordinates = np.concatenate((x_coord.T, y_coord.T), axis=1)

    if gen_plot is True:
        generate_plot(coordinates)

    return coordinates


def nearest_neighbor_dist(coordinates):
    """ Estimate the euclidean distances between any pair of neighbours (the first
    nearest neighbour; k=1). This is a naive implementation of the nearest neighbour
    algorithm, so it is probably not very efficient for very large datasets.

    Parameter
    ---------
    coordinates: array-like
        the coordinates of the points

    Returns
    -------
    A Numpy array with the first nearest neighbour euclidean distances
    """

    # estimate euclidean distances between points using distance.pdist (from Scipy.spatial library)
    dist = distance.pdist(coordinates, metric='euclidean')

    # converts a vector-form distance to a square-form distance matrix
    dist_mat = distance.squareform(dist)

    # replace zeros with nan
    dist_mat[dist_mat == 0] = np.nan

    # estimate the size (number of rows and columns) of the matrix
    n, m = dist_mat.shape

    nearest_dist = np.zeros(n)

    for i in range(n):
        nearest_dist[i] = np.nanmin(dist_mat[i])

    return nearest_dist


def generate_plot(coordinates):
    """Generate a plot with the location of centroids.

    Parameters
    ----------
    coordinates: array-like
        the coordinates of the centroids. Each row contains the x and y
        coordinates. E.g. array([[0, 0], [1, 3],...[5, 7]])
    """

    x, y = coordinates[:, 0], coordinates[:, 1]

    plt.figure(tight_layout=True)
    plt.plot(x, y, 'o')
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')

    return plt.show()


def gen_matrix(x, y):
    """Generate a numpy matrix with the coordinates of the points. Each row is
    a coordinate pair. E.g. array([[0, 0], [1, 3],...[5, 7]]).

    Parameters
    ----------
    x: an integer
        the size of the space in the x-axis
    y: an integer
        the size of the space in the y-axis
    """

    x = np.matrix(x)
    y = np.matrix(y)

    return np.concatenate((x.T, y.T), axis=1)
