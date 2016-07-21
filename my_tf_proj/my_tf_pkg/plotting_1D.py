import sys
import pdb

import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import json

def plot_reconstruction(fig_num, X_original,Y_original, nb_centers, rbf_predictions, colours, markersize=3, title_name='Reconstruction'):
    fig = plt.figure(fig_num)
    plt.xlabel('number of units')
    plt.ylabel('Reconstruction')
    plt.title(title_name)
    plt.plot(X_original, Y_original,'bo', label='Original data', markersize=markersize)
    for i, Y_pred in enumerate(rbf_predictions):
        colour = colours[i]
        K = nb_centers[i]
        plt.plot(X_original, Y_pred, colour+'o', label='RBF'+str(K), markersize=markersize)

def plot_one_func(fig_num, X_original,Y_original, markersize=3, title_name='Reconstruction'):
    fig = plt.figure(fig_num)
    plt.xlabel('number of units')
    plt.ylabel('Reconstruction')
    plt.title(title_name)
    plt.plot(X_original, Y_original,'bo', label='Original data', markersize=markersize)

def plot_errors(nb_centers, rbf_errors,label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of units')
    plt.ylabel('squared error (l2 loss)')
    plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    plt.plot(nb_centers, rbf_errors, colour+'o')
    plt.title("Erors vs centers")

def plot_errors_and_bars(nb_centers, rbf_errors, rbf_error_std, label='Errors', markersize=3, colour='b'):
    plt.xlabel('number of units')
    plt.ylabel('squared error (l2 loss)')
    plt.plot(nb_centers, rbf_errors, colour, label=label, markersize=3)
    plt.plot(nb_centers, rbf_errors, colour+'o')
    plt.errorbar(nb_centers, rbf_errors, yerr=rbf_error_std)
    plt.title("Erors vs units")

def show():
    plt.legend()
    plt.show()

def figure(n=1):
    plt.figure(3)
