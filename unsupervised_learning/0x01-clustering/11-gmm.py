#!/usr/bin/env python3
""" Gaussian Mixture Model """
import sklearn.mixture


def gmm(X, k):
    """ Calculate a GMM from a dataset """
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
