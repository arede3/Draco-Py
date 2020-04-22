import numpy as np

class Probability(object):

     def __init__(self,X):

def factorial(n):
    if n <= 0:
       return 1
    return factorial(n-1)*n;

def permutations_with_repetition(n,k):
    return np.power(k,n)

def sigma_sum(arr):
    return np.sum(arr)

def pi_product(arr):
    prod = 1
    n = len(arr)
    for i in range(n):
        prod *= arr[i]
    return prod

def permutations_of_multi_sets(

def k_combinations(n,k):
    return int(factorial(n)/(factorial(n-k)*factorial(k)))

def k_permutations(n,k):
    return int(factorial(n)/factorial(n-k))

def binomial_distribution():

def basic_counting(m,n):
    return m*n

