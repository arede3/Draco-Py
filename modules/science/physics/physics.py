import numpy as np

class DualBasicEnhanced(object):

    def __init__(self, *args):

        if len(args) == 2:
            value, eps = args
        elif len(args) == 1:
            if isinstance(args[0], (float,int)):
                value, eps = args[0], 0
            else:
                value, eps = args[0].value, args[0].eps
        self.value = value
        self.eps = eps

    def __abs__(self):
        return abs(self.value)

    def __str__(self):
        return "({},{})".format(self.value,self.eps)

    def __repr__(self):
        return str(self)
    
class DualArith(object):
    def __add__(self,other):
        other = Dual(other)
        return Dual(self.value + other.value, self.eps + other.eps)

    def __sub__(self,other):
        other = Dual(other)
        return Dual(self.value + other.value, self.eps + other.eps)

    def __mul__(self,other):
        other = Dual(other)
        return Dual(self.value * other.value, self.eps * other.value + self.value * other.eps)

class DualDiv(object):
    def __truediv__(self, other):
        other = Dual(other)
        if abs(other.value) == 0:
            raise ZeroDivisionError
        else:
            return Dual(self.value / other.value, self.eps / other.value - self.value/ (other.value)**2 * other.eps)

class Dual(DualBasicEnhanced, DualArith, DualDiv):
    pass

def factorial2(n):
    
    if n <= 1:
        return 1

    return n*factorial(n-1)

def fibonacci2(n):

    if n <= 0:
        return 0

    if n > 0 and n <= 2:
        return 1

    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci(n):
    fibonacci_list = []
    fibonacci_list.append(1)
    fibonacci_list.append(1)

    for i in range(2,n+1):
        fibonacci_list.append(fibonacci_list[i-1]+fibonacci_list[i-2])

    return fibonacci_list[n]

def factorial(n):

    factorial_list = []
    factorial_list.append(1)

    for i in range(2,n+1):
        factorial_list.append(factorial_list[i-2]*i)

    return factorial_list[n-1]
    
import time as t

n = 100

start = t.time()
fn2 = factorial2(n)
end = t.time()
print('fn2 = {}.'.format(fn2))
print('Recursive factorial took {} seconds.\n'.format(end-start))

start = t.time()
fn = factorial(n)
end = t.time()
print('fn = {}.'.format(fn))
print('Dynamic programming factorial took {} seconds.\n'.format(end-start))

start = t.time()
Fn2 = fibonacci2(n)
end = t.time()
print('Fn2 = {}.'.format(Fn2))
print('Recursive fibonacci took {} seconds.\n'.format(end-start))

start = t.time()
Fn = fibonacci(n)
end = t.time()
print('Fn2 = {}.'.format(Fn))
print('Dynamic programming factorial took {} seconds.\n'.format(end-start))
