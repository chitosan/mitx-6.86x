import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    A = np.random.random([n,1])
    return A
    raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return A, B, s
    raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    s = np.linalg.norm(A+B)
    return s
    raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    z = np.tanh(np.matmul(weights.T, inputs))
    return z
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x<=y:
        z = x*y
    else:
        z = x/y
    return z
    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    c = np.vectorize(scalar_function)(x, y)
    return c
    raise NotImplementedError

def distance_line_point(theta, theta_0, Xn, Yn):
    """
    calculates the distance between line (theta * a + theta_0) and point a = (a1, a2).
    d = _(0*a + 0o)_            d = _(theta * a + theta_0)_
           ||0||                           ||theta||
    Arg:
        inputs theta NumPy array
        inputs theta_0 NumPy array
        inputs Xn n x 1 NumPy array
        inputs Yn n x 1 NumPy array
    Returns (in this order):
        out - a 1 x 1 NumPy array, representing the the distance d for (x,y) E Sn
    """
    a = np.vstack((Xn,Yn))
    d = (np.matmul(theta, a) + theta_0) / np.linalg.norm(theta)
    return d


v = randomization(3)
print(v)
print(v.T)
print("")

A, B, s = operations(5,3)
print(A)
print(B)
print(s)

A = np.array([1,2])
B = np.array([2,2])
s = norm(A, B)
print(A)
print(B)
print(s)
print("\n")

# A = randomization(10)
# B = randomization(10)
A = np.array([0.67996147,0.42809705,0.96261992,0.88771512,0.60105649,0.05127345,0.14416819,0.77735598,0.47073118,0.69904909])
B = np.array([0.56261346,0.90882682,0.61188960,0.73015091,0.51883126,0.64649271,0.73747600,0.41865184,0.90141100,0.59611053])
theta = np.array([1,1])
m = -theta[1]/theta[0]
theta_0 = 1
print(theta)
print("Xn")
print(A)
print("Yn")
print(B)
r = distance_line_point(theta,theta_0,A,B)
print("result")
print(r)

plt.plot(A, m*A+theta_0, linestyle='solid')
plt.scatter(A,B)
plt.show()
