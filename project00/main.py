import numpy as np

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

