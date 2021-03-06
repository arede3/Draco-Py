import numpy as np

class Matrix(object):

    def __init__(self, data, shape, dtype=np.float64):
        self.data = np.asarray(data,dtype=dtype)
        self.shape = shape
        self.size = self.size()
        self.data.shape = self.shape
        self.T = None
        self.dtype = dtype

    def __repr__(self):
        return str(self)
        
    def __eq__(self, matrix):

        return matrix.shape == self.data.shape and (self.data == matrix.data).all()

    def __ne__(self, matrix):

        return matrix.shape != self.shape or not ((self.data == matrix.data).all())

    def __add__(self, other):

        if other.shape == self.shape and type(other.data) == type(self.data):
            return Matrix(self.data + other.data, self.shape)

        raise Exception('Must be mxn!')
    
    def __sub__(self, other):
        
        if other.shape == self.shape and type(other) == type(np.asarray([1])):
            return Matrix(self.data - other.data, self.shape, self.size)

        raise Exception('Must be mxn!')
    
    def __mul__(self, other):
        
        if type(other) == type(int(1)) or type(other) == type(float(1)):
            return Matrix(self.data*other,self.shape)
        
        if self.shape[1]  == other.shape[0]:
            return Matrix(self.data.dot(other.data),(self.shape[0],other.shape[1]))  

        raise Exception('Must be mxn X nxp!')
    
    def __truediv__(self, other):
        
        if type(other) == type(int(1)) or type(other) == type(float(1)):
            return Matrix(self.data/other,self.shape)
        
        if self.shape == other.shape:
            return Matrix(self.data / other.data, self.shape)

        raise Exception('Must be mxn!')

    def __rmul__(self, other):
        
        if type(other) == type(int(1)) or type(other) == type(float(1)):
            return Matrix(self.data*other,self.shape)
        
        if self.shape[1]  == other.shape[0]:
            return Matrix(other.data.dot(self.data),(other.shape[0],self.shape[1]))

        raise Exception('Must be mxn X nxp!')

    def __rtruediv__(self, other):
        
        if type(other) == type(int(1)) or type(other) == type(float(1)):
            return Matrix(self.data/other,self.shape)
        
        if self.shape == other.shape:
            return Matrix(self.data / other.data, self.shape)

        raise Exception('Must be mxn!')

        
    def __str__(self):
        return '%s' % str(self.data)

    def __hash__(self):
        return hash(Matrix(self.data, self.shape))

    def __getitem__(self,item):

        if type(self.data[item]) == type(self.dtype(1)):
            return self.data[item]
        
        if len(self.data[item]) > 1:
            return Matrix(self.data[item],self.data[item].shape)
        elif len(self.data[item]) == 1:
            return Matrix(self.data[item],self.data[item].shape)
        else:
            raise Exception('Cannot get the matrix values! Out of bounds error!')

    def size(self):

        prod_size = 1
        S = len(self.shape)
        for i in range(S):
            prod_size *= self.shape[i]

        return prod_size
    
    def dot(self, matrix):
        
        prod = np.dot(self.data,matrix.data)
        return Matrix(prod,prod.shape)

    def outer_product(self, matrix):
        
        outer = np.outer(self.data, matrix.data)
        return Matrix(outer,outer.shape)

    def inner_product(self, matrix):
        
        inner = np.inner(self.data, matrix.data)
        return Matrix(inner,inner.shape)

    def vdot(self, matrix):
        return np.vdot(self.data, matrix.data)

    def tensordot(self, matrix):
        
        tensordot = np.tensordot(self.data, matrix.data)
        return Matrix(tensordot,tensordot.shape)

    def multiply(self, matrix):
        
        prod = np.multiply(self.data, matrix.data)
        return Matrix(prod,prod.shape)

    def divide(self, matrix):
        
        quotient = np.divide(self.data, matrix.data)
        return Matrix(quotient,quotient.shape)

    def inverse(self):
        
        inv = np.linalg.inv(self.data)
        return Matrix(inv,inv.shape)

    def pseudo_inverse(self):
        
        pinv = np.linalg.pinv(self.data)
        return Matrix(pinv,pinv.shape)

    def cholesky(self):

        ch = np.linalg.cholesky(self.data)
        return Matrix(ch,ch.shape)

    def cond(self):

        return np.linalg.cond(self.data)

    def qr(self,mode='reduced'):
        
        M = np.linalg.qr(self.data,mode)
        return Matrix(M,M.shape)
    
    def transpose(self):
        
        t = self.data.T
        shape = t.shape
        self.T = Matrix(t,shape)

        return self.T

    def norm(self):
        
        n = np.linalg.norm(self.data)
        return n

    def matmul(self,matrix):
        
        prod = np.matmul(self.data, matrix.data)
        return Matrix(prod,prod.shape)

    def reshape(self,shape):
        return Matrix(np.reshape(self.data,shape),shape)

    def determinant(self):
        return np.linalg.det(self.data)

    def trace(self):
        return np.trace(self.data)

    def eigen_values(self):
        
        eigen_vals = np.linalg.eig(self.data)[0]
        eigen_vals.shape = (max(self.data.shape),1)
        return Matrix(eigen_vals,eigen_vals.shape)

    def eigen_vectors(self):
        
        eigen_vecs = np.linalg.eig(self.data)[1]
        return Matrix(eigen_vecs,eigen_vecs.shape)

    def singular_value_decomposition(self):
        
        svd = np.linalg.svd(self.data)
        return Matrix(svd[0],svd[0].shape), Matrix(svd[1], svd[1].shape), Matrix(svd[2], svd[2].shape)

def rank(M):
    N = np.linalg.matrix_rank(M.data)
    return N

def power(M):
    K = np.linalg.matrix_power(M.data)
    return Matrix(K,K.shape)

def solve(A,b):
    
    x = np.linalg.solve(A.data,b.data)
    return Matrix(x,x.shape)

##def lstsq(M,N):
##
##    val = np.linalg.lstsq(M,N)
##    return Matrix(val,val.shape)

def concatenate(a,b,axis=None):
    if axis == None:
        m = np.concatenate((a.data,b.data),axis=axis)
        return Matrix(m,m.shape)
    elif axis == 0:
        m = np.concatenate((a.data,b.data),axis=axis)
        return Matrix(m,m.shape)
    elif axis == 1:
        m = np.concatenate((a.data,b.data),axis=axis)
        return Matrix(m,m.shape)
    else:
        raise Exception('Only supports axes: None, 0, and 1')
        
def asmatrix(python_iterable):
    
    arr = np.asarray(python_iterable)
    return Matrix(arr,arr.shape)
    
def asarray(python_iterable_or_matrix):

    if type(python_iterable_or_matrix) == type(Matrix(np.asarray([[1]]))):
        return python_iterable_or_matrix.data

    return np.asarray(python_iterable_or_matrix)

def zeros(shape):
    
    M = Matrix(np.zeros(shape),shape)
    return M

def identity(shape):
    
    if shape[0] == shape[1]:
        M = Matrix(np.identity(shape[0]),shape)
        return M
    
    raise Exception('Square matrix required!\n')

def ones(shape):
    
    M = Matrix(np.ones(shape),shape)
    return M

def main():

    arr_list = [[1,0.332,3.32],[4.223,5.7875,6],[7.34,0.8,3.9]]
    arr = np.asarray(arr_list)
    A = Matrix(arr,arr.shape)
    B = asmatrix(arr_list)
    C = 2*A + 0.43*identity(A.shape)
    print(C.tensordot(A))

if __name__ == '__main__':
    main()
