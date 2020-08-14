import numpy as np
from linear_algebra import *
from sympy import *
class Vector3(object):

    def __init__(self,data):
        self.vector = Matrix([data],(3,1))
        self.vector_ijk = '(%s)i + (%s)j + (%s)k' % (self.vector[0,0],self.vector[1,0],self.vector[2,0])
        self.vector_math = '<%s, %s, %s>' % (self.vector[0,0],self.vector[1,0],self.vector[2,0])
        self.x = self.vector[0,0]
        self.y = self.vector[1,0]
        self.z = self.vector[2,0]
        self.magnitude = np.sqrt(self.x**2+self.y**2+self.z**2)
        self.e1 = Matrix([[1],[0],[0]])
        self.e2 = Matrix([[0],[1],[0]])
        self.e3 = Matrix([[0],[0],[1]])

    def __add__(self,other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3([x,y,z])

    def __sub__(self,other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3([x,y,z])

    def __mul__(self,other):
        if type(other) == type(int(0)) or type(other) == type(float(0)):
            x = self.x * other
            y = self.y * other
            z = self.z * other
            return Vector3([x,y,z])

        raise Exception('Vector3 object cannot be multiplied directly to another vector.\n Please use dot or cross product functions.')

    def __radd__(self,other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3([x,y,z])

    def __rsub__(self,other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3([x,y,z])

    def __rmul__(self,other):
        if type(other) == type(int(0)) or type(other) == type(float(0)):
            x = self.x * other
            y = self.y * other
            z = self.z * other
            return Vector3([x,y,z])

        raise Exception('Vector3 object cannot be multiplied directly to another vector.\n Please use dot or cross product functions.')

    def __str__(self):
        return str(self.vector)

    def __repr__(self):
        return str(self.vector)
        def __eq__(self, other):
        return self.vector == other.vector

    def __ne__(self, other):
        return not(self.vector == other.vector)

    def __truediv__(self,other):
        if type(other) == type(float(0)) or type(other) == type(int(0)):
            return Vector3((self.vector.data/other).tolist())
        raise Exception('Vector3 object cannot be divided by another vector!')

    def __rtruediv__(self,other):
        if type(other) == type(float(0)) or type(other) == type(int(0)):
            return Vector3((self.vector.data/other).tolist())
        raise Exception('Vector3 object cannot be divided by another vector!')

    def cross(self,other):
        x = self.y*other.z - self.z*other.y
        y = self.z*other.x - self.x*other.z
        z = self.x*other.y - self.y*other.x
        return Vector3([x,y,z])

    def dot(self,other):
        x = self.x*other.x
        y = self.y*other.y
        z = self.z*other.z
        return x+y+z

    def computeUnitVector(self):
        return Vector3((self.vector.data/self.magnitude).tolist())

    def printVector(self,vector_format=None):
        if vector_format == 'ijk':
            return self.vector_ijk
        elif vector_format == 'calculus':
            return self.vector_math

        return self.vector
    
class Point3D(Vector3):

    def __init__(self):
        pass

class Line3D(object):

    def __init__(self,point1,point2,line_type='line segment',starting_point='point1'):
        
        self.point1 = point1
        self.point2 = point2
        self.line_type = line_type
        
        self.x0 = None
        self.y0 = None
        self.z0 = None
        self.a = None
        self.b = None
        self.c = None
        self.t = None
        self.r = None

        self.start = -inf
        self.end = inf
        self.length = inf

        if self.line_type == 'line segment':
            self.start = point1
            self.end = point2
            self.length = (self.point1-self.point2).magnitude()
        elif self.line_type == 'line':
            self.start = -inf
            self.end = inf
            self.length = inf
        elif self.line_type == 'ray' and starting_point == 'point1':
            self.start = point1
            self.end = inf
            self.length = inf
        elif self.line_type == 'ray' and starting_point == 'point2':
            self.start = point2
            self.end = -inf
            self.length = inf
        else:
            pass
        
        self.findEquationofLines()

    def findEquationofLines(self,min_val=-10,max_val=10,num_of_points=1000):

        point1 = Vector3toMatrix(self.point1)
        point2 = Vector3toMatrix(self.point2)
        
        point1.transpose()
        
        t = linspace(min_val,max_val,num_of_points)
        
        v = point1 - point2
        
        v.transpose()
        
        r = point1.T + t*v.T
        
        [self.a,self.b,self.c] = v
        
        [self.x0,self.y0,self.z0] = point1
        
        self.t = t
        
        self.r = r
        
    def determineIfParallel(self,other):
        self.findEquationofLines()
        other.findEquationofLines()

        prod = self.r.cross(other.r)  

        if np.abs(prod - 0.0) < 0.0001:
            return True

        return False

    def determineIfPerpendicular(self,other):
        self.findEquationofLines()
        other.findEquationofLines()

        prod = self.r.dot(other.r)

        if np.abs(prod - 0.0) < 0.0001:
            return True

        return False
        
def Vector3toMatrix(vector3):
    return vector3.vector

def MatrixtoVector3(matrix):
    print(matrix.shape)
    if matrix.shape == (3,1) or matrix.shape == (1,3):
        return Vector3(matrix.data.tolist())
    raise Exception('Must be a vector of size (1,3) or (3,1)')

class Kinematics(object):

    def __init__(self):
        pass

class Dynamics(object):

    def __init__(self):
        pass



    


        
                

        
        
