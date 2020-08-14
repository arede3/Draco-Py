import numpy as np
pi = np.pi
e = np.exp(1)
inf = np.inf
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
class Matrix(object):

    def __init__(self, data, dtype=np.float64):
        self.data = np.asmatrix(data,dtype=dtype)
        self.shape = self.data.shape
        self.size = self.size()
        self.T = self.transpose
        self.dtype = dtype
    
    def __neg__(self):
        return Matrix(-self.data)

    def __pos__(self):
        return Matrix(+self.data)
    
    def __repr__(self):
        return str(self)
        
    def __eq__(self, matrix):

        return matrix.shape == self.data.shape and np.allclose(self.data,matrix.data)

    def __ne__(self, matrix):

        return matrix.shape != self.shape or not (np.allclose(self.data,matrix.data))

    def __add__(self, other):

        if other.shape[0] == self.shape[0]:
            return Matrix(np.add(self.data,other.data))

        raise Exception('Must be mxn!')
    
    def __sub__(self, other):
        
        if other.shape[0] == self.shape[0]:
            return Matrix(np.subtract(self.data,other.data))

        raise Exception('Must be mxn!')
    
    def __mul__(self, other):

        if type(other) == type(int(1)) or type(other) == type(float(1)):
            return Matrix(self.data*float(other))
        
        if self.shape[1]  == other.shape[0]:
            return Matrix(self.data.dot(other.data))  

        raise Exception('Must be mxn X nxp!')
    
    def __truediv__(self, other):
        
        if type(other) == type(int(1)) or type(other) == type(float(1)):
            other = float(other)
            return Matrix(self.data/other)
        
        if self.shape == other.shape:
            return Matrix(self.data/other.data)

        raise Exception('Must be mxn!')

    def __rmul__(self, other):
        
        if type(other) == type(int(1)) or type(other) == type(float(1)):
            return Matrix(self.data)*other
        
        if self.shape[1]  == other.shape[0]:
            return Matrix(other.data.dot(self.data))

        raise Exception('Must be mxn X nxp!')

    def __rtruediv__(self, other):
        
        if self.shape == other.shape:
            return Matrix(other.data/self.data)

        raise Exception('Must be mxn!')
  
    def __str__(self):
        return '%s' % str(self.data)

    def __hash__(self):
        return hash(Matrix(self.data))

    def __getitem__(self,item):

        if type(self.data[item]) == type(self.dtype(1)):
            return self.data[item]
        
        if len(self.data[item]) > 1:
            return Matrix(self.data[item])
        elif len(self.data[item]) == 1:
            return Matrix(self.data[item])
        else:
            raise Exception('Cannot get the matrix values! Out of bounds error!')

    def __setitem__(self,item,value):

       if type(self) == type(value):
           self.data[item] = value.data  

           return self
       else:
           self.data[item] = value
           return self

       raise Exception('Cannot set the matrix values! Out of bounds error!')

    def size(self):

        prod_size = 1
        S = len(self.shape)
        for i in range(S):
            prod_size *= self.shape[i]

        return prod_size
    
    def dot(self, matrix, out=None):
        
        prod = np.dot(self.data,matrix.data, out)
        return Matrix(prod)

    def outer_product(self, matrix, out=None):
        
        outer = np.outer(self.data, matrix.data, out)
        return Matrix(outer)

    def inner_product(self, matrix):
        if self.shape[1] == 1 and matrix.shape[1] == 1:
            self.T()
            return self.T*matrix
        
        inner = np.inner(self.data, matrix.data)
        return Matrix(inner)
    
    def vdot(self, matrix):
        return np.vdot(self.data, matrix.data)

    def tensordot(self, matrix, axes=2):
        
        tensordot = np.tensordot(self.data, matrix.data, axes)
        return Matrix(tensordot)

    def diagonal(self):
        pass

    def sum(self):
        pass
    
    def multiply(self, matrix):
        
        prod = np.multiply(self.data, matrix.data)
        return Matrix(prod)

    def divide(self, matrix):
        
        quotient = np.divide(self.data, matrix.data)
        return Matrix(quotient)

    def inverse(self):
        
        inv = np.linalg.inv(self.data)
        return Matrix(inv)

    def pseudo_inverse(self,rcond=1e-15,hermitian=False):
        
        pinv = np.linalg.pinv(self.data,rcond,hermitian)
        return Matrix(pinv)

    def cholesky(self):

        ch = np.linalg.cholesky(self.data)
        return Matrix(ch)

    def condition_number(self,p=None):

        return np.linalg.cond(self.data,p)

    def qr(self,mode='reduced'):
        
        M = np.linalg.qr(self.data,mode)
        return Matrix(M)
    
    def transpose(self):
        
        t = self.data.T
        self.T = Matrix(t)

        return self.T

    def norm(self,order=2,axis=None,keepdims=False):
        
        n = np.linalg.norm(self.data,order,axis,keepdims)
        return n

    def matrix_multiplication(self,matrix,out=None):
        
        prod = np.matmul(self.data, matrix.data)
        return Matrix(prod)

    def reshape(self,shape):
        data = np.reshape(self.data,shape)
        self.data = data
        self.shape = shape
        return self

    def slogdet(self):
        data = np.linalg.slogdet(self.data)
        M = Matrix(data)
        return M

    def kronecker_product(self,matrix):
        prod = np.kron(self.data,matrix.data)
        M = Matrix(prod)
        return M
    
    def determinant(self):
        return np.linalg.det(self.data)

    def trace(self,offset=0,axis1=0,axis2=1,dtype=None,out=None):
        return np.trace(self.data,offset,axis1,axis2,dtype,out)

    def eigen_values(self,dtype=np.complex128):
        
        eigen_vals = np.linalg.eig(self.data)[0]
        eigen_vals.shape = (max(self.data.shape),1)
        M = Matrix(eigen_vals,dtype)
        M.reshape(eigen_vals.shape)
        return M

    def eigen_vectors(self,dtype=np.complex128):
        
        eigen_vecs = np.linalg.eig(self.data)[1]
        M = Matrix(eigen_vecs,dtype)
        M.reshape(eigen_vecs.shape)
        return M

    def eigen_values_complex_hermitian(self,UPLO='L',dtype=np.complex128):
        eigen_vals = np.linalg.eigh(self.data, UPLO)[0]
        M = Matrix(eigen_values,dtype)
        M.reshape(eigen_vals.shape)
        return M

    def eigen_vectors_complex_hermitian(self,UPLO='L',dtype=np.complex128):
        eigen_vecs = np.linalg.eigh(self.data,UPLO)[1]
        M = Matrix(eigen_vecs,dtype)
        M.reshape(eigen_vecs.shape)
        return M

    def eigen_values_general(self,dtype=np.complex128):
        eigen_vals = np.linalg.eigvals(self.data)
        M = Matrix(eigen_vals,dtype)
        M.reshape(eigen_vals.shape)
        return M

    def eigen_value_general_complex_hermitian(self,UPLO='L',dtype=np.complex128):
        eigen_vals = np.linalg.eigvalsh(self.data,UPLO)
        M = Matrix(eigen_vals,dtype)
        M.reshape(eigen_vals.shape)
        return M
            
    def singular_value_decomposition(self,full_matrices=True,compute_uv=True,hermatian=False):
        
        svd = np.linalg.svd(self.data,full_matrices,compute_uv,hermitian)
        M0 = Matrix(svd[0])
        M1 = Matrix(svd[1])
        M2 = Matrix(svd[2])
        M0.reshape(svd[0].shape)
        M1.reshape(svd[1].shape)
        M2.reshape(svd[2].shape)
        return [M0,M1,M2]
    
    def __len__(self):
        return len(self.data)

    def __pow__(self,n):
        prod = identity(self.shape[0])
        for i in range(n):
            prod *= self

        return prod
    
def rank(M,tol=None,hermitian=False):
    N = np.linalg.matrix_rank(M.data,tol,hermitian)
    return N

def power(M,n=2):
    K = np.linalg.matrix_power(M.data,n)
    M = Matrix(K)
    M.reshape(K.shape)
    return M

def solve(A,b):
    x = np.linalg.solve(A.data,b.data)
    M = Matrix(x)
    M.reshape(x.shape)
    return M

def lstsq(M,N,rcond='warn'):

    val = np.linalg.lstsq(M,N,rcond)
    return Matrix(val)

def concatenate(a,b,axis=None):
    if axis == None:
        m = np.concatenate((a.data,b.data),axis=axis)
        M = Matrix(m)
        M.reshape(m.shape)
        return M
    elif axis == 0:
        m = np.concatenate((a.data,b.data),axis=axis)
        M = Matrix(m)
        M.reshape(m.shape)
        return M
    elif axis == 1:
        m = np.concatenate((a.data,b.data),axis=axis)
        M = Matrix(m)
        M.reshape(m.shape)
        return M
    else:
        raise Exception('Only supports axes: None, 0, and 1')
        
def asMatrix(python_iterable):
    
    arr = np.asarray(python_iterable)
    M = Matrix(arr)
    shape = (len(arr),len(arr[0]))
    M.reshape(shape)
    return M

def asmatrix(python_iterable_or_matrix):

    if type(python_iterable_or_matrix) == type(Matrix(np.asarray([[1]]))):
        return np.asmatrix(python_iterable_or_matrix.data)

    return np.asarray(python_iterable_or_matrix)
    
def asarray(python_iterable_or_matrix):

    if type(python_iterable_or_matrix) == type(Matrix(np.asarray([[1]]))):
        return np.asarray(python_iterable_or_matrix.data) 

    return np.asarray(python_iterable_or_matrix)

def asPythonList(MatrixObject):
    return MatrixObject.data.tolist()

def zeros(shape):
    
    M = Matrix(np.zeros(shape))
    return M

def identity(n):

    M = Matrix(np.identity(n))
    return M
    
def augmented_matrix(A,b):
    return concatenate(A,b,1)

def ones(shape):
    
    M = Matrix(np.ones(shape))
    return M

def sin(x):

    ans = np.sin(x)

    if abs(ans-0.0) < 10**(-3):
        return 0.0
    
    if abs(ans-1.0) < 10**(-3):
        return 1.0
    
    if abs(ans+1.0) < 10**(-3):
        return -1.0

    return ans

def cos(x):

    ans = np.cos(x)

    if abs(ans-0.0) < 10**(-3):
        return 0.0
    
    if abs(ans-1.0) < 10**(-3):
        return 1.0
    
    if abs(ans+1.0) < 10**(-3):
        return -1.0

    return ans

def tan(x):

    ans = np.tan(x)

    if abs(ans-0.0) <= 10**(-3):
        return 0.0
    
    if abs(ans-1.0) <= 10**(-3):
        return 1.0
    
    if abs(ans+1.0) <= 10**(-3):
        return -1.0

    if ans > 10**6:
        return inf

    if ans < -10**6:
        return -inf

    return ans

def cot(x):

    ans = 1/np.tan(x)
    
    return ans

def csc(x):

    ans = 1/np.sin(x)
    
    if abs(ans-0.0) <= 10**(-3):
        return 0.0
    
    if abs(ans-1.0) <= 10**(-3):
        return 1.0
    
    if abs(ans+1.0) <= 10**(-3):
        return -1.0

    if ans > 10**3:
        return inf

    if ans < -10**3:
        return -inf

    return ans

def sec(x):

    ans = 1/np.cos(x)
    
    if abs(ans-0.0) <= 10**(-3):
        return 0.0
    
    if abs(ans-1.0) <= 10**(-3):
        return 1.0
    
    if abs(ans+1.0) <= 10**(-3):
        return -1.0

    if ans > 10**10:
        return inf

    if ans < -10**10:
        return -inf

    return ans

def linspace(x0,xf,num_points):
    
    x = np.linspace(x0,xf,num_points)
    n = len(x)
    M = Matrix(x)
    M.reshape((n,1))
    return M

def logspace(x0,xf,num_points):
    
    x = np.logspace(x0,xf,num_points)
    n = len(x)
    M = Matrix(x)
    M.reshape((n,1))
    return M

def convertMatrixListIntoMatrix(MatrixList):
    (u,v) = MatrixList[0].shape
    n = len(MatrixList)

    for i in range(n):
        if i == 0:
            M = MatrixList[i]
        else:
            M = concatenate(M,MatrixList[i],1)
            
    return M
        
# Linear Algebra for Robotics
def nearZero(u):
    return abs(u) < 1e-6

def skew3(omega):
    return Matrix([[0,-omega[2,0],omega[1,0]],[omega[2,0],0,-omega[0,0]],[-omega[1,0],omega[0,0],0]])

def skew6(V):
    omega = V[0:3,0]
    omega.reshape((3,1))
    v = V[3:6,0]
    v.reshape((3,1))
    I = zeros((4,4))
    I[0:3,0:3] = skew3(omega)
    I[0:3,3] = v
    return I

def skew3ToVec3(omegaSkew):
    return Matrix([[omegaSkew[2,1]],[omegaSkew[0,2]],[omegaSkew[1,0]]])

def skew6ToVec6(VSkew):
    omegaSkew = VSkew[0:3,0:3]
    v = VSkew[0:3,3]
    V = zeros((6,1))
    V[0:3,0] = skew3ToVec3(omegaSkew)
    V[3:6,0] = v
    return V

def cross(u,v):
    return skew3(u)*v

def compute_m(r,f):
    return cross(r,f)

def compute_v(omega,q,h=0):
    return -cross(omega,q) + h*omega

def OmegaVToS(omega,v):
    S = zeros((6,1))
    S[0:3,0] = omega
    S[3:6,0] = v
    return S

def SToOmegaV(S):
    return [S[0:3,0],S[3:6,0]]

def RpToTrans(R,p):
    T = identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = p
    return T

def TransToRp(T):
    return [T[0:3,0:3],T[0:3,3]]

def rotate(omega,theta):
    return identity(3) + float(sin(theta))*skew3(omega) + float(1-cos(theta))*skew3(omega)*skew3(omega)

def G(omega,theta):
    return identity(3)*float(theta) + float(1-cos(theta))*skew3(omega) + float(theta-sin(theta))*skew3(omega)*skew3(omega)

def transform(S,theta):
    [omega,v] = SToOmegaV(S)
    p = G(omega,theta)*v
    R = rotate(omega,theta)
    T = RpToTrans(R,p)
    return T

def adjoint(T):
    adT = zeros((6,6))
    [R,p] = TransToRp(T)
    adT[0:3,0:3] = R
    adT[3:6,0:3] = skew3(p)*R
    adT[3:6,3:6] = R
    return adT

def mftoF(m,f):
    F = zeros((6,1))
    F[0:3,0] = m
    F[3:6,0] = f
    return F

def Ftomf(F):
    return F[0:3,0],F[3:6,0]

def RChangeOfFrame(Rab,Rbc):
    return Rab*Rbc

def TChangeOfFrame(Tab,Tbc):
    return Tab*Tbc

def p3x1ChangeOfFrame(pb,Rab):
    return Rab*pb

def p4x1ChangeOfFrame(pb,Tab):
    return Tab*pb

def omegaChangeOfFrame(omegab,Rab):
    return Rab*omegab

def twistChangeOfFrame(Vb,Tab):
    adT = adjoint(Tab)
    return adT*Vb

def momentChangeOfFrame(mb,Rab):
    return Rab*mb

def wrenchChangeOfFrame(Fb,Tab):
    Tba = Tab.inverse()
    AdT = adjoint(Tba)
    AdT_T = AdT.T()
    return AdT_T*Fb

def ad(V):
    [omega,v] = SToOmegaV(V)
    M = zeros((6,6))
    M[0:3,0:3] = skew(omega)
    M[3:6,0:3] = skew(v)
    M[3:6,3:6] = skew(omega)
    return M
    
def computeSpaceJacobian(SList,thetaList):
    JsList = []
    TList = []
    
    i = 0
    for S in SList:
        if i == 0:
            TList.append(transform(S,thetaList[i]))
        else:
            TList.append(TList[i-1]*transform(S,thetaList[i]))
        i += 1

    Ti = identity(4)
    n = len(SList)
    JsList.append(SList[0])

    for i in range(1,n):
        JsList.append(adjoint(TList[i-1])*SList[i])

    return JsList
    
def computeBodyJacobian(BList,thetaList):
    JbList = []
    TList = []

    n = len(BList)
    for i in range(n-1,0,-1):
        if i == n-1:
            TList.append(transform(BList[i],thetaList[i]))
        elif i < n-1 and i > -1:
            TList.append(transform(BList[i],thetaList[i])*TList[(n-1)-(i+1)])
    
    TList.reverse()

    for i in range(0,n-1):
        if i != n-1:
            JbList.append(adjoint(TList[i].inverse())*BList[i])

    JbList.append(BList[n-1])
                      
    return JbList

def computeStaticsBody(JbList,Fb):
    Jb = convertMatrixListToMatrix(JbList)
    Jb.T()

    return Jb.T*Fb

def computeStaticsSpace(JsList,Fs):
    Js = convertMatrixListToMatrix(JsList)
    Js.T()

    return Js.T*Fs

def checkForKinematicSingularity(JList):
    J = convertMatrixListToMatrix(JList)
    [m,_] = J.shape

    r = rank(J)

    if r >= 6 or r >= m:
        return True

    return False

def computeManipulabilityEllipsoid(JList):
    J = convertMatrixListToMatrix(JList)
    J.T()
    M = J*J.T

    eig_vals = M.eigen_values()
    eig_vecs = M.eigen_vectors()

    principalSemiAxesLengths = Matrix(np.sqrt(eig_vals.data))
    principalAxes = eig_vecs
    mu1 = float(max(principalSemiAxesLengths.data))/float(min(principalSemiAxesLengths.data))
    mu2 = mu1**2
    mu3 = float(np.sqrt(M.determinant()))
    
    return [principalAxes, principalSemiAxesLengths, mu1, mu2, mu3]

def computeForceEllipsoid(JList):
    J = convertMatrixListToMatrix(JList)
    J.T()
    M = J*J.T
    M = M.inverse()

    eig_vals = M.eigen_values()
    eig_vecs = M.eigen_vectors()

    principalSemiAxesLengths = Matrix(np.sqrt(eig_vals.data))
    principalAxes = eig_vecs
    mu1 = float(max(principalSemiAxesLengths.data))/float(min(principalSemiAxesLengths.data))
    mu2 = mu1**2
    mu3 = float(np.sqrt(M.determinant()))
    
    return [principalAxes, principalSemiAxesLengths, mu1, mu2, mu3]

def computeMassMatrix(thetaList,MList,GList,SList):
    n = len(thetaList)
    M = zeros((6,6))

    for i in range(n):
        ddthetaList = [0]*n
        ddthetaList[i] = 1
        M[:,i] = computeInverseDynamics(thetaList, [0] * n, ddthetaList,[0, 0, 0], [0, 0, 0, 0, 0, 0], MList, GList, SList)

    return M

def computeCoriolisMatrix(thetaList,dthetaList,MList,GList,SList):
    return computeInverseDynamics(thetaList, dthetaList, [0] * len(thetaList), [0, 0, 0], [0, 0, 0, 0, 0, 0], MList, GList, SList)

def computeGravityMatrix(thetaList, gravityVector, MList, GList, SList):
    n = len(thetaList)
    return computeInverseDynamics(thetaList,[0] * n, [0] * n, gravityVector, [0,0,0,0,0,0], MList, GList, SList)

def computeEndEffectorForces(thetaList, Ftip, MList, GList, SList):
    n = len(thetalist)
    return computeInverseDynamics(thetaList, [0] * n, [0] * n, [0, 0, 0], Ftip, MList, GList, SList)

def computeForwardDynamics(thetaList, dthetaList, tauList, gravityVector, Ftip, MList, GList, SList):
    M = computeMassMatrix(thetaList,MList,GList,SList)
    c = computeCoriolisMatrix(thetaList, dthetaList, MList, GList, SList)
    g = computeGraviyMatrix(thetaList, gravityVector, MList, GList, SList)
    Fee = computeEndEffectorForces(thetaList, Ftip, MList, GList, SList)
    tau = Matrix([tauList])
    tau.reshape((len(tauList),1))
    return M.inverse()*(tau-c-g-Fee)

def computeInverseDynamics(thetaList, dthetaList, ddthetaList, gravityVector, Ftip, MList, GList, SList):
    n = len(thetaList)
    Mi = identity(4)
    Ai = zeros((6,n))
    AdTi = [[None]] * (n+1)
    Vi = zeros((6,n+1))
    Vdi = zeros((6,n+1))
    Vdi[:,0] = Matrix([[0],[0],[0],[gravityVector[0,0]],[gravityVector[1,0]],[gravityVector[2,0]]])
    AdTi[n] = adjoint((MList[n]).inverse())
    Fi = Matrix(Ftip)
    tauList = [[None]] * n

    for i in range(n):
        Mi = Mi*MList[i]
        Ai[:,i] = adjoint(Mi.inverse())*SList[i]
        AdTi[i] = adjoint(transform(Ai[:,i],-thetaList[i])*MList[i].inverse())
        Vi[:,i+1] = AdTi[i]*Vi[:,i] + Ai[:,i] * dthetaList[i]
        Vdi[:,i+1] = AdTi[i]*Vd[:,i] + Ai[:,i] * ddthetaList[i] + ad(Vi[:,i+1])*Ai[:,i] * dthetaList[i]

    for i in range(n-1, -1, -1):
        AdTi.T()
        adVi = (ad(Vi[:,i+1])).T()
        Fi = AdTi[i+1].T*Fi + GList[i]*Vdi[:,i+1] - adVi.T*GList[i]*Vi[:,i+1]
        Fi.T()
        tauList[i] = Fi.T*Ai[:,i]

    return tauList

def computeForwardKinematicsSpace(M,SList,thetaList):
    T = identity(4)

    n = len(SList)

    for i in range(n):
        Si = SList[i]
        thetai = thetaList[i]
        Ti = transform(Si,thetai)
        T *= Ti

    return T*M

def computeForwardKinematicsBody(M,BList,thetaList):
    T = M

    n = len(BList)

    for i in range(n):
        Bi = BList[i]
        thetai = thetaList[i]
        Ti = transform(Bi,thetai)
        T *= Ti

    return T
    
def computeInverseKinematicsSpace(T,M,SList,theta0List,eOmega,eV,numberOfIterations=20):
    theta = asMatrix([theta0List])
    theta.reshape((len(theta0List),1))
    thetaList = theta0List
    Tsb = computeForwardKinematicsSpace(M,SList,thetaList)
    VbTheta = matrixLog6(Tsb.inverse()*T)
    Vb = skew6ToVec6(VbTheta)
    [omegas,vs] = SToOmegaV(adjoint(Tsb)*Vb)
    curr_iter = 0
    success = omegas.norm() > eOmega or vs.norm() > eV
    theta.T()
    
    while success and curr_iter < numberOfIterations:
       print('Current Iteration: %s' % (curr_iter + 1))
       Js = convertMatrixListIntoMatrix(computeSpaceJacobian(SList,thetaList))
       print('Jacobian:\n%s\n' % (Js))
       print('omegab = %s\n\nvbx = %s\n\nvby = %s\n' % (omegas[2,0],vs[0,0],vs[1,0]))
       theta += Js.pseudo_inverse()*OmegaVToS(omegas,vs)
       theta.T()
       thetaList = asPythonList(theta.T)
       thetaList = thetaList[0]
       Tsb = computeForwardKinematicsBody(M,SList,thetaList)
       VbTheta = matrixLog6(Tsb.inverse()*T)
       Vb = skew6ToVec6(VbTheta)
       [omegas,vs] = SToOmegaV(adjoint(Tsb)*Vb)
       print('theta:\n%s\n\n' % (theta))
       curr_iter += 1

       success = omegas.norm() > eOmega or vs.norm() > eV

    return [thetaList,not success]
                
def computeInverseKinematicsBody(T,M,BList,theta0List,eOmega,eV,numberOfIterations=20):
    theta = asMatrix([theta0List])
    theta.reshape((len(theta0List),1))
    thetaList = theta0List
    Tsb = computeForwardKinematicsBody(M,BList,thetaList)
    VbTheta = matrixLog6(Tsb.inverse()*T)
    Vb = skew6ToVec6(VbTheta)
    [omegab,vb] = SToOmegaV(Vb)
    curr_iter = 0
    success = omegab.norm() > eOmega or vb.norm() > eV
    theta.T()
    
    while success and curr_iter < numberOfIterations:
       print('Current Iteration: %s' % (curr_iter + 1))
       Jb = convertMatrixListIntoMatrix(computeBodyJacobian(BList,thetaList))
       print('Jacobian:\n%s\n' % (Jb))
       print('omegab = %s\n\nvbx = %s\n\nvby = %s\n' % (omegab[2,0],vb[0,0],vb[1,0]))
       theta += Jb.pseudo_inverse()*OmegaVToS(omegab,vb)
       theta.T()
       thetaList = asPythonList(theta.T)
       thetaList = thetaList[0]
       Tsb = computeForwardKinematicsBody(M,BList,thetaList)
       VbTheta = matrixLog6(Tsb.inverse()*T)
       Vb = skew6ToVec6(VbTheta)
       [omegab,vb] = SToOmegaV(Vb)
       print('theta:\n%s\n\n' % (theta))
       curr_iter += 1

       success = omegab.norm() > eOmega or vb.norm() > eV

    return [thetaList,not success]

def createUnitAxes():
    return [Matrix([[1],[0],[0]]),Matrix([[0],[1],[0]]),Matrix([[0],[0],[1]])]
    
def axisAngle3(omegaTheta):
    return [omegaTheta/float(omegaTheta.norm()), float(omegaTheta.norm())]

def axisAngle6(STheta):

    omegaTheta = STheta[0:3,0:3]
    v = STheta[0:3,3]

    [_,theta] = axisAngle3(omegaTheta)

    if nearZero(theta):
        theta = v.norm()

    return [STheta/float(theta),float(theta)]
                
def matrixLog3(R):
    I = identity(3)
    acosinput = 0.5*(R.trace()-1)
    if acosinput >= 1:
        return zeros((3,3))
    elif acosinput <= -1:
        r11 = R[0,0]
        r21 = R[1,0]
        r31 = R[2,0]
        r12 = R[0,1]
        r22 = R[1,1]
        r32 = R[2,1]
        r13 = R[0,2]
        r23 = R[1,2]
        r33 = R[2,2]
        omega = None
        if nearZero(1+r33):
            omega = (1/(np.sqrt(2*(1+r33))))*Matrix([[r13],[r23],[1+r33]])
        elif nearZero(1+r22):
            omega = (1/(np.sqrt(2*(1+r22))))*Matrix([[r12],[1+r22],[r32]])
        else:
            omega = (1/(np.sqrt(2*(1+r11))))*Matrix([[1+r11],[r21],[r31]])

        return skew3(pi * omega)
    else:
        theta = np.arccos(0.5*(R.trace()-1))
        R.T()
        omegaSkew = float(theta/(2*np.sin(theta)))*(R-R.T)
    
        return omegaSkew

    raise Exception('Error! Check inputs!')

def matrixLog6(T):
    [R,p] = TransToRp(T)
    omegaSkew = matrixLog3(R)
    if omegaSkew == zeros((3,3)):
        M = zeros((4,4))
        M[0:3,0:3] = omegaSkew
        M[0:3,3] = T[0:3,3]
        return M
    else:
        theta = float(np.arccos(0.5*(R.trace()-1)))
        M = zeros((4,4))
        v = Ginv(skew3ToVec3(omegaSkew),theta)*p
        M[0:3,0:3] = omegaSkew
        M[0:3,3] = v
        return M
    
    raise Exception('Error! Check inputs!')

def Ginv(omega,theta):
    I = identity(3)
    omgMat = skew3(omega)
    ginv = I - omgMat / float(2.0) + float(1.0 / float(theta) - 1.0 / np.tan(float(theta) / 2.0) / 2)*omgMat*omgMat / float(theta)

    return ginv

def computeVelocityKinematicsSpace(JsList,thetaDotList):
    Vs = zeros((6,1))
    n = len(JsList)
    for i in range(n):
        Vs += JsList[i]*thetaDotList[i]

    return Vs

def computeVelocityKinematicsBody(JbList,thetaDotList):
    Vb = zeros((6,1))
    n = len(JbList)
    for i in range(n):
        Vb += JbList[i]*thetaDotList[i]

    return Vb

def generateExample1Inputs():
    L1 = 0.5
    L2 = 1.5
    thetaList = [pi/3,-pi/4,pi/6]
    omegasList = [Matrix([[0],[0],[1]]),Matrix([[0],[-1],[0]]),Matrix([[1],[0],[0]])]
    M = Matrix([[0,0,1,L1],[0,1,0,0],[-1,0,0,-L2],[0,0,0,1]])
    thetaDotList = [0.2,1,0.5]
    return [L1,L2,thetaList,omegasList,M,thetaDotList]

def generateExample2Inputs():
    L1 = 0.5
    L2 = 1.5
    thetaList = [pi/3,-pi/4,pi/6]
    omegabList = [Matrix([[-1],[0],[0]]),Matrix([[0],[-1],[0]]),Matrix([[0],[0],[1]])]
    M = Matrix([[0,0,1,L1],[0,1,0,0],[-1,0,0,-L2],[0,0,0,1]])
    thetaDotList = [0.2,1,0.5]
    return [L1,L2,thetaList,omegabList,M,thetaDotList]

def generateExample3Inputs():
    M = identity(4)
    M[0,3] = 2

    BList = [Matrix([[0],[0],[1],[0],[2],[0]]),Matrix([[0],[0],[1],[0],[1],[0]])]

    Tsd = Matrix([[-0.5,-0.866,0,0.366],[0.866,-0.5,0,1.366],[0,0,1,0],[0,0,0,1]])

    eOmega = 0.001
    eV = 0.0001
    
    numberOfIterations = 1000

    theta0List = [0,pi/6]

    return [Tsd,M,BList,theta0List,eOmega,eV,numberOfIterations]

def solveExampleProblem1():
    
    [L1,L2,thetaList,omegasList,M,thetaDotList] = generateExample1Inputs()
    
    
    [omegas1,omegas2,omegas3] = omegasList
    
    qs1 = zeros((3,1))
    qs2 = Matrix([[L1],[0],[0]])
    qs3 = Matrix([[0],[0],[-L2]])

    vs1 = compute_v(omegas1,qs1)
    vs2 = compute_v(omegas2,qs2)
    vs3 = compute_v(omegas3,qs3)
    
    S1 = OmegaVToS(omegas1,vs1)
    S2 = OmegaVToS(omegas2,vs2)
    S3 = OmegaVToS(omegas3,vs3)
    
    SList = [S1,S2,S3]

    Tee = computeForwardKinematicsSpace(M,SList,thetaList)
    JsList = computeSpaceJacobian(SList,thetaList)
    Vs = computeVelocityKinematicsSpace(JsList,thetaDotList)
    
    return Tee,SList,JsList,Vs

def solveExampleProblem2():
    
    [L1,L2,thetaList,omegabList,M,thetaDotList] = generateExample2Inputs()
    
    
    [omegab1,omegab2,omegab3] = omegabList
    
    qb1 = Matrix([[-L2],[0],[-L1]])
    qb2 = Matrix([[-L2],[0],[0]])
    qb3 = zeros((3,1))

    vb1 = compute_v(omegab1,qb1)
    vb2 = compute_v(omegab2,qb2)
    vb3 = compute_v(omegab3,qb3)
    
    B1 = OmegaVToS(omegab1,vb1)
    B2 = OmegaVToS(omegab2,vb2)
    B3 = OmegaVToS(omegab3,vb3)
    
    BList = [B1,B2,B3]

    Tee = computeForwardKinematicsBody(M,BList,thetaList)
    JbList = computeBodyJacobian(BList,thetaList)
    Vb = computeVelocityKinematicsBody(JbList,thetaDotList)

    return Tee,BList,JbList,Vb

def solveExampleProblem3():
    [Tsd,M,BList,theta0List,eOmega,eV,numberOfIterations] = generateExample3Inputs()

    [thetaList,success] = computeInverseKinematicsBody(Tsd,M,BList,theta0List,eOmega,eV,numberOfIterations)

    return [thetaList,success]
    
# Linear Algebra for Computer Vision, Image Processing, Etc.

# Linear Algebra for Natural Language Processing

# Linear Algebra for Machine Learning

# Linear Algebra for Signal Processing
def geometricSeries(r,n,a=1):
    if r != 1:
        if abs(r) < 1:
            return a/(1-r)
        else:
            return a*(1-r**n)/(1-r)
    else:
        raise Exception('Geometric series does not exist for r = 1!')
 
# Linear Algebra for Control Systems

# Linear Algebra for Physics

# Linear Algebra for Chemistry

# Linear Algebra for Probability & Statistics

# Linear Algebra for Biology

# Linear Algebra for Engineering & Computer Science
