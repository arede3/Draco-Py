from logistics.system import *

cd('C:\\Users\\irona\\source\\repos\\dracopy\\Draco-Py\\modules\\mathematics\\linear_algebra\\')
print(cwd())

from linear_algebra import *
import xml.etree.ElementTree as ET

urdf = 'C://Users//irona//Desktop//Ohio State Graduate Degree//ECE 5463 - Introduction to Real-Time Robotics//ECE 5463 Project//ur5.urdf'
pi = np.pi
e = np.exp(1)
thetalist = [pi/3,pi/2,-pi/6,pi/4,pi/2,pi/3]
dthetalist = [1,0.3,0.5,-0.1,-0.2,-4]

def string_to_number(string):
        n = len(string)
        num = ''
        i = 0
        for ch in string:
            i += 1
            if ch != ' ' and i != n:
                num+=ch
            else:
                if i == n:
                    num+=ch

        return float(num)

def string_to_matrix_r3(string):
    numbers = []
    n = len(string)
    num = ''
    i = 0
    for ch in string:
        i += 1
        if ch != ' ' and i != n:
            num+=ch
        else:
            if i == n:
                num+=ch

            numbers.append(float(num))
            num = ''

    return Matrix(numbers,(3,1))

class Robot(object):

    def __init__(self,urdf,thetalist,dthetalist):
        self.links = {}
        self.joints = {}
        self.load_urdf(urdf)
        self.thetalist = thetalist
        self.dthetalist = dthetalist
        self.Tsb = zeros((4,4))
        self.Rsb = zeros((3,3))
        self.psb = zeros((3,1))
        self.Slist = []
        self.Blist = []
        self.Js = []
        self.Jb = []
        self.Vs = zeros((6,1))
        self.Vb = zeros((6,1))
        self.Iblist = []
        self.Gblist = []
        self.Gslist = []
        self.Jji = []
        self.T0list = []
        self.T0Slist = []
        self.T0Blist = []
        self.TiSlist = []
        self.TiBlist = []
        self.TSlist = []
        self.TBlist = []
        self.n = len(list(self.joints.values()))
        self.compute_T0()
        self.compute_Blist_and_Slist()
        self.compute_forward_kinematics_space()
        self.modify_TBlist(self.n)
        #self.compute_jacobian_and_twist()
    
    def load_urdf(self,urdf):
        tree = ET.parse(urdf)
        root = tree.getroot()
        
        link_list = root.findall('link')
        joint_list = root.findall('joint')
             
        for link in link_list:
            name = link.attrib['name']
            if name == 'world' or name == 'ee_link':
                self.links[name] = name
            else:
                inertial_data = link.findall('inertial')
                mass = inertial_data[0].findall('mass')[0].attrib['value']
                origin_data = inertial_data[0].findall('origin')[0].attrib
                inertia_data = inertial_data[0].findall('inertia')[0].attrib
                self.links[name] = {'mass': string_to_number(mass), 'rpy': string_to_matrix_r3(origin_data['rpy']),'xyz': string_to_matrix_r3(origin_data['xyz']), 'Ib': Matrix([[string_to_number(inertia_data['ixx']),string_to_number(inertia_data['ixy']),string_to_number(inertia_data['ixz'])],[string_to_number(inertia_data['ixy']), string_to_number(inertia_data['iyy']), string_to_number(inertia_data['iyz'])],[string_to_number(inertia_data['ixz']),string_to_number(inertia_data['iyz']),string_to_number(inertia_data['izz'])]],(3,3))}
        
        for joint in joint_list:
            joint_data = joint.attrib
            name = joint_data['name']
            joint_type = joint_data['type']
            if joint_type == 'continuous' or joint_type == 'revolute':
                parent = joint.findall('parent')[0].attrib['link']
                child = joint.findall('child')[0].attrib['link']
                origin_data = joint.findall('origin')[0].attrib
                axis = joint.findall('axis')[0].attrib['xyz']
                self.joints[name] = {'type': joint_type, 'parent': parent, 'child': child, 'rpy': string_to_matrix_r3(origin_data['rpy']), 'xyz': string_to_matrix_r3(origin_data['xyz']), 'axis': string_to_matrix_r3(axis)}
            elif joint_type == 'prismatic':
                parent = joint.findall('parent')[0].attrib['link']
                child = joint.findall('child')[0].attrib['link']
                origin_data = joint.findall('origin')[0].attrib
                axis = joint.findall('axis')[0].attrib['xyz']
                self.joints[name] = {'type': joint_type, 'parent': parent, 'child': child, 'rpy': string_to_matrix_r3(origin_data['rpy']), 'xyz': string_to_matrix_r3(origin_data['xyz']), 'axis': string_to_matrix_r3(axis)}
            else:
                parent = joint.findall('parent')[0].attrib['link']
                child = joint.findall('child')[0].attrib['link']
                origin_data = joint.findall('origin')[0].attrib
                self.joints[name] = {'type': joint_type, 'parent': parent, 'child': child, 'rpy': string_to_matrix_r3(origin_data['rpy']), 'xyz': string_to_matrix_r3(origin_data['xyz'])}
                
    def rotation(self,omega,theta):
        return identity((3,3)) + sin(theta)*self.skew(omega) + (1-cos(theta))*self.skew(omega)*self.skew(omega)

    def rpy_rotation(self,rpy):
        r = rpy[0][0]
        p = rpy[1][0]
        y = rpy[2][0]
        z_axis = Matrix([[0],[0],[1]],(3,1))
        y_axis = Matrix([[0],[1],[0]],(3,1))
        x_axis = Matrix([[1],[0],[0]],(3,1))
        return self.rotation(z_axis,y)*self.rotation(y_axis,p)*self.rotation(x_axis,r)

    def skew(self,vec):
        return Matrix([[0,-vec[2][0],vec[1][0]],[vec[2][0],0,-vec[0][0]],[-vec[1][0],vec[0][0],0]],(3,3))

    def translation(self,p):
        return identity((4,4)) + Matrix([[0,0,0,p[0][0]],[0,0,0,p[1][0]],[0,0,0,p[2][0]],[0,0,0,0]],(4,4))
    
    def transformation(self,omega,theta,p):
        R4by4 = concatenate(concatenate(self.rotation(omega,theta),Matrix([[0],[0],[0]],(3,1)),1),Matrix([[0,0,0,1]],(1,4)),0)
        return self.translation(p)*R4by4

    def g(self,omega,theta):
        return theta*identity((3,3)) + (1-cos(theta))*self.skew(omega) + (theta-sin(theta))*self.skew(omega)*self.skew(omega)

    def compute_p_from_Gv(self,omega,theta,v):
        return self.g(omega,theta)*v

    def compute_theta_omega_v_from_T(self,T):
        [R,p] = self.Trans_to_Rp(T)

        [omega,theta] = self.rot_log(R)
        Ginv = g(omega,theta).inverse()
        v = Ginv*p
        return omega,theta,v

    def rot_log(self,R):
        I = identity((3,3))
        Rtr = R.trace()
        omega = None
        theta = None
        if R == I:
           omega = 'undefined'
           theta = 0
        elif Rtr == -1:
           r11 = R[0,0]
           r21 = R[1,0]
           r31 = R[2,0]
           omega = (1/sqrt(2*(1+r11)))*Matrix([[1+r11],[r21],[r31]],(3,1))
           theta = pi
        else:
           theta = np.arccos(0.5*(Rtr-1))
           skew_omega = (1/(2*sin(theta)))*(R-R.transpose())
           omega = Matrix([[skew_omega[2,1]],[skew_omega[2,0]],[skew_omega[1,0]]],(3,1))

        return omega,theta

    def compute_T0(self):
        joint_names = list(self.joints.keys())
        
        i = 0
        for name in joint_names:
            if name == 'world_joint':
                T0 = self.Rp_to_Trans(self.rpy_rotation(self.joints[name]['rpy']),self.joints[name]['xyz'])
                self.T0list.append(T0)
                i = i + 1
            else:
                Rypr = self.rpy_rotation(self.joints[name]['rpy'])
                p = self.joints[name]['xyz']
                T0 = self.Rp_to_Trans(Rypr,p)
                self.T0list.append(self.T0list[i-1]*T0)
                i = i + 1
        
    def compute_Mlist_and_Glist(self):
        link_names = list(self.joints.keys())

        i = 0
        for name in link_names:
            return None
        
    def compute_Blist_and_Slist(self):
        joint_names = list(self.joints.keys())[1:len(self.joints)-1]
        i = 1
        for name in joint_names:
            if self.joints[name]['type'] == 'continuous' or self.joints[name]['type'] == 'revolute':
                Bi = concatenate(self.joints[name]['axis'],Matrix([[0],[0],[0]],(3,1)),0)
                self.Blist.append(Bi)
                self.Slist.append(self.adjoint(self.T0list[i])*Bi)
                i += 1
            elif self.joints[name]['type'] == 'prismatic':
                Bi = concatenate(Matrix([[0],[0],[0]],(3,1)),self.joints[name]['axis'],0)
                self.Blist.append(Bi)
                self.Slist.append(self.adjoint(self.T0list[i])*Bi)
                i += 1
            else:
                continue                

    def Rp_to_Trans(self,R,p):
        return concatenate(concatenate(R,p,1),Matrix([[0,0,0,1]],(1,4)),0)

    def Trans_to_Rp(self,T):
        R = np.take(T.data,[[0,1,2],[4,5,6],[8,9,10]])
        p = np.take(T.data,[[3],[7],[11]])
        return [Matrix(R,R.shape),Matrix(p,p.shape)]

    def adjoint(self,T):
        [R,p] = self.Trans_to_Rp(T)
        return concatenate(concatenate(R,zeros((3,3)),1),concatenate(self.skew(p)*R,R,1),0)

    def compute_omega_and_v_from_V(self,V):
        omega = np.take(V.data,[[0],[1],[2]])
        v = np.take(V.data,[[3],[4],[5]])
        return [Matrix(omega,omega.shape),Matrix(v,v.shape)]

    def adV(self,V):
        [omega,v] = self.compute_omega_and_v_from_V(V)
        omega_skew = self.skew(omega)
        v_skew = self.skew(v)
        return concatenate(concatenate(omega_skew,zeros(omega_skew.shape),1),concatenate(v_skew,omega_skew,1),0)

    def compute_jacobian_and_twist(self):
        n = len(self.TBlist)
        
        for i in range(n):
            if i == 0:
               self.Js.append(self.Slist[i])
               self.Vs = self.Vs + self.Js[i]
            else:
               self.Js.append(self.adjoint(self.TSlist[i-1])*self.Slist[i])
               self.Vs = self.Vs + self.Js[i]
        j = 0
        for i in range(n-1,-1,-1):
            if i == n-1:
               self.Jb.append(self.Blist[i])
               self.Vb = self.Vb + self.Jb[j]
               j += 1
            else:
               self.Jb.append(self.adjoint(self.TBlist[n-1-i].inverse())*self.Blist[i])
               self.Vb = self.Vb + self.Jb[j]
               j += 1
        
    def compute_forward_kinematics_space(self):
        joint_names = list(self.joints.keys())[1:len(self.joints)]

        i = 0
        j = 1
        for name in joint_names: 
            if name != 'world_joint':
               if name == 'joint1':
                  [omega,v] = self.compute_omega_and_v_from_V(self.Slist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TiSlist.append(self.Rp_to_Trans(R,p))
                  self.TSlist.append(self.TiSlist[i]*self.T0list[j])
                  i += 1
                  j += 1
               elif name != 'joint1' and name != 'ee_joint':
                  [omega,v] = self.compute_omega_and_v_from_V(self.Slist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TiSlist.append(self.Rp_to_Trans(R,p))
                  m = len(self.TiSlist)
                  T = None
                  for k in range(m):
                      if k == 0:
                         T = self.TiSlist[k]
                      else:
                         T = T*self.TiSlist[k]
                  self.TSlist.append(T*self.T0list[j])
                  i += 1
                  j += 1
               else:
                  m = len(self.TiSlist)
                  T = None
                  for k in range(m):
                      if k == 0:
                         T = self.TiSlist[k]
                      else:
                         T = T*self.TiSlist[k]
                  self.TSlist.append(T*self.T0list[j])
                 
        
            
##    def compute_inverse_kinematics_space(self):
##        return
    def modify_TBlist(self,n):
        temp_TBlist = []
        for i in range(1,n):
                self.compute_forward_kinematics_body(i)
                temp_TBlist.append(self.TBlist[i-1])
        self.TBlist = temp_TBlist
        
    def compute_forward_kinematics_body(self,joint_number):
        joint_names = list(self.joints.keys())[1:len(self.joints)]
        
##        for idx in range(l,0,-1):
        i = 0
        j = 1
        n = joint_number
        for name in joint_names:
            if name != 'world_joint':
               if name == 'joint1':
                  [omega,v] = self.compute_omega_and_v_from_V(self.adjoint(self.T0list[n].inverse())*self.Slist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TiBlist.append(self.Rp_to_Trans(R,p))
##                                  if l == 1:
                  
                  self.TBlist.append(self.T0list[n]*self.TiBlist[i])
                  i += 1
                  j += 1
               elif name != 'joint1' and name != 'ee_joint':
                  [omega,v] = self.compute_omega_and_v_from_V(self.adjoint(self.T0list[n].inverse())*self.Slist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TiBlist.append(self.Rp_to_Trans(R,p))      
                  m = len(self.TiBlist)
                  T = None
                  for k in range(m):
                      if k == 0:
                         T = self.TiBlist[k]
                      else:
                         T = T*self.TiBlist[k]
##                                  if l == 7-i:

                  self.TBlist.append(self.T0list[n]*T)
                  
                  i += 1
                  j += 1
               else:
                  m = len(self.TiBlist)
                  T = None
                  for k in range(m):
                      if k == 0:
                         T = self.TiBlist[k]
                      else:
                         T = T*self.TiBlist[k]
##
                  self.TBlist.append(self.T0list[n]*T)


                
        
##    def compute_inverse_kinematics_space(self)
##        return

##    def compute_forward_dynamics(self):
##        return
##
##    def compute_inverse_dynamics(self):
##        return
def sqrt(x):
    return np.sqrt(x)
def Rlist_and_plist(robot,Tlist):
        Rs = []
        ps = []

        for T in Tlist:
                [R,p] = robot.Trans_to_Rp(T)
                Rs.append(R)
                ps.append(p)
        return Rs,ps

r = Robot(urdf, thetalist, dthetalist)
[Rs,ps] = Rlist_and_plist(r,r.TSlist)
[Rb,pb] = Rlist_and_plist(r,r.TBlist)
[Rs1,Rs2,Rs3,Rs4,Rs5,Rs6,Rs7] = Rs
[Rb1,Rb2,Rb3,Rb4,Rb5,Rb6,Rb7] = Rb

         
