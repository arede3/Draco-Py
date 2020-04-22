from logistics.system import *

cd('C:\\Users\\irona\\source\\repos\\dracopy\\Draco-Py\\test\\math\\linear_algebra\\')
print(cwd())

from linear_algebra_test import *
import xml.etree.ElementTree as ET

urdf = 'C://Users//irona//Desktop//Ohio State Graduate Degree//ECE 5463 - Introduction to Real-Time Robotics//ECE 5463 Project//ur5.urdf'
pi = np.pi
e = np.exp(1)
thetalist = [pi/2,pi/3,pi/6,-pi/3,-pi/4,pi/2]
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
        self.TsiSlist = []
        self.TsiBlist = []
        self.TiSlist = []
        self.TiBlist = []
        self.TSlist = []
        self.TBlist = []
        self.compute_T0()
        self.compute_Blist_and_Slist()
        self.compute_forward_kinematics_space()
        self.compute_forward_kinematics_body()
        self.compute_jacobian_and_twist()
    
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
        return identity((3,3)) + np.sin(theta)*self.skew(omega) + (1-np.cos(theta))*self.skew(omega)*self.skew(omega)

    def rpy_rotation(self,rpy):
        r = rpy[0][0]
        p = rpy[1][0]
        y = rpy[2][0]
        z_axis = Matrix([[0],[0],[1]],(3,1))
        y_axis = Matrix([[0],[1],[0]],(3,1))
        x_axis = Matrix([[0],[0],[1]],(3,1))
        return self.rotation(z_axis,y)*self.rotation(y_axis,p)*self.rotation(x_axis,r)

    def skew(self,vec):
        return Matrix([[0,-vec[2][0],vec[1][0]],[vec[2][0],0,-vec[0][0]],[-vec[1][0],vec[0][0],0]],(3,3))

    def translation(self,p):
        return identity((4,4)) + Matrix([[0,0,0,p[0][0]],[0,0,0,p[1][0]],[0,0,0,p[2][0]],[0,0,0,0]],(4,4))
    
    def transformation(self,omega,theta,p):
        R4by4 = concatenate(concatenate(self.rotation(omega,theta),Matrix([[0],[0],[0]],(3,1)),1),Matrix([[0,0,0,1]],(1,4)),0)
        return self.translation(p)*R4by4

    def g(self,omega,theta):
        return theta*identity((3,3)) + (1-np.cos(theta))*self.skew(omega) + (theta-np.sin(theta))*self.skew(omega)*self.skew(omega)

    def compute_p_from_Gv(self,omega,theta,v):
        return self.g(omega,theta)*v

    def compute_T0(self):
        joint_names = list(self.joints.keys())
        
        i = 0
        for name in joint_names:
            if name == 'world_joint':
                T0 = self.Rp_to_Trans(self.rpy_rotation(self.joints[name]['rpy']),self.joints[name]['xyz'])
                self.T0list.append(T0)
                self.joints[name]['T0'] = self.T0list[i]
                i = i + 1
            else:
                Rypr = self.rpy_rotation(self.joints[name]['rpy'])
                p = self.joints[name]['xyz']
                T0 = self.Rp_to_Trans(Rypr,p)
                self.T0list.append(self.T0list[i-1]*T0)
                self.joints[name]['T0'] = self.T0list[i]
                i = i + 1
        
    def compute_Mlist_and_Glist(self):
        link_names = list(self.joints.keys())

        i = 0
        for name in link_names:
            return None
        
    def compute_Blist_and_Slist(self):
        joint_names = list(self.joints.keys())[1:len(self.joints)-1]

        for name in joint_names:
            if self.joints[name]['type'] == 'continuous' or self.joints[name]['type'] == 'revolute':
                Bi = concatenate(self.joints[name]['axis'],Matrix([[0],[0],[0]],(3,1)),0)
                self.Blist.append(Bi)
                self.Slist.append(self.adjoint(self.joints[name]['T0'])*Bi)
            elif self.joints[name]['type'] == 'prismatic':
                Bi = concatenate(Matrix([[0],[0],[0]],(3,1)),self.joints[name]['axis'],0)
                self.Blist.append(Bi)
                self.Slist.append(self.adjoint(self.joints[name]['T0'])*Bi)
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
        n = len(self.TiBlist)
        
        for i in range(n):
            if i == 0:
               self.Js.append(self.Slist[i])
               self.Vs = self.Vs + self.Js[i]
            else:
               self.Js.append(self.adjoint(self.TiSlist[i-1])*self.Slist[i])
               self.Vs = self.Vs + self.Js[i]
        j = 0
        for i in range(n-1,-1,-1):
            if i == n-1:
               self.Jb.append(self.Blist[i])
               self.Vb = self.Vb + self.Jb[j]
               j += 1
            else:
               self.Jb.append(self.adjoint(self.TiBlist[n-1-i].inverse())*self.Blist[i])
               self.Vb = self.Vb + self.Jb[j]
               j += 1
        
    def compute_forward_kinematics_space(self):
        joint_names = list(self.joints.keys())[1:len(self.joints)-1]

        i = 0
        j = 1
        for name in joint_names: 
            if name != 'world_joint' or name != 'ee_joint':
               if name == 'joint1':
                  [omega,v] = self.compute_omega_and_v_from_V(self.Slist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TSlist.append(self.Rp_to_Trans(R,p))
                  self.TiSlist.append(self.Rp_to_Trans(R,p))
                  self.joints[name]['TsiS'] = self.Rp_to_Trans(R,p)*self.T0list[j]
                  self.TsiSlist.append(self.joints[name]['TsiS'])
                  i += 1
                  j += 1
               else:
                  [omega,v] = self.compute_omega_and_v_from_V(self.Slist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TSlist.append(self.Rp_to_Trans(R,p))
                  m = len(self.TSlist)
                  T = None
                  for k in range(m):
                      if k == 0:
                         T = self.TSlist[k]
                      else:
                         T = T*self.TSlist[k]
                  self.joints[name]['TsiS'] = T*self.T0list[j]
                  self.TsiSlist.append(self.joints[name]['TsiS'])
                  i += 1
                  j += 1
            
##    def compute_inverse_kinematics_space(self):
##        return

    def compute_forward_kinematics_body(self):
        joint_names = list(self.joints.keys())[1:len(self.joints)-1]
        i = 0
        j = 1
        for name in joint_names:
                
            if name != 'world_joint' or name != 'ee_joint':
               if name == 'joint1':
                  [omega,v] = self.compute_omega_and_v_from_V(self.Blist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TBlist.append(self.Rp_to_Trans(R,p))
                  self.TiBlist.append(self.Rp_to_Trans(R,p))
                  self.joints[name]['TsiB'] = self.T0list[j]*self.Rp_to_Trans(R,p)
                  self.TsiBlist.append(self.joints[name]['TsiB'])
                  i += 1
                  j += 1
               else:
                  [omega,v] = self.compute_omega_and_v_from_V(self.Blist[i])
                  R = self.rotation(omega,self.thetalist[i])
                  p = self.compute_p_from_Gv(omega,self.thetalist[i],v)
                  self.TBlist.append(self.Rp_to_Trans(R,p))
                  m = len(self.TBlist)
                  T = None
                  for k in range(m):
                      if k == 0:
                         T = self.TBlist[k]
                      else:
                         T = T*self.TBlist[k]
                  self.joints[name]['TsiB'] = self.T0list[j]*T
                  self.TsiBlist.append(self.joints[name]['TsiB'])
                  i += 1
                  j += 1


##    def compute_inverse_kinematics_space(self)
##        return

##    def compute_forward_dynamics(self):
##        return
##
##    def compute_inverse_dynamics(self):
##        return

r = Robot(urdf, thetalist, dthetalist)       
         
