from logistics.system import *
from robot import *
cd('C:\\Users\\irona\\source\\repos\\dracopy\\Draco-Py\\modules\\mathematics\\linear_algebra\\')
print(cwd())

from linear_algebra import *
import xml.etree.ElementTree as ET

urdf = 'C://Users//irona//Desktop//Ohio State Graduate Degree//ECE 5463 - Introduction to Real-Time Robotics//ECE 5463 Project//ur5.urdf'
pi = np.pi
e = np.exp(1)
thetalist = [pi/3,pi/2,-pi/6,pi/4,pi/2,pi/3]
dthetalist = [1,0.3,0.5,-0.1,-0.2,-4]
t = 0
eint = 0
r = Robot(urdf,thetalist, dthetalist)

def cubic_time_scaling(t,Tf):
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

def quintic_time_scaling(t,Tf):
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5

def joint_trajectory(theta0, thetaf, Tf, N, method):
    N = np.int64(N)
    t = Tf / (N-1.0)

    traj = zeros((len(theta0),N))
    for i in range(N):
        s = None
        if method == 3:
            s = cubic_time_scaling(t*i,Tf)
        else:
            s = quintic_time_scaling(t*i,Tf)

        traj[:,i] = s*theta0 + (1-s)*thetaf
    traj = traj.transpose()
    return traj

def screw_trajectory(Tsb0, Tsbf, Tf, N, method):
    N = np.int64(N)
    t = Tf / (N-1.0)
    traj = [[None]] * N
    s = None
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(t*i,Tf)
        else:
            s = quintic_time_scaling(t*i,Tf)
        [omega,theta,v] = r.compute_omega_theta_v_from_T((Tsb0.inverse()*Tsbf))
        S = concatenate(omega,v,0)*s
        [omega2,v2] = r.compute_omega_and_v_from_V(S)

        T = r.Rp_to_Trans(r.rotation(omega2,theta),r.compute_p_from_Gv(omega2,theta,v2))
        traj[i] = Tsb0*T

    return traj

def cartesian_trajectory(Tsb0, Tsbf, Tf, N, method):
    N = np.int64(N)
    t = Tf / (N-1.0)
    traj = [[None]] * N
    s = None
    [R0,p0] = r.Trans_to_Rp(Tsb0)
    [Rf,pf] = r.Trans_to_Rp(Tsbf)
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(t*i,Tf)
        else:
            s = quintic_time_scaling(t*i,Tf)
        [omega,theta] = r.rot_log(R0.transpose*Rf)
        R = r.rotation(omega,theta)*s
        R2 = R0*R
        T = r.Rp_to_Trans(R2,s*pf+(1-s)*p0)
        traj[i] = T

    return traj
        
def trajectory(t, X0, Xf, Tf, method, trajectory_type):
    tspan = t
    if trajectory_type == 'cartesian':
        traj = cartesian_trajectory(X0, Xf, Tf, N, method)
    elif trajectory_type == 'screw':
        traj = screw_trajectory(X0, Xf, Tf, N, method)
    else:
        traj = joint_trajectory(X0, Xf, Tf, N, method)
    
    return traj,traj/t

def pid(t, q0, qf, Tf, N, method, Kp, Kd, Ki, trajectory_type='joint'):
    qprev = q0
    dt = 0.01
    t = 0
    
    while qprev != goal:

        
        q = qprev
        qdot = (q-qprev)/dt
        qprev = q

        e = qd-q
        edot = qdotd-qdot
        eint = eint + e*dt

        tau = Kp*e + Kd*edot + Ki*eint
        commandTorque(tau)

        t = t + dt
