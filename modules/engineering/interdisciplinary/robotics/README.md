Robotics Library supports various applications such as robot kinematics, dynamics, control, motion planning, and perception. Hopefully
with time we will include some support for various types of sensors as part of a decent sensor fusion. Although there are libraries
out there that can support robotics, they don't seem to support multiple aspects of robotics (i.e. vision, control, kinematics, etc.) and in many cases are potentially clunky to use in conjunction with heavy CAD/CAM/CAE software.

The goal is to create software that allows you to use minimal function calls to compute robot kinematics and dynamics if required for
your given problem so as to apply it to robot control. Perception will also be configured similarly with support for computer vision,
machine learning, etc. There may be NLP support for voice commands to the robots, but I haven't quite looked into that yet.

One other area of interest is soft robotics, but the goal here is mainly to simulate how soft robots may react to certain physical/chemical
stimuli.

Much of the code for robot kinematics, dynamics, and control utilizes variations of the functions used in Kevin M. Lynch's Modern Robotics: Mechanics, Planning, and Control textbook and source code. Main changes to the code are to compute kinematics, dynamics, etc. for all frames required for the robot. Dr. Lynch's book mostly uses end-effector calculations due to the chapter in robotic manipulation. In most robotic computations, the joint configuration frame is also required to provide useful functionality to the robot. For planning we also consider motion planning in regards to the linkages of the robot as well as the control of the linkages and joints using torque control.
To show other control topics from the book, we also consider other methods of control such as impedance and joint velocity control. Trajectory optimization and generation do not change much from the book source code.

http://hades.mech.northwestern.edu/index.php/Modern_Robotics
