Robotics Library supports various applications such as robot kinematics, dynamics, control, motion planning, and perception. Hopefully
with time we will include some support for various types of sensors as part of a decent sensor fusion. Although there are libraries
out there that can support robotics, they don't seem to support multiple aspects of robotics (i.e. vision, control, kinematics, etc.) and in many cases are potentially clunky to use in conjunction with heavy CAD/CAM/CAE software.

The goal is to create software that allows you to use minimal function calls to compute robot kinematics and dynamics if required for
your given problem so as to apply it to robot control. Perception will also be configured similarly with support for computer vision,
machine learning, etc. There may be NLP support for voice commands to the robots, but I haven't quite looked into that yet.

One other area of interest is soft robotics, but the goal here is mainly to simulate how soft robots may react to certain physical/chemical
stimuli.