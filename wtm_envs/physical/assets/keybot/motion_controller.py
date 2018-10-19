# Controller class for the Keybot robot
# Gnu License
# Erik Strahl. Modified by Fares Abawi


import math
import time
import pypot.robot
import sys
from threading import Thread

import numpy as np
from numpy import linalg as la
from pypot.robot import from_json

from ikpy import plot_utils
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from random import uniform
import datetime
import os
from subprocess import call


class Keybot:
    keybot_robot_config = {
        'controllers': {
            'my_dxl_controller': {
                'sync_read': False,
                'attached_motors': ['base'],
                'port': 'auto'
            }
        },
        'motorgroups': {
            'base': ['rotator', 'distancer', 'player']
        },
        'motors': {
            'rotator': {
                'orientation': 'direct',  # indirect
                'type': 'AX-12',
                'id': 10,
                'angle_limit': [-25.0, 25.0],
                'offset': 0.0
            },
            'distancer': {
                'orientation': 'direct',
                'type': 'AX-12',
                'id': 11,
                'angle_limit': [-90.0, 10.0],
                'offset': 0.0
            },
            'player': {
                'orientation': 'direct',  # indirect
                'type': 'AX-12',
                'id': 12,
                'angle_limit': [-90.0, 10.0],
                'offset': 0.0
            }
        }
    }

    def __init__(self, bus):
        kb_config = dict(self.keybot_robot_config)
        # kb_config['controllers']['my_dxl_controller']['port']  = 'auto'
        kb_config['controllers']['my_dxl_controller']['port'] = bus
        self.kb_robot = pypot.robot.from_config(kb_config)
        #print self.kb_robot.motors
        self.motor_dict = dict((x.name, x) for x in self.kb_robot.motors)

    def __exit__(self, exc_type, exc_value, traceback):
        # raw_input("Stop synchronizing task")
        self.kb_robot.stop_sync()
        self.kb_robot.close()

    def __enter__(self):
        return self

    def compliant(self, comp_val=True):

        self.kb_robot.compliant = comp_val

    def get_status_string(self):
        ret = ""
        for m in self.kb_robot.motors:
            ret += "\n"
            ret += m.name + " : " + str(m.id) + " : " + str(m.present_position) + " : limits " + str(
                m.angle_limit) + " goal_speed " + str(m.goal_speed)

        return (ret)
        # Read the goal position from command line (in degrees -180 to 180)

    def goto_position(self,motor,goal,duration=1.0,wait=True):
        motor_goal = {motor: goal}
        print(motor_goal)
        self.kb_robot.goto_position(motor_goal, duration=duration, wait=wait)

    def goto_pose(self,pose,duration,wait):

        for motor, properties in self.keybot_robot_config['motors'].items():
            # convert radians to degrees
            limit = properties['angle_limit']
            degrees = math.degrees(pose[motor])
            pose[motor] = degrees
            # print('degrees', degrees)
            # print(m_kb_robot.get_position('rotator'))

            # Check the limits. Might be unnecessary, but used as an extra safety measure
            if pose[motor] < limit[0]:
                pose[motor] = limit[0]
            elif pose[motor] > limit[1]:
                pose[motor] = limit[1]

        self.kb_robot.goto_position(pose, duration=duration, wait=wait)

    def get_position(self,joint_name):
        return(self.motor_dict[joint_name].present_position)

    def get_positions(self):
        angles = []
        for joint_name, joint_value in self.motor_dict.items():
            angles.append(math.radians(self.motor_dict[joint_name].present_position))
        return angles

class Controller:
    ik_keybot_robot_chain = Chain(name='keybot', active_links_mask=[False, True, True, True, False], links=[
        OriginLink(),
        URDFLink(
            name="rotator",
            translation_vector=[0.065, 0, 0.03], # [0.7, 0.4, 0.52]
            orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-1.5, 1.5)
        ),
        URDFLink(
            name="distancer",
            translation_vector=[0.057, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-0.99, 0.99)
        ),
        URDFLink(
            name="player",
            translation_vector=[0.055, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-0.99, 0.99)
        ),
        URDFLink(
            name="last-link",
            translation_vector=[0.060, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        )
    ])

    def __init__(self, interval_max=1, bus='/dev/ttyACM0'):
        self._interval = 0
        self.interval_max = interval_max
        self._bus = bus

    def move_to_pose(self, pose, duration=0.5, wait=True):
        if self._interval < self.interval_max:
            self._interval += 1
            return
        else:
            with Keybot(self._bus) as m_kb_robot:
                m_kb_robot.compliant = False
                m_kb_robot.goto_pose(pose, duration, wait)
                # Get back the positions after actuation
                angles = m_kb_robot.get_positions()
            self._interval = 0
            return angles

    def move_to_pose_threaded(self, pose, target_vector=None):
        t = Thread(target=self.move_to_pose, args=(pose))
        t.start()

    def compute_inverse_kinematics(self, target_location):
        target_vector = target_location  # 0.37222645, -0.08202711, -0.14208141
        target_frame = np.eye(4)
        target_frame[:3, 3] = target_vector
        # Perform inverse kinemaics
        angles = self.ik_keybot_robot_chain.inverse_kinematics(target_frame)
        return {"rotator": angles[1], "distancer": angles[2], "player": angles[3]}

    def compute_forward_kinematics(self, angles):
        # Append and prepend for dummy links
        angles.insert(0, 0.0)
        angles.append(0.0)
        # Perform forward kinematics
        real_frame = self.ik_keybot_robot_chain.forward_kinematics(angles)
        return real_frame

    def ctrl_set_action(self, action, plot_frame=True):
        # Take only the first 3 elements of the action which represent the position
        action = action[:3]
        # Flipping the x axis due to difference in convention
        # action[0] = -action[0]
        # action[1] = -action[1]

        pose = self.compute_inverse_kinematics(action)
        angles = self.move_to_pose(pose)
        observation = self.compute_forward_kinematics(angles)

        observation = observation[:3,3]

        if plot_frame:
            self.plot_motion(observation, angles, action)

        # Flip back the x axis on the observation
        # observation[0] = -observation[0]
        # observation[1] = -observation[1]
        return observation

    def plot_motion(self, real_vector, joint_angles, target_vector):
        # print("Computed position vector : %s, original position vector : %s" % (real_vector, target_vector))
        # Plot the resultant robot position
        import matplotlib.pyplot as plt
        ax = plot_utils.init_3d_figure()
        self.ik_keybot_robot_chain.plot(joint_angles, ax, target=target_vector)

        ax.set_ylim3d([-0.4, 0.4])
        ax.set_ylabel('Y')

        ax.set_xlim3d([0.0, 0.4])
        ax.set_xlabel('X')

        ax.set_zlim3d([-0.4, 0.4])
        ax.set_zlabel('Z')


        plt.show()

if __name__ == '__main__':
    #     # goal = {str(sys.argv[2]): float(sys.argv[1])}
    with Keybot('/dev/ttyACM0') as m_kb_robot:

        print(m_kb_robot.get_status_string())

        input()

        m_kb_robot.compliant = False
        # m_kb_robot.goto_position(str(sys.argv[2]),float(sys.argv[1]), duration=1, wait=True)

        time.sleep(1)

        print("\ngoto pose")
        pose = { "rotator":0,"distancer":0,"player":15}
        m_kb_robot.goto_pose(pose, duration=1, wait=False)
        time.sleep(1)
        print("\ngoto position")
        m_kb_robot.goto_position("rotator", 40, duration=1, wait=True)
        m_kb_robot.goto_position("distancer", -40, duration=1, wait=True)
        m_kb_robot.goto_position("player", -40, duration=0.4, wait=True)
        time.sleep(1)
        # After enter the synchronizing task stops
        input("Press Enter to end...")
