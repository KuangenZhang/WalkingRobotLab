import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

'''
Obtain position and velocity of the hip.
Obtain gait event.
'''

class Biped2D(object):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.ground = p.loadURDF("plane.urdf")
        self.robot = p.loadMJCF("sustech_biped2d.xml", flags = p.MJCF_COLORS_FROM_FILE)[0]
        self.joints = self.get_joints()
        self.n_j = len(self.joints)
        self.base_dof = 3  # degree of freedom of the base
        self.simu_f = 500 # Simulation frequency, Hz
        self.motion_f = 2 # Controlled motion frequency, Hz
        self.q_vec = np.zeros(self.n_j)
        self.dq_vec = np.zeros(self.n_j)
        self.q_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.q_d_mat = np.zeros((self.simu_f * 3, self.n_j-self.base_dof))
        self.t = 0

        self.init_plot()

    def run(self):
        for i in range(int(5e3)):
            self.t += 1 / self.simu_f
            torque_array = self.controller(self.t)
            self.q_vec, self.dq_vec = self.step(torque_array)
            if 0 == i % 20:
                self.update_plot()
            time.sleep(1/self.simu_f)
        p.disconnect()

    def step(self, torque_array):
        self.set_motor_torque_array(torque_array)
        p.stepSimulation()
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        return self.get_joint_states()

    def get_joints(self):
        all_joints = []
        for j in range(p.getNumJoints(self.robot)):
            # Disable motor in order to use direct torque control.
            info = p.getJointInfo(self.robot, j)
            joint_type = info[2]
            if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
                all_joints.append(j)
                p.setJointMotorControl2(self.robot, j,
                                        controlMode=p.VELOCITY_CONTROL, force=0)
        joints = all_joints
        return joints

    def get_joint_states(self):
        '''
        :return: q_vec: joint angle, dq_vec: joint angular velocity
        '''
        q_vec = np.zeros(self.n_j)
        dq_vec = np.zeros(self.n_j)
        for j in range(self.n_j):
            q_vec[j], dq_vec[j], _, _  = p.getJointState(self.robot, self.joints[j])
        return q_vec, dq_vec

    def set_motor_torque_array(self, torque_array = None):
        '''
        :param torque_array: the torque of [lthigh, lshin, lfoot, rthigh, rshin, rfoot]
        '''
        if torque_array is None:
            torque_array = np.zeros(self.n_j)
        for j in range(len(self.joints[self.base_dof:])):
            p.setJointMotorControl2(self.robot, self.joints[j], p.TORQUE_CONTROL, force=torque_array[j])

    def controller(self, t, type='joint'):
        if 'joint' == type:
            return self.joint_controller(t)

    def joint_controller(self, t):
        a = np.deg2rad(40)
        phi_vec = np.r_[np.zeros(3), 0.5/self.motion_f * np.ones(3)]
        q_d_vec = a*np.sin(2*np.pi*self.motion_f*(t-phi_vec))
        q_d_vec[[1, 4]] -= a
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d_vec
        dq_d_vec = 2*np.pi*self.motion_f*a*np.cos(2*np.pi*self.motion_f*(t-phi_vec))
        return self.joint_impedance_controller(self.q_vec[self.base_dof:], self.dq_vec[self.base_dof:], q_d_vec, dq_d_vec)

    def joint_impedance_controller(self, q_vec, dq_vec, q_d_vec, dq_d_vec):
        k = 2000
        b = 0.5
        return k*(q_d_vec-q_vec) + b*(dq_d_vec - dq_vec)

    def init_plot(self):
        self.fig = plt.figure(figsize=(5, 9))
        joint_names = ['lthigh', 'lshin', 'lfoot',
                       'rthigh', 'rshin', 'rfoot', ]
        self.q_d_lines = []
        self.q_lines = []
        for i in range(6):
            plt.subplot(6, 1, i+1)
            q_d_line, = plt.plot(self.q_d_mat[:, i], '-')
            q_line, = plt.plot(self.q_mat[:, i], '--')
            self.q_d_lines.append(q_d_line)
            self.q_lines.append(q_line)
            plt.ylabel('q_{} (rad)'.format(joint_names[i]))
            plt.ylim([-1.5, 1.5])
        plt.xlabel('Simulation steps')
        self.fig.legend(['q_d', 'q'], loc='lower center', ncol=2, bbox_to_anchor=(0.49, 0.97), frameon=False)
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        for i in range(6):
            self.q_d_lines[i].set_ydata(self.q_d_mat[:, i])
            self.q_lines[i].set_ydata(self.q_mat[:, i])
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    robot = Biped2D()
    robot.run()