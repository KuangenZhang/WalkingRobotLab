import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib
import planner
import time
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

'''
Obtain position and velocity of the hip.
Obtain gait event.
'''


class Biped2D(object):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])
        p.setGravity(0, 0, -9.81) # Explicitly set gravity.
        self.ground = p.loadURDF("plane.urdf")
        self.robot = p.loadMJCF("sustech_biped2d.xml", flags=p.MJCF_COLORS_FROM_FILE)[0]
        self.base_dof = 3  # degree of freedom of the base
        self.simu_f = 500  # Simulation frequency, Hz
        self.motion_f = 2  # Controlled motion frequency, Hz
        self.zh = 0.55 # height of robot COM.
        self.stance_idx = 0
        self.pre_foot_contact = np.array([1, 0])
        self.foot_contact = np.array([1, 0])

        self.joints = self.get_joints()
        self.n_j = len(self.joints)


        self.q_vec = np.zeros(self.n_j)
        self.dq_vec = np.zeros(self.n_j)
        self.q_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.q_d_mat = np.zeros((self.simu_f * 3, self.n_j - self.base_dof))
        self.t = 0
        self.init_pos_and_vel_of_robot()
        # self.init_vedio()
        # self.init_plot()

    # def init_video(self):
    #     image_info = p.getCameraImage(640, 480)
    #     image = np.array(image_info[2])
    #     self.img_width, self.img_height = (image.shape[1], image.shape[0])
    #     print(image.shape)
    #     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #     self.out_video = cv2.VideoWriter('locomotion.mp4', fourcc, 30, (self.img_width, self.img_height))

    def init_pos_and_vel_of_robot(self):
        self.walking_paras = planner.update_paras(zh=self.zh)
        q_d_vec = np.zeros(self.n_j)
        d_q_d_vec = np.zeros(self.n_j)
        q_d_vec[self.base_dof:], d_q_d_vec[self.base_dof:], vh_t = self.joint_planner(self.walking_paras, t=1 / self.simu_f)
        q_d_vec[1] = self.zh+0.01-0.7
        d_q_d_vec[0] = vh_t
        for j in range(self.n_j):
            p.resetJointState(self.robot, self.joints[j], targetValue=q_d_vec[j], targetVelocity=d_q_d_vec[j])
        p.stepSimulation()

    def run(self):
        for i in range(int(1e5)):
            self.t += 1 / self.simu_f
            self.update_foot_contact_state()
            q_d_vec, dq_d_vec = self.finite_state_controller()
            self.q_vec, self.dq_vec = self.step(q_d_vec, dq_d_vec)
            # self.capture_robot_motion()
            time.sleep(1 / self.simu_f)
        p.disconnect()

    def step(self, q_d_vec, dq_d_vec):
        self.set_motor_pos_and_vel(q_d_vec, dq_d_vec)
        p.stepSimulation()
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        self.pre_foot_contact[:] = self.foot_contact[:]
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[self.q_vec[0], 0, 0.6])
        # self.out_video.release()
        return self.get_joint_states()

    def finite_state_controller(self):
        gait_event = self.foot_contact - self.pre_foot_contact
        swing_idx = 1 - self.stance_idx
        if (gait_event[swing_idx] == 1 and self.t > 0.9 * self.walking_paras['T_sw']) \
                or (self.foot_contact[swing_idx] == 1 and self.t >= self.walking_paras['T_sw']):
            self.stance_idx = 1-self.stance_idx
            self.t = 0
            self.update_walking_paras()
            print('gait event: {}'.format(gait_event))
        q_d_vec, dq_d_vec = self.joint_controller()
        return q_d_vec, dq_d_vec

    def update_walking_paras(self):
        self.walking_paras = planner.update_paras(zh=self.zh, vh_actual=self.dq_vec[0])


    def get_joints(self):
        all_joints = []
        for j in range(p.getNumJoints(self.robot)):
            # Disable motor in order to use direct torque control.
            info = p.getJointInfo(self.robot, j)
            print(info)
            joint_type = info[2]
            if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
                all_joints.append(j)
                p.setJointMotorControl2(self.robot, j,
                                        controlMode=p.VELOCITY_CONTROL, force=0) # Release all joints.
        joints = all_joints
        return joints


    def get_joint_states(self):
        '''
        :return: q_vec: joint angle, dq_vec: joint angular velocity
        '''
        q_vec = np.zeros(self.n_j)
        dq_vec = np.zeros(self.n_j)
        for j in range(self.n_j):
            q_vec[j], dq_vec[j], _, _ = p.getJointState(self.robot, self.joints[j])
        if self.base_dof == 3:
            q_vec[1] += 0.7
        return q_vec, dq_vec

    def set_motor_pos_and_vel(self, q_d_vec, dq_d_vec):
        '''
        :param torque_array: the torque of [lthigh, lshin, lfoot, rthigh, rshin, rfoot]
        '''
        for j in range(len(self.joints[self.base_dof:])):
            p.setJointMotorControl2(self.robot, self.joints[j + self.base_dof],
                                    p.POSITION_CONTROL, q_d_vec[j], dq_d_vec[j])

    def set_motor_torque_array(self, torque_array=None):
        '''
        :param torque_array: the torque of [lthigh, lshin, lfoot, rthigh, rshin, rfoot]
        '''
        if torque_array is None:
            torque_array = np.zeros(self.n_j - self.base_dof)
        for j in range(len(self.joints[self.base_dof:])):
            # p.setJointMotorControl2(self.robot, self.joints[j + self.base_dof], p.POSITION_CONTROL)
            p.setJointMotorControl2(self.robot, self.joints[j + self.base_dof], p.TORQUE_CONTROL, force=torque_array[j])


    def calc_q_d_vec(self, xh_t, xf_t, zf_t):
        q_d_vec = np.zeros(self.n_j - self.base_dof)
        q_d_vec[self.stance_idx * 3:self.stance_idx * 3 + 3] = planner.calc_joint_angle(xh_t, self.zh, 0, 0)
        swing_idx = 1 - self.stance_idx
        q_d_vec[swing_idx * 3:swing_idx * 3 + 3] = planner.calc_joint_angle(xh_t, self.zh, xf_t, zf_t)
        return q_d_vec

    def joint_planner(self, walking_paras, t):
        t = np.clip(t, 1 / self.simu_f, walking_paras['T_sw'])
        # print('t: {:.3f}, T_sw: {:.3f}'.format(t, walking_paras['T_sw']))
        xh_pre, xf_pre, zf_pre = planner.calc_hip_and_ankle_position(t - 1 / self.simu_f, walking_paras)
        xh_t, xf_t, zf_t = planner.calc_hip_and_ankle_position(t, walking_paras)
        q_d_vec = self.calc_q_d_vec(xh_t, xf_t, zf_t)
        q_d_vec_pre = self.calc_q_d_vec(xh_pre, xf_pre, zf_pre)
        dq_d_vec = (q_d_vec - q_d_vec_pre)/(1/self.simu_f)
        vh_t = (xh_t-xh_pre)/(1/self.simu_f)
        return q_d_vec, dq_d_vec, vh_t

    def joint_controller(self):
        q_d_vec, dq_d_vec, _ = self.joint_planner(self.walking_paras, self.t)
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d_vec
        return q_d_vec, dq_d_vec

    def update_foot_contact_state(self):
        '''Detect foot contact with ground plane
        Body 0: plane;
        Body 1: robot
        '''
        foot_name_list = ['lfoot', 'rfoot']
        for x in p.getContactPoints(0, 1):
            info = p.getJointInfo(x[2], x[4])
            part_name = str(info[12].decode())
            for i in range(len(foot_name_list)):
                if part_name in foot_name_list[i]:
                    self.foot_contact[i] = 1
                else:
                    self.foot_contact[i] = 0

    def init_plot(self):
        self.fig = plt.figure(figsize=(5, 9))
        joint_names = [
            'rootx', 'rootz', 'rooty',
            'lthigh', 'lshin', 'lfoot',
            'rthigh', 'rshin', 'rfoot', ]

        self.q_d_lines = []
        self.q_lines = []
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            # q_d_line, = plt.plot(self.q_d_mat[:, i], '-')
            q_line, = plt.plot(self.q_mat[:, i], '--')
            # self.q_d_lines.append(q_d_line)
            self.q_lines.append(q_line)
            plt.ylabel('{}'.format(joint_names[i]))
            plt.ylim([-1.5, 1.5])
        plt.xlabel('Simulation steps')
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        for i in range(6):
            # self.q_d_lines[i].set_ydata(self.q_d_mat[:, i])
            self.q_lines[i].set_ydata(self.q_mat[:, i])
        plt.draw()
        plt.pause(0.001)

    # def capture_robot_motion(self):
    #     image_info = p.getCameraImage(640, 480)
    #     image = np.array(image_info[2])
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     # cv2.imshow('image', image)
    #     # cv2.waitKey(1)
    #     self.out_video.write(image[...,:3])
    #     cv2.imwrite('results/{}.jpg'.format(time.time()))



if __name__ == '__main__':
    robot = Biped2D()
    robot.run()
