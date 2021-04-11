import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
g = 9.81  # m/s^2
l_leg = 0.3


def update_paras(zh, vh_actual=None):
    w = np.sqrt(g / zh)
    half_step_length = calc_gait_paras(zh)
    xh_0 = -half_step_length
    vh_0 = 1.5 * w * abs(xh_0)
    xh_1 = -xh_0
    vh_1 = vh_0
    xf_0 = 2 * xh_0
    if vh_actual is None:
        vh_actual = vh_0
    k = -0.1 - 0.05 * (vh_0 - vh_actual)
    print(k)
    xf_1 = (2 + k) * xh_1
    T_sw = calc_T_sw(xh_0, vh_0, xh_1, vh_1, zh)
    paras = {'xh_0': xh_0,
             'vh_0': vh_0,
             'xh_1': xh_1,
             'vh_1': vh_1,
             'xf_0': xf_0,
             'xf_1': xf_1,
             'T_sw': T_sw,
             'zh': zh}
    return paras


def calc_gait_paras(z=0.4):
    leg_length = 0.58
    half_step_length = np.sqrt(leg_length ** 2 - z ** 2)
    return half_step_length


def calc_foot_trajectory_control_points(xf_0, xf_1):
    control_points = np.asarray([
        [xf_0, 0],
        [0.5 * xf_0, 0.05],
        [-0.05, 0.1],
        [0, 0.1],
        [0.05, 0.1],
        [xf_1 + 0.2, 0.1],
        [xf_1, 0]])
    return control_points


def calc_T_sw(xh_0, vh_0, xh_1, vh_1, zh):
    w = np.sqrt(g / zh)
    T_sw = np.log((w * xh_1 + vh_1) / (w * xh_0 + vh_0)) / w
    return T_sw

def calc_hip_and_ankle_position(t, paras):
    xh_t, vh_t = calc_hip_trajectory(t, paras['xh_0'], paras['vh_0'], paras['zh'])
    xzf_t, v_xzf_t = calc_foot_tajectory(t, paras['T_sw'], paras['xf_0'], paras['xf_1'])
    return xh_t, xzf_t[0], xzf_t[1]

def calc_hip_trajectory(t, xh_0, vh_0, zh):
    w = np.sqrt(g / zh)
    xh_t = xh_0 * np.cosh(w * t) + vh_0 * np.sinh(w * t) / w
    vh_t = xh_0 * w * np.sinh(w * t) + vh_0 * np.cosh(w * t)
    return xh_t, vh_t

def calc_foot_tajectory(t, T_sw, xf_0, xf_1):
    control_points = calc_foot_trajectory_control_points(xf_0, xf_1)
    s = t / T_sw
    xzf_t, v_xzf_t = bezier_curve(s, control_points, T_sw)
    return xzf_t, v_xzf_t


def bernstein_poly(k, n, s):
    """
     The Bernstein polynomial of n, k as a function of t
    """

    return comb(n, k) * (s ** k) * (1 - s) ** (n - k)


def bezier_curve(s, points, T_sw):
    """
        Given a set of control points, return the
        bezier curve defined by the control points.

        points should be a 2d numpy array:
               [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        s in [0, 1] is the current phase
        See https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
        :return
    """
    n = points.shape[0] - 1
    B_vec = np.zeros((1, n + 1))
    for k in range(n + 1):
        B_vec[0, k] = bernstein_poly(k, n, s)
    x_d = np.matmul(B_vec, points).squeeze()
    d_points = points[1:] - points[:-1]
    v_d = (1 / T_sw) * n * np.matmul(B_vec[:, :-1], d_points).squeeze()
    return x_d, v_d


def calc_joint_angle(xh, zh, xf, zf):
    l_hf = np.sqrt((xh-xf)**2 + (zh - zf)**2)
    q_knee = -(np.pi - 2 * np.arcsin(np.clip(0.5 * l_hf/l_leg, -1, 1)))
    theta = -q_knee / 2
    q_hip = np.arctan2(xf - xh, zh - zf) + theta
    q_ankle = -q_knee - q_hip
    return np.array([q_hip, q_knee, q_ankle])


def joint_2_cartesian_position(q_vec, l_vec = None):
    '''
    :param q_vec: [q_hip, q_knee]
    :param l_vec: [l_thigh, l_shank]
    :return: [x z]
    '''
    if l_vec is None:
        l_vec = np.array([0.3, 0.3])
    x = 0
    z = 0
    for i in range(len(q_vec)):
        q_sum = np.sum(q_vec[:i+1])
        x += l_vec[i] * np.sin(q_sum)
        z += -(l_vec[i] * np.cos(q_sum))
    return np.asarray([x, z])

def calc_robot_points(xh, zh, q_hip, q_knee, q_ankle):
    l_foot = 0.1
    x_knee = xh + l_leg * np.sin(q_hip)
    x_ankle = x_knee + l_leg * np.sin(q_hip + q_knee)
    x_toe = x_ankle + l_foot * np.cos(q_hip + q_knee + q_ankle)
    z_knee = zh - l_leg * np.cos(q_hip)
    z_ankle = z_knee - l_leg * np.cos(q_hip + q_knee)
    z_toe = z_ankle + l_foot * np.sin(q_hip + q_knee + q_ankle)
    return np.array([[xh, zh],
                     [x_knee, z_knee],
                     [x_ankle, z_ankle],
                     [x_toe, z_toe]])


def plot_trajectory(xh_t, z_h, xzf_t):
    plt.plot(xzf_t[:, 0], xzf_t[:, 1], '.')
    plt.plot(xh_t, z_h * np.ones(xh_t.shape), 'o')
    plt.xlabel('x (m)')
    plt.xlabel('z (m)')
    plt.show()


def main():
    zh = 0.55
    paras = update_paras(zh=zh)
    t_vec = np.linspace(0, paras['T_sw'], num=100)
    xh_t, vh_t = calc_hip_trajectory(t_vec, paras['xh_0'], paras['vh_0'], paras['zh'])
    xzf_t, v_xzf_t = (np.zeros((len(t_vec), 2)), np.zeros((len(t_vec), 2)))
    for i in range(len(t_vec)):
        xzf_t[i], v_xzf_t[i] = calc_foot_tajectory(t_vec[i], paras['T_sw'], paras['xf_0'], paras['xf_1'])
        q_hip, q_knee, q_ankle = calc_joint_angle(xh_t[i], zh, xzf_t[i, 0], xzf_t[i, 1])
        robot_points = calc_robot_points(xh_t[i], zh, q_hip, q_knee, q_ankle)
        plt.plot(robot_points[:, 0], robot_points[:, 1])
    plot_trajectory(xh_t, zh, xzf_t)

if __name__ == '__main__':
    main()