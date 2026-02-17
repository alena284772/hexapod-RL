from matplotlib import pyplot as plt
import numpy as np
import mujoco
import roboticstoolbox as rtb

from traj_calculation import LegRTB
from traj_calculation import InputHolder, param_traj, process_action
from traj_calculation import thetas_traj
from scene_generation import generate_scene
from simulation import PosVelOutput, SensorOutput, SimHandler


# define control function
def qdes2tau_CTC(qdes, dqdes, ddqdes, model: mujoco.MjModel, data: mujoco.MjData):
    legdofs=model.jnt_dofadr[1:]
    legqpos=model.jnt_qposadr[1:]

    # nj = 4
    nlegs = 6

    e = data.qpos[legqpos]-qdes
    de = data.qvel[legdofs]-dqdes

    kp, kd = np.diag([5000,4000,3000,5000]*nlegs), np.diag([90,300,200,200]*nlegs)
    u = np.zeros(model.nv)
    u[legdofs] = ddqdes - kp@e - kd@de
    
    Mu = np.empty(model.nv)
    mujoco.mj_mulM(model, data, Mu, u)#+c)
    tau = Mu + data.qfrc_bias
    tau = tau[legdofs]

    return tau

def qdes2tau_impedance(qdes, dqdes, ddqdes, model: mujoco.MjModel, data: mujoco.MjData):
    legdofs=model.jnt_dofadr[1:]
    legqpos=model.jnt_qposadr[1:]

    # nj = 4
    nlegs = 6    

    kp, kd = np.diag([50000,400000,300000,30000]*nlegs), np.diag([500,1000,500,500]*nlegs)
    ddqdes_full = np.zeros(model.nv)

    e = data.qpos[legqpos]-qdes
    de = data.qvel[legdofs]-dqdes
    ddqdes_full[legdofs] = ddqdes
    
    Mddq = np.empty(model.nv)
    mujoco.mj_mulM(model, data, Mddq, ddqdes_full)#+c)
    tau = (Mddq + data.qfrc_bias)[legdofs] - kp@e - kd@de

    return tau

def action2qdes_3pod(t, action, holder: InputHolder, leg_virtual: LegRTB):

    use_traj = 1
    use_memory = 1
    use_rtb_jacs = 1

    nj = 4
    nlegs = 6
    # qdes = np.array([0, 1.22, 4.01, 5.76])
    qdes = np.zeros(nj*nlegs)
    dqdes = np.zeros(nj*nlegs)
    ddqdes = np.zeros(nj*nlegs)

    if use_traj:
        if use_memory:
            holder.update(action, t)
            T_f, T_b, C_x_left, C_y_left, C_z_left, a_left, C_x_right, C_y_right, C_z_right, a_right = holder.get_output()
        else:
            # C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alfa, delta_thetas)   
            C_x_left, C_y_left, C_z_left, a_left = param_traj(True, action)
            C_x_right, C_y_right, C_z_right, a_right = param_traj(False, action)

        if use_rtb_jacs:
            qdes1_left, dqdes1_left, ddqdes1_left = thetas_traj(t, T_f, T_b, 0, C_x_left, C_y_left, C_z_left, a_left, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
            qdes1_right, dqdes1_right, ddqdes1_right = thetas_traj(t, T_f, T_b, 0, C_x_right, C_y_right, C_z_right, a_right, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
            qdes2_left, dqdes2_left, ddqdes2_left = thetas_traj(t, T_f, T_b, T_f, C_x_left, C_y_left, C_z_left, a_left, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
            qdes2_right, dqdes2_right, ddqdes2_right = thetas_traj(t, T_f, T_b, T_f, C_x_right, C_y_right, C_z_right, a_right, leg_virtual.calc_Jinv, leg_virtual.calc_Jdot)
            
            qdes1_left[2] = -qdes1_left[2]
            qdes1_right[2] = -qdes1_right[2]
            qdes2_left[2] = -qdes2_left[2]
            qdes2_right[2] = -qdes2_right[2]
        else:
            # qdes1, dqdes1, ddqdes1 = thetas_traj(t, T_f, T_b, 0, C_x, C_y, C_z, a)
            # qdes2, dqdes2, ddqdes2 = thetas_traj(t, T_f, T_b, delta_T, C_x, C_y, C_z, a)
            qdes1_left, dqdes1_left, ddqdes1_left = thetas_traj(t, T_f, T_b, 0, C_x_left, C_y_left, C_z_left, a_left) # deltaT = 0
            qdes1_right, dqdes1_right, ddqdes1_right = thetas_traj(t, T_f, T_b, 0, C_x_right, C_y_right, C_z_right, a_right) # deltaT = 0
            qdes2_left, dqdes2_left, ddqdes2_left = thetas_traj(t, T_f, T_b, T_f, C_x_left, C_y_left, C_z_left, a_left) # deltaT = Tf
            qdes2_right, dqdes2_right, ddqdes2_right = thetas_traj(t, T_f, T_b, T_f, C_x_right, C_y_right, C_z_right, a_right) # deltaT = Tf
            qdes1_left[2] = -qdes1_left[2]
            qdes1_right[2] = -qdes1_right[2]
            qdes2_left[2] = -qdes2_left[2]
            qdes2_right[2] = -qdes2_right[2]

        q0 = [0, 1.22, 4.01-2*np.pi, 5.76-2*np.pi]
        qdes1_left = qdes1_left - np.array(q0)
        qdes1_right = qdes1_right - np.array(q0)
        qdes2_left = qdes2_left - np.array(q0)
        qdes2_right = qdes2_right - np.array(q0)

    for i in range(nlegs):
        if use_traj:
            if i in (0, 4):
                qdes[0+i*4:4+i*4] = qdes1_right
                dqdes[0+i*4:4+i*4] = dqdes1_right
                ddqdes[0+i*4:4+i*4] = ddqdes1_right
            elif i == 2:
                qdes[0+i*4:4+i*4] = qdes1_left
                dqdes[0+i*4:4+i*4] = dqdes1_left
                ddqdes[0+i*4:4+i*4] = ddqdes1_left
            elif i in (1, 3):
                qdes[0+i*4:4+i*4] = qdes2_left
                dqdes[0+i*4:4+i*4] = dqdes2_left
                ddqdes[0+i*4:4+i*4] = ddqdes2_left
            else:
                qdes[0+i*4:4+i*4] = qdes2_right
                dqdes[0+i*4:4+i*4] = dqdes2_right
                ddqdes[0+i*4:4+i*4] = ddqdes2_right
    return qdes, dqdes, ddqdes


def ctrl_f(t, model, data, memory: InputHolder, leg_virtual: LegRTB, planner_func, reg_func, planner_params):
    qdes, dqdes, ddqdes = planner_func(t, planner_params, memory, leg_virtual)
    torques = reg_func(qdes, dqdes, ddqdes, model, data)
    return torques


if __name__ == '__main__':
    spec = generate_scene()
    # spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)

    spec.compile()
    model_xml = spec.to_xml()

    simtime = 20

    # prepare data logger
    simout = (SensorOutput(sensor_names=[sen.name for sen in spec.sensors],
                          sensor_dims=[3,1,1,1,1,1,1]),
                          PosVelOutput(qpos_idxs=[0,1,2],qvel_idxs=[0]))
    # simout = None
    # prepare sim params
    simh = SimHandler(model_xml, None, simlength=simtime, simout=simout)  
    memory = InputHolder(simh.timestep, process_action)
    leg_virtual = LegRTB()
    
    T_f = 2
    T_b = 0
    L = 2
    alpha = 0.26
    delta_thetas = np.array([0, 0.25,-0.2,0])

    action = (T_f, T_b, L, alpha, *delta_thetas)
    reg_func = qdes2tau_CTC
    # reg_func = qdes2tau_impedance

    # run MuJoCo simulation
    fin_dur = simh.simulate(is_slowed=1, control_func=ctrl_f, control_func_args=(memory, leg_virtual, action2qdes_3pod, reg_func, action))



    # Plotting graphs
    pos = np.asarray(simout[1].pos)
    horiz_deviation = np.asarray(simout[0].sensordata["rob_Y"][:,2])
    touches = []
    for i in range(6):
        touches.append(simout[0].sensordata[f"touch{i+1}"])

    fig, axs = plt.subplots(3, 1, figsize=(8, 6),sharex=True)  # 2 rows, 1 column

    axs[0].plot(simout[0].times, pos, label=['x','y','z'])
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Coordinate [m]")
    axs[0].set_title("Robot position")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(simout[0].times, horiz_deviation)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("sin(gamma) [-]")
    axs[1].set_title("Deviation from horizontal")
    axs[1].grid(True)

    for i in (0,5,4):
        axs[2].plot(simout[0].times, touches[i], label=f'F_{i+1}')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Force [N]")
    axs[2].set_title("Normal contact forces on leg tips")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # print out xml
    # print(model_xml)
