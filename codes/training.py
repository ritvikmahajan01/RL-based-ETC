import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control
import functions as F
import random
from ddpg_torch import Agent
import math
import os
import csv


# A = np.array([[0, 1], [-2, 3]])
# B = np.array([[0], [1]])
# K = np.array([0, -6])
# poles = np.array([-1,-2]) 

# A = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])
# B = np.array([[1],[0.5],[1],[0.5]])
# poles = np.array([-1,-1.5,-2,-2.5])
# K = -control.place(A,B,poles)[0]

A = np.array([[1,0,0,0], [0,2,0,0], [0,0,-3,0], [0,0,0,-4]])
B = np.array([[1],[0.5],[1],[0.5]])
poles = np.array([-1,-1.5,-2,-2.5])
K = -control.place(A,B,poles)[0]

sys1 = F.lti_system(A,B,K)


episodes = 1000
gamma = 0.99
obs_size = sys1.size + 1
agent = Agent(alpha=0.000025, beta=0.003, input_dims=[obs_size], tau=0.001, batch_size=32, layer1_size=100, layer2_size=200, n_actions=1)

initial_conditions = np.random.uniform(0,1,(episodes, sys1.size))

rl_vs_lyap_avg = np.zeros(episodes)
rl_dis_sum = np.zeros(episodes)
lyap_dis_sum = np.zeros(episodes)
avg_iet_rl_ser = []
avg_iet_lyap_ser = []

for i in range(episodes):
    initial_conditions[i] = initial_conditions[i]/np.linalg.norm(initial_conditions[i])

    x0 = np.copy(initial_conditions[i])
    done = False
    agent.noise.reset()
    obs = np.zeros(obs_size)
    t = 0

    score = 0
    print(x0)
    while not done:
        lyap_iet = F.get_lyap_iet(sys1, x0)
        u = (sys1.K@x0).item()
        obs = np.concatenate((x0, np.array([lyap_iet])))
        tmp_sol, action, reward, new_state = F.rl_step(agent, obs, sys1, x0, t)

        if t == 0:
            x_ser = np.copy(tmp_sol.y)
            t_ser = np.copy(tmp_sol.t)
            tevent_ser = np.array([tmp_sol.t[-1]])
            u_ser = np.ones(len(tmp_sol.t))*u
        else:
            u_ser = np.concatenate((u_ser, np.ones(len(tmp_sol.t))*u))
            x_ser = np.concatenate((x_ser, tmp_sol.y), axis = 1)
            t_ser = np.concatenate((t_ser, tmp_sol.t))
            tevent_ser = np.concatenate((tevent_ser, np.array([tmp_sol.t[-1]])))

        t = tmp_sol.t[-1]
        x0 = tmp_sol.y.T[-1]

        if np.linalg.norm(x0) < 0.01:
            done = True
        
        agent.remember(obs, action, reward, new_state, int(done))
        agent.learn()
        score += reward

    if i >= 50 and i % 50 == 0:
        agent.save_models()
    
    iet_series_rl = np.zeros(len(tevent_ser) - 1)

    for event_i in range(len(tevent_ser)-1):
        iet_series_rl[event_i] = tevent_ser[event_i+1] - tevent_ser[event_i]

    iet_dis_sum_rl = 0
    for iet_i in range(len(iet_series_rl)):
        iet_dis_sum_rl += pow(gamma, iet_i)*iet_series_rl[iet_i]

    avg_iet_rl = np.mean(iet_series_rl)
    avg_iet_rl_ser.append(avg_iet_rl)

    print(initial_conditions[i])
    t_lyap, x_lyap, tevent_, iet_series_lyap, u_lyap = F.lyap_etc_episode(sys1, initial_conditions[i], t)
    
    iet_dis_sum_lyap = 0
    for iet_i in range(len(iet_series_lyap)):
        iet_dis_sum_lyap += pow(gamma, iet_i)*iet_series_lyap[iet_i]

    avg_iet_lyap = np.mean(iet_series_lyap)
    avg_iet_lyap_ser.append(avg_iet_lyap)

    rl_vs_lyap_avg[i] = avg_iet_rl - avg_iet_lyap
    rl_dis_sum[i] = iet_dis_sum_rl
    lyap_dis_sum[i] = iet_dis_sum_lyap

    # print(x_ser.T[0], x_ser.T[-1], t_ser[0], t_ser[-1], x_lyap.T[0], x_lyap.T[-1], t_lyap[0], t_lyap[-1])

    # plt.figure(1)
    # plt.plot(t_ser, x_ser.T)
    # plt.plot(t_ser, u_ser)
    # plt.xlabel('t')
    # plt.legend(['x1', 'x2', 'u'], shadow=True)
    # plt.title("RL")

    # plt.figure(2)
    # plt.plot(t_lyap, x_lyap.T)
    # plt.plot(t_lyap, u_lyap)
    # plt.xlabel('t')
    # plt.legend(['x1', 'x2', 'u'], shadow=True)
    # plt.title("Lyapunov")
    # plt.show()


    print(i, "RL Avg IET = %.3f" % avg_iet_rl, " Lyap Avg IET = %.3f" % avg_iet_lyap)
    F.write_csv([i, initial_conditions[i], t, avg_iet_rl, len(tevent_ser), avg_iet_lyap])

plt.figure(1)
plt.plot(range(episodes), rl_vs_lyap_avg)
plt.xlabel("Episodes")
plt.ylabel("Difference in Avg IET")

plt.figure(2)
plt.plot(range(episodes), avg_iet_rl_ser)
plt.plot(range(episodes), avg_iet_lyap_ser)
plt.legend(["RL Avg IET", "Lyap Avg IET"])
plt.xlabel("Episodes")
plt.ylabel("Time (s)")
# plt.plot(range(episodes), rl_dis_sum)
# plt.plot(range(episodes), lyap_dis_sum)
# plt.legend(["RL_Dis_Sum", "Lyap_Dis_Sum"])
# plt.xlabel("Episodes")
# plt.ylabel("Discounted Sum of IET")

plt.show()





