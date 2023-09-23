from ddpg_torch import Agent
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import control
import csv
import copy

# A = np.array([[1,0,0], [0,2,0], [0,0,3]])
# B = np.array([[1], [1], [1]])
# Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# A = np.array([[0, 1], [-2, 3]])
# B = np.array([[0], [1]])
A = np.diag([0.1,0.2,0.3,0.4])
size = int(A.shape[0])
B = np.ones(size)

A = np.matrix(A)
B = B.reshape(-1,1)
B = np.matrix(B)

poles = np.array([-0.5+0.2j,-0.5-0.2j,-1+0.5j,-1-0.5j])
# poles = np.array([-2, -3]) 
K = -control.place(A,B,poles)
K = np.matrix(K)
Ac = A+(B*K)
Q = np.eye(size)
P = control.lyap(np.transpose(Ac), Q)
P = np.matrix(P)

sampling_period=1e-3
sampling_freq = int(1/sampling_period)
obs_size = size+1

def sigma(P, B, K):
    tmp = P*B
    s = 0.99/(2*(np.linalg.norm(tmp*K)))
    return s

def relative_threshold(x0, K, sampling_period):
    s = sigma(P, B, K)
    # print(s)
    i = 0
    te = 0
    xinit = np.copy(x0)
    u = K*xinit
    u = u.item()
    while True:
        x,_ = simulate(A,B,u,xinit,sampling_period,sampling_period)
        x = x[:,-1]
        x = x.reshape(-1,1)
        x = np.matrix(x)
        norm1 = np.linalg.norm(x0-x)
        norm2 = np.linalg.norm(x)
        # print(x0,x, norm1, s*norm2)
        if norm1 >= s*norm2:
            te = i*sampling_period
            break
        i += 1
        xinit = np.copy(x)
    return te


def lyapunov_trigger(x0, K, A, B, P, sampling_period):
    i = 0
    te = 0
    xinit = np.copy(x0)
    u = K*x0
    u = u.item()
    while True:
        x,_ = simulate(A,B,xinit,u,sampling_period,sampling_period)
        x = x[:,-1]
        x = x.reshape(-1,1)
        x = np.matrix(x)

        xdot = A*x +B*u
        
        lhs = 2*np.transpose(x)*P*xdot
        lhs = lhs.item()

        rhs = np.transpose(x)*P*x
        rhs = rhs.item()
        rhs = -0.1*(rhs)
        # print(lhs, rhs)
        te = i*sampling_period
        if lhs >= 0: #rhs:
            if i == 0:
                te = sampling_period
                print("IET < Sampling Time!!!")
            break
        # print("te", te)
        i += 1  
        xinit = np.copy(x)
    return te

def simulate(A,B,initial_state, u, time,sampling_period):
    time_steps = int(time/sampling_period)
    I=np.identity(A.shape[0]) # this is an identity matrix
    Ad=np.linalg.inv(I-sampling_period*A)
    Bd=Ad*sampling_period*B
    Xd=np.zeros(shape=(A.shape[0],time_steps+1))
    t_series = np.zeros(time_steps+1)
    # print(time)
    for i in range(0,time_steps):
        if i==0:
            Xd[:,[i]]=initial_state
            x=Ad*initial_state+Bd*u
        else:
            Xd[:,[i]]=x
            x=Ad*x+Bd*u
        t_series[i] = i*sampling_period
    Xd[:,[-1]]=x
    t_series[-1] = time
    return Xd,t_series

def floor_value(x, decimal):
    x = x*decimal
    x = math.floor(x)
    x = x/decimal
    return x

def write_csv(data):
    with open('data_training.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


def classical_episode(x0, time):
    x_init = np.copy(x0)
    done = 0
    x_series = np.zeros(shape=(A.shape[0],1))
    x_series[:,[0]] = x_init
    t_series = np.array([0])
    counter = 0
    triggered = 0
    while not done:
        tau_e = lyapunov_trigger(x_init, K, A, B, P, sampling_period)
        # print("Inside fn1", tau_e)
        if counter*sampling_period +tau_e > time:
            tau_e = (time - counter*sampling_period)
            done = 1
            # print("Inside fn2", tau_e, time, counter*sampling_period)
            # print("x = ", x_series[:,-1])
        tau_e = floor_value(tau_e, sampling_freq)
        if tau_e < sampling_period*0.1:
            break
        # print("Inside fn3", tau_e)
        u = K*x_init
        u = u.item()
        x,t = simulate(A, B, x_init, u, tau_e, sampling_period)
        triggered += 1
        new_timesteps = x.shape[1] - 1
        # print(new_timesteps, x.shape)
        # print("x k t u",x_init, K, t_series[-1], u)
        if new_timesteps == 1:
            x_series = np.concatenate((x_series, x[:,-1:]), axis = 1)
            t_series = np.concatenate((t_series, [t_series[-1] + t[-1]]))
        else:
            x_series = np.concatenate((x_series, x[:,1:]), axis = 1)
            t_series = np.concatenate((t_series, t_series[-1] + t[1:]))
        counter += new_timesteps
        x_init = np.copy(x[:,-1])
        x_init = x_init.reshape(-1,1)
        x_init = np.matrix(x_init)
    te_avg = t_series[-1]/triggered
    return x_series,t_series, te_avg


# x0 = np.array([random.uniform(-1,1) for i in range(size)])
# x0 = x0.reshape(-1,1)
# x0 = np.matrix(x0)
# x,t,avg_iet = classical_episode(x0, 10)
# print("Avg IET = ", avg_iet)
# plt.plot(t,x[0,:])
# plt.plot(t,x[1,:])
# plt.xlabel('Time')
# plt.ylabel('State')
# plt.show()

agent = Agent(alpha=0.000025, beta=0.003, input_dims=[obs_size], tau=0.001, batch_size=16, layer1_size=16, layer2_size=32, n_actions=1)
agent.load_models()

btavg_rl = []
btavg_classical = []
brl_vs_classical = []
batches = 10
episodes = 5

for batch in range(batches):
    tavg_rl = []
    tavg_classical = []
    rl_vs_classical = []
    for episode in range(episodes):
        triggered = 0
        score = 0
        x0 = np.array([random.uniform(-1,1) for i in range(size)])
        x0 = x0.reshape(-1,1)
        x0 = np.matrix(x0)
        x_init = np.copy(x0)
        u_init = np.array([K*x_init])
        done = False
        agent.noise.reset()
        obs = np.zeros(obs_size)

        x_series = np.zeros(shape=(A.shape[0],1))
        x_series[:,[0]] = x_init
        t_series = np.array([0])

        while not done:
            tau_e = lyapunov_trigger(x_init, K, A, B, P, sampling_period)
            for j in range(obs_size-1):
                obs[j] = x_init[j,0]
            obs[-1] = tau_e
            act = agent.choose_action(obs)
            # print(t_series[-1], tau_e, obs)
            triggered += 1
            tau_e_rl = act*tau_e
            tau_e_rl = floor_value(tau_e_rl, sampling_freq)
            if (tau_e_rl < sampling_period):
                tau_e_rl = sampling_period
            u = K*x_init
            u = u.item()
            x,t = simulate(A, B, x_init, u, tau_e_rl, sampling_period)
            triggered += 1

            new_timesteps = x.shape[1] - 1
            if new_timesteps == 1:
                x_series = np.concatenate((x_series, x[:,-1:]), axis = 1)
                t_series = np.concatenate((t_series, [t_series[-1] + t[-1]]))
            else:
                x_series = np.concatenate((x_series, x[:,1:]), axis = 1)
                t_series = np.concatenate((t_series, t_series[-1] + t[1:]))

            x_end = np.copy(x[:,-1])
            x_norm = np.linalg.norm(x_end)
            # print(x_norm, t_series[-1])
            if x_norm < 0.1:
                done = 1

            x_init = np.copy(x[:,-1])
            x_init = x_init.reshape(-1,1)
            x_init = np.matrix(x_init)

            new_state = np.concatenate((x_end, [tau_e]))
            reward = tau_e_rl
            score += reward

        te_avg_rl = t_series[-1]/triggered
        x_classical,t_classical,te_avg_classical = classical_episode(x0, t_series[-1])
        tavg_rl.append(te_avg_rl)
        tavg_classical.append(te_avg_classical)
        rl_by_classical = te_avg_rl/te_avg_classical
        rl_vs_classical.append(rl_by_classical)

    btavg_rl.append(sum(tavg_rl)/(len(tavg_rl)))
    btavg_classical.append(sum(tavg_classical)/(len(tavg_classical)))
    brl_vs_classical.append(btavg_rl[-1]/btavg_classical[-1])
    print(batch, " RL IET = %.3f" % btavg_rl[-1], " Classic IET = %.3f" % btavg_classical[-1], " RL/Classical = %.3f" % brl_vs_classical[-1])


plt.figure(1)
plt.plot(range(batches),btavg_rl,'g:',label='RL')
plt.plot(range(batches),btavg_classical,'b-',label='Classical')
plt.ylabel('IET')
plt.xlabel('batch')
plt.legend(loc='best')
name = "plots/IET"
plt.savefig(name)
plt.close()

plt.figure(2)
plt.plot(range(batches), brl_vs_classical)
plt.ylabel('IET(RL) / IET(Classical)')
plt.xlabel('batch')
name = "plots/RL_vs_Classical"
plt.savefig(name)
plt.close()
        
