from ddpg_torch import Agent
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import control
import csv
#import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from scipy import linalg
import statistics

#A = np.array([[0,1,0], [0,0,1], [-6,7,0]])
#B = np.array([[0], [0], [1]])

A = np.array([[0, 1], [-2, 3]])
B = np.array([[0], [1]])

A = np.matrix(A)
B = np.matrix(B)
size = int(A.shape[0])

poles = np.array([-1,-2]) 
K = -control.place(A,B,poles)
#K=np.array([[0, -4]])
K = np.matrix(K)
Ac = A+(B*K)

Q = np.eye(size)
P = control.lyap(np.transpose(Ac), Q)
#P=np.array([[1, .25], [.25, 1]])
#Q=np.array([[.5, .25], [.25, 1.5]])
P = np.matrix(P)
Q = np.matrix(Q)

s = 0.99/(2*(np.linalg.norm(P*B*K)))
rho=0.5;
gamma=.99;

sampling_period=1e-3
sampling_freq = int(1/sampling_period)
obs_size = size+1

I=np.identity(A.shape[0])
    
def relative_threshold(x0,sp,A,Ac,s):
    te=0
    i=1
    while True:
       x=(I+np.linalg.inv(A)*(linalg.expm(A*i*sp)-I)*Ac)*x0
       x = np.matrix(x)
       norm1 = np.linalg.norm(x0-x)
       norm2 = np.linalg.norm(x)
       if norm1 >= s*norm2:
            te = i*sampling_period
            break
       i += 1
    return te

def lyapunov_trigger(x0, A, Ac, B, K, P, Q, sp,rho):
    i = 1
    te = 0
    while True:
        x=(I+np.linalg.inv(A)*(linalg.expm(A*i*sp)-I)*Ac)*x0
        x = np.matrix(x)

        xdot = A*x +B*K*x0
        
        lhs = 2*np.transpose(x)*P*xdot
        lhs = lhs.item()

        rhs = np.transpose(x)*Q*x
        rhs = rhs.item()
        
        if lhs >= -rho*rhs:
            te = i*sampling_period
            break
        i += 1
    return te

def floor_value(x, decimal):
    x = x*decimal
    x = math.floor(x)
    x = x/decimal
    return x

def write_csv(data):
    with open('data_training.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


def classical_rt_episode(x0, T, sp, A, Ac, s,gamma):
    Ts=math.floor(T/sp);
    Te=[];
    Ted=[]
    i=1;
    j=0;
    k=0
    x_k = np.copy(x0)
    while i <= Ts-j:
        x=(I+np.linalg.inv(A)*(linalg.expm(A*i*sp)-I)*Ac)*x_k
        x = np.matrix(x)
        norm1 = np.linalg.norm(x_k-x)
        norm2 = np.linalg.norm(x)
        if norm1 >= s*norm2:
            x_k=x;
            j=j+i;
            Te.append(i*sp)
            Ted.append(pow(gamma,k)*i*sp)
            k+=1
            i=1;
        i+=1;
    #te_avg=statistics.mean(Te) 
    te_dis=sum(Ted)
    return te_dis

def classical_lyp_episode(x0, T, sp, A, Ac, B, K, P, Q,rho,gamma):
    Ts=math.floor(T/sp);
    Te=[];
    Ted=[];
    i=1;
    j=0;
    k=0
    x_k = np.copy(x0)
    while i <= Ts-j:
        x=(I+np.linalg.inv(A)*(linalg.expm(A*i*sp)-I)*Ac)*x_k
        x = np.matrix(x)
        xdot = A*x +B*K*x_k
        
        lhs = 2*np.transpose(x)*P*xdot
        lhs = lhs.item()

        rhs = np.transpose(x)*Q*x
        rhs = rhs.item()
        
        if lhs >= -rho*rhs:
            x_k=x;
            j=j+i;
            Te.append(i*sp)
            Ted.append(pow(gamma,k)*i*sp)
            k+=1
            i=1;
        i+=1;
    #te_avg=statistics.mean(Te) 
    te_dis=sum(Ted)
    return te_dis


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

Tavg_rl = []
Tdis_rl=[]
Tavg_classical = []
Tdis_classical=[]
Rl_vs_classical = []
episodes = 100
score_history = []

#reward_option = 0 # Options: 0, 1

for i in range(episodes):
    
    #triggered = 0
    score = 0
    theta = (i/episodes)*2*math.pi
    x0 = np.array([math.cos(theta), math.sin(theta)])
    # x0 = np.array([random.uniform(-1,1) for i in range(size)])
    x0 = x0.reshape(-1,1)
    x0 = np.matrix(x0)
    x_init = np.copy(x0)
   
    done = False
    agent.noise.reset()
    obs = np.zeros(obs_size)

    #x_series = np.zeros(shape=(A.shape[0],1))
    ##x_series[:,[0]] = x_init
    #t_series = np.array([0])
    
    terminal_reward = 0
    less_comm = 0
    
    Te_rl=[];
    Ted_rl=[]
    k=0
    while not done:
       
        tau_e = lyapunov_trigger(x_init, A, Ac, B, K, P, Q, sampling_period,rho)
        for j in range(obs_size-1):
            obs[j] = x_init[j,0]
        obs[-1] = tau_e
        act = agent.choose_action(obs)
       
        #triggered += 1
        tau_e_rl = act*tau_e
        tau_e_rl = floor_value(tau_e_rl, sampling_freq)
        if (tau_e_rl < sampling_period):
            tau_e_rl = sampling_period
        
        #triggered += 1
        #print(tau_e_rl)
        Te_rl.append(tau_e_rl)
        Ted_rl.append(pow(gamma,k)*tau_e_rl)
        k+=1
        #print(te_rl)
        new_state=(I+np.linalg.inv(A)*(linalg.expm(A*tau_e_rl)-I)*Ac)*x_init
        new_state = new_state.reshape(-1,1)
        new_state = np.matrix(new_state)
        x_norm = np.linalg.norm(new_state)
        
        if x_norm < 0.1:
            done = 1
            T=sum(Te_rl)
            #print(T)
            te_dis_classical = classical_lyp_episode(x0, T, sampling_period, A, Ac, B, K, P, Q, rho, gamma)
            te_avg_rl = statistics.mean(Te_rl)
            te_dis_rl=sum(Ted_rl)
            #terminal_reward = 10*((te_avg_rl/te_avg_classical) - 1)

        x_init = new_state
        x_init = x_init.reshape(-1,1)
        x_init = np.matrix(x_init)
        
        new_state = np.concatenate([new_state, [[tau_e]]])
        new_state = new_state.flatten()
        #print(new_state)
        #reward = (1-reward_option)*tau_e_rl + reward_option*terminal_reward
        reward = tau_e_rl
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
    # print(i)
    # te_avg_rl = t_series[-1]/triggered
    # plt.plot(t_series,x_series[0,:])
    # plt.plot(t_series,x_series[1,:])
    # plt.xlabel('Time')
    # plt.ylabel('State')
    # plt.show()
    # if i >=50 and i % 50 == 0:
    #     agent.save_models()

    # print("Input to lyapunov based sim", x0, t_series[-1])
    # x_classical,t_classical,te_avg_classical = classical_episode(x0, t_series[-1])

    Tdis_rl.append(te_dis_rl)
    Tdis_classical.append(te_dis_classical)
    rl_classical = te_dis_rl-te_dis_classical
    Rl_vs_classical.append(rl_classical)
    score_history.append(score)
    print(i, " Score: %.2f" % score, " RL IET = %.3f" % te_dis_rl, " Classic IET = %.3f" % te_dis_classical, " RL-Classical = %.3f" % rl_classical)
    write_csv([i, score, x0, te_dis_rl, te_dis_classical, rl_classical])
    


plt.figure(1)
plt.plot(range(episodes),Tdis_rl,'g:',label='RL')
plt.plot(range(episodes),Tdis_classical,'b-',label='Classical')
plt.ylabel('IET')
plt.xlabel('Episode')
plt.legend(loc='best')
name = "plots/IET"
plt.savefig(name)
plt.close()

plt.figure(2)
plt.plot(range(episodes), Rl_vs_classical)
plt.ylabel('IET(RL) - IET(Classical)')
plt.xlabel('Episode')
name = "plots/RL_vs_Classical"
plt.savefig(name)
plt.close()
        
