import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control
import csv


class lti_system:

    def __init__(self, A, B, K):
        self.A = np.copy(A)
        self.B = np.copy(B)
        self.K = np.copy(K)

        self.A = np.matrix(self.A)
        self.B = np.matrix(self.B)
        self.size = int(self.A.shape[0])

        self.Ac = self.A+(self.B*self.K)

        self.Q = np.eye(self.size)
        self.Q = np.matrix(self.Q)
        self.P = control.lyap(np.transpose(self.Ac), self.Q)
        self.P = np.matrix(self.P)
        
        self.iet_upper_limit = 1
        self.rho = 0.5


    def model(self, t, x, _, u):
        x_local = np.copy(x)
        x_local = x_local.reshape(-1,1)
        xdot = self.A@x_local + self.B*u
        return np.array(np.transpose(xdot))[0]


    def controlled_model(self, t, x):
        u = (self.K@x).item()
        x_local = np.copy(x)
        x_local = x_local.reshape(-1,1)
        xdot = self.A@x_local + self.B*u
        return np.array(np.transpose(xdot))[0]


    def step(self, x_init, t_start, t_end, u):
        x0 = np.copy(x_init)
        x0 = np.array(x0)
        sol = solve_ivp(self.model, [t_start, t_end], x0, args = (u, u), max_step = 0.1)
        return sol



def lyap_condition(t, x, system, u):
    x_vector = np.copy(x)
    x_vector = np.array(x_vector)
    x_vector = x_vector.reshape(-1,1)

    xdot = np.matmul(system.A,x_vector) + system.B*u
    
    lhs = np.matmul(2*system.P, xdot)
    lhs = np.matmul(np.transpose(x_vector), lhs)
    lhs = lhs.item()

    rhs = np.matmul(system.P, x_vector)
    rhs = np.matmul(np.transpose(x_vector), rhs)
    rhs = rhs.item()
    rhs = -system.rho*(rhs)

    return (lhs - rhs)

lyap_condition.terminal = True
lyap_condition.direction = 1


def get_lyap_iet(system, x_init):
    x0 = np.copy(x_init)
    x0 = np.array(x0)
    u = (system.K@x0).item()
    sol = solve_ivp(system.model, [0, system.iet_upper_limit], x0, events = (lyap_condition), args = (system, u), max_step = 0.1)
    return sol.t[-1]

# A = np.array([[0, 1], [-2, 3]])
# B = np.array([[0], [1]])
# K = np.array([0, -6])

A = np.array([[1,0,0,0], [0,2,0,0], [0,0,-3,0], [0,0,0,-4]])
B = np.array([[1],[0.5],[1],[0.5]])
poles = np.array([-1,-1.5,-2,-2.5])
K = -control.place(A,B,poles)[0]
# K = np.array([0,0,0,0])

sys1 = lti_system(A,B,K)

def lyap_etc_episode(system, x_init, t_end, termination = 'time'):
    x0 = np.copy(x_init)
    x0 = np.array(x0)
    done = 0
    t = 0
    t_end_tmp = 0
    while not done:
        u = (system.K@x0).item()
        iet = get_lyap_iet(system, x0)

        if termination == 'time':
            if t+iet > t_end:
                done = 1
                t_end_tmp = t_end
            else:
                t_end_tmp = t+iet
        elif termination == 'state_norm':
            t_end_tmp = t+iet
            if np.linalg.norm(x0) < 0.01:
                done = 1

        tmp_sol = system.step(x0, t, t_end_tmp, u)

        if t == 0:
            x_return = np.copy(tmp_sol.y)
            t_return = np.copy(tmp_sol.t)
            tevent_return = np.array([iet])
            u_return = np.ones(len(tmp_sol.t))*u
        else:
            u_return = np.concatenate((u_return, np.ones(len(tmp_sol.t))*u))
            x_return = np.concatenate((x_return, tmp_sol.y), axis = 1)
            t_return = np.concatenate((t_return, tmp_sol.t))
            tevent_return = np.concatenate((tevent_return, np.array([tmp_sol.t[-1]])))

        t = tmp_sol.t[-1]
        x0 = tmp_sol.y.T[-1]

    iet_series = np.zeros(len(tevent_return) - 1)
    for i in range(len(tevent_return)-1):
        iet_series[i] = tevent_return[i+1] - tevent_return[i]

    return t_return, x_return, tevent_return, iet_series, u_return

# x_init = np.array([1,2,3,4])
# # x_init = np.array([1,2])
x_init = np.array([0.82904106, 0.29785153, 0.43644708, 0.1830009])
t,x,te, iet, u_ser = lyap_etc_episode(sys1, x_init, 1.4)

def rl_step(agent, obs, system, x_init, t_start):
    x0 = np.copy(x_init)
    x0 = np.array(x0)

    action = agent.choose_action(obs)
    action = action.item()
    if action < 0.001:
        action = 0.001
    lyap_iet = get_lyap_iet(system, x0)
    iet = action*lyap_iet
    u = (system.K@x0).item()
    sol = system.step(x0, t_start, t_start+iet, u)
    new_state = np.concatenate((np.array(sol.y.T[-1]), np.array([iet])))
    reward = iet

    return sol, action, reward, new_state


def write_csv(data):
    with open('data_training.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

print(iet)
plt.plot(t, x.T)
plt.plot(t, u_ser)
plt.xlabel('t')
plt.legend(['x1', 'x2', 'x3', 'x4', 'u'], shadow=True)
plt.show()