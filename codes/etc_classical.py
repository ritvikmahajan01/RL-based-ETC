import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


# Initial Conditions
x0 = [0.1, 1]
u0 = 1
u = u0

# Control Gain
k1 = 0
k2 = -6

# System Dynamics
def model(x, t, u):
    x1 = x[0]
    x2 = x[1]
    dx1dt = x2
    dx2dt = -2*x1 + 3*x2 + u
    dxdt = np.array([dx1dt,dx2dt], dtype=np.float32)
    return dxdt


sample_freq = 100  # sample_time = 1/100 = 0.01s

round_to = int(math.log10(sample_freq)) # All time instants should be converted to the same step size

t = np.arange(0, 10.0, 1/sample_freq) # Array containing timesteps from 0 to end time
t = [round(ti, round_to) for ti in t]
total_steps = int(t[-1]*sample_freq)

# Initializing variables
x1 = [x0[0]]
x2 = [x0[1]]
u_ser = [u0]
u_ser[0] = u0

controlled_at = [1]
communication_saved = 0
for i in range(1,total_steps+1):
    # span for next time step

    tspan = [t[i-1],t[i]]

    # solve for next step
    x = odeint(model,x0,tspan,args=(u,))

    # store solution for plotting
    x1.append(x[1][0])
    x2.append(x[1][1])

    # if any(t[i] == c for c in t_trigger):
    if abs(x1[i]) > 0.1:
        u = k1*x1[i] + k2*x2[i]
        # print("Triggered!!! at t = ", t[i])
        controlled_at.append(1)
        communication_saved += 1

    # Else, u(t_k) = u(t_k-t_k-1)
    else:
        u = u0
        controlled_at.append(0)

    # next initial condition
    x0 = x[1]
    u0 = u
    u_ser.append(u)

communication_saved = (communication_saved/total_steps)*100.0
print("%.2f%%  less communication" % (100-communication_saved))
plt.plot(t,controlled_at,'k-',label='communication instants')
plt.plot(t,u_ser,'g:',label='u(t)')
plt.plot(t,x1,'b-',label='x1(t)')
plt.plot(t,x2,'r--',label='x2(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

# fig, ax1 = plt.subplots()

# ax2 = ax1.twinx()
# ax1.plot(t,u_ser,'g:',label='u(t)')
# ax1.plot(t,x1,'b-',label='x1(t)')
# ax1.plot(t,x2,'r--',label='x2(t)')
# ax2.plot(t,controlled_at,'k-',label='communication instants')

# ax1.set_xlabel('time')
# ax1.set_ylabel('Values', color='g')
# ax2.set_ylabel('Communication', color='b')

# plt.show()