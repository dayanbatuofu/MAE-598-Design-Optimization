import logging
import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import scipy.io

torch.manual_seed(10)

logger = logging.getLogger(__name__)

# environment parameters
dx_ini = -1  # initial position at x axis
vx_ini = 0  # initial velocity at x axis
dy_ini = 3  # initial position at y axis
vy_ini = 0  # initial velocity at y axis
theta = math.pi / 3  # initial angle deviation of rocket axis from y axis

FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL = 0.18  # thrust constant
DRAG_ACCEL = 0.005  # drag constant

class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    def forward(self, state, action):
        """
        action[0] = controller
        action[1] = theta

        state[0] = x
        state[1] = x_dot
        state[2] = y
        state[3] = y_dot
        state[4] = theta
        """
        # Apply gravity
        # Note: Here gravity is used to change velocity which is the second element of the state vector
        # Normally, we would do x[1] = x[1] + gravity * delta_time
        # but this is not allowed in PyTorch since it overwrites one variable (x[1]) that is part of the computational graph to be differentiated.
        # Therefore, I define a tensor dx = [0., gravity * delta_time], and do x = x + dx. This is allowed...
        delta_state_gravity = torch.tensor([0., 0., 0., -GRAVITY_ACCEL * FRAME_TIME, 0.])

        # Thrust
        # Note: Same reason as above. Need a 5-by-1 tensor.
        N = len(state)
        state_tensor = torch.zeros((N, 5))
        state_tensor[:, 1] = -torch.sin(state[:, 4])
        state_tensor[:, 3] = torch.cos(state[:, 4])
        delta_state = BOOST_ACCEL * FRAME_TIME * torch.mul(state_tensor, action[:, 0].reshape(-1, 1))

        # Theta
        delta_state_theta = FRAME_TIME * torch.mul(torch.tensor([0., 0., 0., 0, -1.]), action[:, 1].reshape(-1, 1))

        state = state + delta_state + delta_state_gravity + delta_state_theta

        # Update state
        step_mat = torch.tensor([[1., FRAME_TIME, 0., 0., 0.],
                                 [0., 1., 0., 0., 0.],
                                 [0., 0., 1., FRAME_TIME, 0.],
                                 [0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 1.]])

        state = torch.matmul(step_mat, state.T)

        return state.T

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden:
        """
        super(Controller, self).__init__()
        # little linear network with ReLU for embeddings
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid())

    def forward(self, state):
        action = self.network(state)
        return action


# the simulator that rolls out x(1), x(2), ..., x(T)
# Note:
# 0. Need to change "initialize_state" to optimize the controller over a distribution of initial states
# 1. self.action_trajectory and self.state_trajectory stores the action and state trajectories along time

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T, N):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.N = N
        self.theta_trajectory = torch.empty((1, 0))
        self.u_trajectory = torch.empty((1, 0))

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller(state)
            state = self.dynamics(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        # state = [1., 0.]  # TODO: need batch of initial states
        state = torch.rand((N, 5))
        state[:, 1] = 0  # vx = 0
        state[:, 3] = 0  # vy = 0
        # TODO: need batch of initial states
        return torch.tensor(state, requires_grad=False).float()

    def error(self, state):
        return torch.mean(state ** 2)

class Optimize:

    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)
        self.loss_list = []

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            self.loss_list.append(loss)
            print('[%d] loss: %.3f' % (epoch + 1, loss))
        self.visualize()

    def visualize(self):
        data = np.array([[self.simulation.state_trajectory[i][N].detach().numpy() for i in range(self.simulation.T)] for N in range(self.simulation.N)])
        for i in range(self.simulation.N):
            dx = data[i, :, 0]
            dy = data[i, :, 2]
            vx = data[i, :, 1]
            vy = data[i, :, 3]
            theta = data[i, :, 4]
            fig1, axs0 = plt.subplots()
            fig2, axs1 = plt.subplots()
            fig3, axs2 = plt.subplots()
            fig4, axs3 = plt.subplots()
            axs0.plot(dx, dy)
            axs0.set_title('Position Changeable for Rocket Landing')
            axs0.set_xlabel('Rocket X Position(m)')
            axs0.set_ylabel('Rocket Y Position(m)')

            axs1.plot(list(range(self.simulation.T)), vx)
            axs1.set_title('Velocity X Changeable for Rocket Landing')
            axs1.set_xlabel('Time Step')
            axs1.set_ylabel('Rocket X Velocity(m/s)')

            axs2.plot(list(range(self.simulation.T)), vy)
            axs2.set_title('Velocity Y Changeable for Rocket Landing')
            axs2.set_xlabel('Time Step')
            axs2.set_ylabel('Rocket Y Velocity(m/s)')

            axs3.plot(list(range(self.simulation.T)), theta)
            axs3.set_title('Theta Changeable for Rocket Landing')
            axs3.set_xlabel('Time Step')
            axs3.set_ylabel('Rocket Theta(rad)')
        plt.show()

N = 10  # number of samples  for random case
T = 100  # number of time steps
dim_input = 5  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T, N)  # define simulation, N is the number of samples to be considered
o = Optimize(s)  # define optimizer
o.train(80)  # solve the optimization problem

plt.title('Objective Function Convergence Curve')
plt.xlabel('Training Iteration')
plt.ylabel('Error')
plt.plot(list(range(80)), o.loss_list)
plt.show()

# store data
# data = np.array([[s.state_trajectory[i][j].detach().numpy() for i in range(s.T)] for j in range(s.N)])
# idx = 6
#
# time = np.linspace(0, 3, 100)
#
# save_data = {'X': data[idx, :, :].T,
#              't': time}
# save_path = 'data_rocket_landing_test.mat'
# scipy.io.savemat(save_path, save_data)
