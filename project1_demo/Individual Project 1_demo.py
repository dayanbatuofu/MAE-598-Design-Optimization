import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import random
import scipy.io
import math

torch.manual_seed(0)

logger = logging.getLogger(__name__)

# environment parameters
dx_ini = -0.3
vx_ini = 0
dy_ini = 6
vy_ini = 0
theta = math.pi / 3

GRAVITY_ACCEL = 0.12  # gravity constant
FRAME_TIME = 0.1  # time interval
BOOST_ACCEL = 3.  # thrust constant

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
        if state[0] <= 0:
            delta_state = BOOST_ACCEL * FRAME_TIME * torch.tensor([0., torch.sin(state[4]), 0., -torch.cos(state[4]), 0]) * action[0]
        else:
            delta_state = BOOST_ACCEL * FRAME_TIME * torch.tensor([0., -torch.sin(state[4]), 0., torch.cos(state[4]), 0]) * action[0]

        delta_state_theta = FRAME_TIME * torch.tensor([0., 0., 0., 0, -1.]) * action[1]

        # Update velocity
        state = state + delta_state + delta_state_gravity + delta_state_theta

        # Update state
        # Note: Same as above. Use operators on matrices/tensors as much as possible. Do not use element-wise operators as they are considered inplace.
        step_mat = torch.tensor([[1., FRAME_TIME, 0., 0., 0.],
                                 [0., 1., 0., 0., 0.],
                                 [0., 0., 1., FRAME_TIME, 0.],
                                 [0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 1.]])

        state = torch.matmul(step_mat, state)

        return state

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
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid())

    def forward(self, state):
        action = self.network(state)
        return action

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
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
        state = [dx_ini, vx_ini, dy_ini, vy_ini, theta]  # TODO: need batch of initial states
        return torch.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2 + state[4]**2

class Optimize:

    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.001)
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
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        dx = data[:, 0]
        dy = data[:, 2]
        vx = data[:, 1]
        vy = data[:, 3]
        theta = data[:, 4]
        plt.plot(dx, dy)
        plt.title('Position Changeable for Rocket Landing')
        plt.xlabel('Rocket X Position(m)')
        plt.ylabel('Rocket Y Position(m)')
        plt.show()

        plt.plot(list(range(100)), vx)
        plt.title('Velocity X Changeable for Rocket Landing')
        plt.xlabel('Time Step')
        plt.ylabel('Rocket Y Velocity(m)')
        plt.show()

        plt.plot(list(range(100)), vy)
        plt.title('Velocity Y Changeable for Rocket Landing')
        plt.xlabel('Time Step')
        plt.ylabel('Rocket Y Velocity(m)')
        plt.show()

        plt.plot(list(range(100)), theta)
        plt.title('Theta Changeable for Rocket Landing')
        plt.xlabel('Time Step')
        plt.ylabel('Rocket Theta(rad)')
        plt.show()

T = 100  # time steps
dim_input = 5  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(120)

plt.title('Objective Function Convergence Curve')
plt.xlabel('Training Iteration')
plt.ylabel('Error')
plt.plot(list(range(100)), o.loss_list)
plt.show()

# store data
# data = np.array([s.state_trajectory[i].detach().numpy() for i in range(s.T)])
# time = np.linspace(0, 3, 100)
#
# save_data = {'X': data.T,
#              't': time}
# save_path = 'data_rocket_landing.mat'
# scipy.io.savemat(save_path, save_data)
