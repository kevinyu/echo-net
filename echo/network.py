import numpy as np
from scipy import sparse


class ReservoirState(object):
    """Representation of reservoir state"""
    def __init__(self, x):
        self.x = x

    @property
    def r(self):
        return np.tanh(self.x)


class Reservoir(object):
    """Recurrent reservoir network

    Instantiates N leaky integrator neurons. The reservoir is defined by
    three sets of weights:
        * recurrent weights (N x N), randomly set and normalized by N * sparsity
        * input weights (N_in x N), randomly set
        * feedback weights (N_out x N), randomly set

    Params:
    N (int): number of nodes
    N_in (int, default=1): Number of input channels
    N_out (int, default=1): Number of output channels
    sparsity (float, default=0.1): probability of a recurrent weight to be nonzero
    dt (float): Size of timestep to simulate (should be << time constant)
    t_const (float, default=10): Time constant of network dynamics (tau)
        Can be either a number, or a column vector of dimension Nx1
    """

    def __init__(self,
            N,
            N_in=1,
            N_out=1,
            N_feedback=None,
            sparsity=0.1,
            dt=0.1,
            t_const=10,
            g=1.5,
            feedback_on=0,
            noise=0.0):
        self.N = N
        self.N_in = N_in
        self.N_out = N_out
        self.sparsity = sparsity
        self.dt = dt
        self.t_const = t_const
        self.noise = noise
        self.g = g
        self.feedback_on = feedback_on
        self.N_feedback = N_feedback or N_out

        self.W_input = np.random.uniform(-1, 1, size=(N_in, N))  # gaussian, unit variance
        self.W_feedback = np.random.uniform(-1, 1, size=(self.N_feedback, N))  # uniform -1 1
        # self.W_feedback = np.random.normal(scale=0.1, size=(self.N_feedback, N))

        def randnorm(length):
            return np.random.normal(scale=g / np.sqrt(sparsity * N), size=length)

        # self.W_recurrent = sparse.random(N, N, density=sparsity, format="csr")  # 
        self.W_recurrent = sparse.random(N, N, density=sparsity, format="csr", data_rvs=randnorm)  # 
        # _temp = self.W_recurrent.toarray()
        # _temp[np.where(_temp != 0)] = 1.0 - 2.0 * _temp[np.where(_temp != 0)]
        # self.W_recurrent = sparse.csr_matrix(_temp)
        # self.W_recurrent = np.sqrt(sparsity * N) * self.W_recurrent / g

    def step(self, state, input_data=None, feedback_data=None):
        """Run one time step of the network dynamics using leaky integrator
        
        Params:
        state (ReservoirState obj): State representing neuron states and firing rates
        input_data (np.ndarray, default=None): Input data array of shape (N_in, 1)
        feedback_data (np.ndarray, default=None): Input data array of shape (N_out, 1)

        Returns:
        state (ReservoirState obj): New state after running one step of leaky integrator
        """
        if input_data is None:
            input_data = np.zeros((self.N_in, 1))
        if feedback_data is None:
            feedback_data = np.zeros((self.N_feedback, 1))

        dx = (self.dt / self.t_const) * (
                - state.x +
                self.W_recurrent.T @ state.r +
                self.W_input.T @ input_data +
                self.feedback_on * (self.W_feedback.T @ feedback_data) +
                (np.random.normal(scale=self.noise, size=(state.x.shape)) if self.noise else 0)
        )

        return ReservoirState(state.x + dx)


class ReservoirNetwork(object):
    """Chaotic net system invovling direct feedback connections

    Params:
    reservoir (Reservoir obj)
    feedback_reservoir (Reservoir obj, default=None)
    """

    def __init__(self, res=None, **res_kwargs):
        """
        res (Reservoir obj): Reservoir
        """
        if res:
            self.res = res
        else:
            self.res = ReservoirNetwork(**res_kwargs)
        
        # this network will be learned
        self.W_readout = np.random.uniform(-1, 1, size=(res.N, res.N_out))
        self.W_readout = self.W_readout / np.sqrt(res.N)

    def readout(self, state):
        """Generate output from reservoir state"""
        return self.W_readout.T @ state.r  # TODO: add noise here?

    def run(self, input_data=None, steps=None, state=None):
        """run network driven by input data

        if input_data is a number, run network for that many timesteps with no input
        """
        if state is None:
            # initialize random state
            state = ReservoirState(np.random.normal(size=(self.res.N, 1)))

        if input_data is None and not steps:
            raise Exception("Need either input_data array or number of steps to run without input""")
        elif input_data is None:
            input_data = np.zeros((self.res.N_in, steps))

        samples = input_data.shape[1]

        outputs = np.zeros((self.res.N_out, samples))
        hidden_state_history = np.zeros((self.res.N, samples))
        visible_state_history = np.zeros((self.res.N, samples))

        output = None
        for ind in range(samples):
            state = self.res.step(
                    state,
                    input_data=input_data[:, ind:ind+1],
                    feedback_data=output)
            output = self.readout(state)
            outputs[:, ind:ind+1] = output
            hidden_state_history[:, ind:ind+1] = state.x
            visible_state_history[:, ind:ind+1] = state.r

        return {
            "z": outputs,
            "x": hidden_state_history,
            "r": visible_state_history,
        }

    def teach(self, input_data, teacher, state=None, exclude_steps=0):
        """Learn input using linear regression (offline method)

        Note: Does not update the weights in place
        """
        if state is None:
            # initialize random state
            state = ReservoirState(np.random.normal(size=(self.res.N, 1)))

        samples = input_data.shape[1]

        outputs = np.zeros((self.res.N_out, samples))
        hidden_state_history = np.zeros((self.res.N, samples))
        visible_state_history = np.zeros((self.res.N, samples))

        output = None
        for ind in range(samples):
            state = self.res.step(
                    state,
                    input_data=input_data[:, ind:ind+1],
                    feedback_data=teacher[:, ind:ind+1])
            output = self.readout(state)
            outputs[:, ind:ind+1] = output
            hidden_state_history[:, ind:ind+1] = state.x
            visible_state_history[:, ind:ind+1] = state.r

        # do linear regression here
        train_x = hidden_state_history[:, exclude_steps:]
        train_y = teacher[:, exclude_steps:]

        self.W_readout = np.linalg.lstsq(train_x.T, train_y.T)[0]
        print("Finished learning")

        return {
            "z": outputs,
            "x": hidden_state_history,
            "r": visible_state_history,
            "W_readout": self.W_readout
        }
