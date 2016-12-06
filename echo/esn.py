"""A lot of code was plucked from https://github.com/cknd/pyESN (user cknd)
"""

import numpy as np


def compute_spectral_radius(M):
    return np.max(np.abs(np.linalg.eigvals(M)))


def set_spectral_radius(M, radius):
    return M * radius / compute_spectral_radius(M)


class EchoStateNetwork(object):
    def __init__(self,
            n_res,
            n_inputs,
            n_outputs,
            spectral_radius=0.95,
            sparsity=0,
            noise=0.01,
            random_state=None):
        """
        """
        self.n_res = n_res
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self._random = (np.random.RandomState(random_state) if random_state
                else np.random.mtrand._rand)

        self._init_weights()

    def _init_weights(self):
        # first initialize the recurrent weights
        W_res = self._random.uniform(-1, 1, size=(self.n_res, self.n_res))
        W_res[self._random.rand(*W_res.shape) < self.sparsity] = 0
        self.W_res = set_spectral_radius(W_res, self.spectral_radius)

        self.W_in = self._random.uniform(-1, 1, size=(self.n_res, self.n_inputs))
        self.W_fb = self._random.uniform(-1, 1, size=(self.n_res, self.n_outputs))

    def step(self, state, input_data=None, feedback_data=None):
        if input_data is None:
            input_data = np.zeros(self.n_inputs)
        if feedback_data is None:
            feedback_data = np.zeros(self.n_outputs)

        preactivation = (self.W_res @ state +
             self.W_in @ input_data + 
             self.W_fb @ feedback_data)
        noise = self._random.normal(scale=self.noise, size=self.n_res)

        return np.tanh(preactivation) + noise

    def fit(self, inputs, teacher, exclude_steps=100):
        """Learn output weights for teacher
        """
        n_samples = inputs.shape[0]

        # run network over inputs, feeding back the teacher data
        states = np.zeros((n_samples, self.n_res))
        for n in range(1, n_samples):
            states[n] = self.step(states[n-1], inputs[n], teacher[n-1])

        # exclude steps at beginning from fit
        train_x = np.hstack((states, inputs))[exclude_steps:]
        train_y = teacher[exclude_steps:]

        self.W_out = (np.linalg.pinv(train_x) @ train_y).T

        training_outputs = np.hstack((states, inputs)) @ self.W_out.T

        return {"r": states, "inputs": inputs, "outputs": training_outputs}

    def predict(self, inputs, W_out=None, initial_state=None,
            initial_input=None, initial_output=None):
        n_samples = inputs.shape[0]
        W_out = W_out or self.W_out

        if initial_state is None:
            initial_state = np.zeros(self.n_res)
        if initial_input is None:
            initial_input = np.zeros(self.n_inputs)
        if initial_output is None:
            initial_output = np.zeros(self.n_outputs)

        # initialize inputs, states, and outputs with an extra row for the initial state
        states = np.vstack((initial_state, np.zeros((n_samples, self.n_res))))
        inputs = np.vstack((initial_input, inputs))
        outputs = np.vstack((initial_output, np.zeros((n_samples, self.n_outputs))))

        for n in range(n_samples):
            states[n+1] = self.step(states[n], inputs[n+1], outputs[n])
            outputs[n+1] = W_out @ np.concatenate((states[n+1], inputs[n+1]))

        return {"r": states[1:], "inputs": inputs[1:], "outputs": outputs[1:]}
