import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """

        # Weights initialization (normal distribution)
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialization (zeros)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function"""
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: previous hidden state (m, h)
        c_prev: previous cell state (m, h)
        x_t: input data (m, i)

        Returns:
        h_next: next hidden state
        c_next: next cell state
        y: output
        """

        # Concatenate hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)

        # Update gate
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)

        # Candidate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Output gate
        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        # Next cell state
        c_next = f * c_prev + u * c_tilde

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
