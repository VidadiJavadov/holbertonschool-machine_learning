import numpy as np


class GRUCell:
    """
    Represents a Gated Recurrent Unit (GRU) cell
    """

    def __init__(self, i, h, o):
        """
        Initialize the GRU cell

        i: input dimension
        h: hidden dimension
        o: output dimension
        """

        # Update gate parameters
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate parameters
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Candidate hidden state parameters
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output parameters
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step

        h_prev: (m, h)
        x_t:    (m, i)

        Returns:
        h_next, y
        """

        # Concatenate hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)

        # Reset gate
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        # Apply reset gate
        r_h_prev = r * h_prev

        # Concatenate reset-hidden with input
        concat_reset = np.concatenate((r_h_prev, x_t), axis=1)

        # Candidate hidden state
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Final hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        z_out = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(z_out)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
