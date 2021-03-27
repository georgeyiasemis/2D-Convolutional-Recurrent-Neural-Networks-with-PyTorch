import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2dLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(Conv2dLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.bias = bias
        self.x2h = nn.Conv2d(in_channels=input_size,
                             out_channels=hidden_size * 4,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=hidden_size,
                             out_channels=hidden_size * 4,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.Wc = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)
        #       cy: of shape (batch_size, hidden_size, height_size, width_size)

        if self.Wc == None:
            self.Wc = nn.Parameter(torch.zeros((1, self.hidden_size * 3, input.size(2), input.size(3))))

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size, input.size(2), input.size(3)))
            hx = (hx, hx)
        hx, cx = hx

        gates = self.x2h(input) + self.h2h(hx)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        Wci, Wcf, Wco = self.Wc.chunk(3, 1)

        i_t = torch.sigmoid(input_gate + Wci * cx)
        f_t = torch.sigmoid(forget_gate + Wcf * cx)
        g_t = torch.tanh(cell_gate)

        cy = f_t * cx + i_t * torch.tanh(g_t)
        o_t = torch.sigmoid(output_gate + Wco * cy)

        hy = o_t * torch.tanh(cy)


        return (hy, cy)

class Conv2dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True, nonlinearity="tanh"):
        super(Conv2dRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Conv2d(in_channels=input_size,
                             out_channels=hidden_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=hidden_size,
                             out_channels=hidden_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size, input.size(2), input.size(3)))

        hy = (self.x2h(input) + self.h2h(hx))

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy

class Conv2dGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(Conv2dGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.bias = bias
        self.x2h = nn.Conv2d(in_channels=input_size,
                             out_channels=hidden_size * 3,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=hidden_size,
                             out_channels=hidden_size * 3,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size, input.size(2), input.size(3)))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy
