import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from conv2d_rnncells import Conv2dLSTMCell, Conv2dGRUCell, Conv2dRNNCell

class Conv2dRNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, bias, output_size, activation='tanh'):
        super(Conv2dRNN, self).__init__()

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

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if activation == 'tanh':
            self.rnn_cell_list.append(Conv2dRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.kernel_size,
                                                       self.bias,
                                                       "tanh"))

        elif activation == 'relu':
            self.rnn_cell_list.append(Conv2dRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dRNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "relu"))
        else:
            raise ValueError("Invalid activation.")

        self.conv = nn.Conv2d(in_channels=self.hidden_size,
                             out_channels=self.output_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()

        out = self.conv(out)

        return out

class Conv2dLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, bias, output_size):
        super(Conv2dLSTM, self).__init__()

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

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(Conv2dLSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.kernel_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(Conv2dLSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.kernel_size,
                                                self.bias))

        self.conv = nn.Conv2d(in_channels=self.hidden_size,
                             out_channels=self.output_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer], h0[layer]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        out = self.conv(out)

        return out

class Conv2dGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, bias, output_size):
        super(Conv2dGRU, self).__init__()

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

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(Conv2dGRUCell(self.input_size,
                                          self.hidden_size,
                                          self.kernel_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(Conv2dGRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.kernel_size,
                                              self.bias))

        self.conv = nn.Conv2d(in_channels=self.hidden_size,
                             out_channels=self.output_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)


    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()

        out = self.conv(out)

        return out

class Conv2dBidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, kernel_size, num_layers, bias, output_size):
        super(Conv2dBidirRecurrentModel, self).__init__()

        self.mode = mode
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

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if mode == 'LSTM':

            self.rnn_cell_list.append(Conv2dLSTMCell(self.input_size,
                                              self.hidden_size,
                                              self.kernel_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dLSTMCell(self.hidden_size,
                                                    self.hidden_size,
                                                    self.kernel_size,
                                                    self.bias))

        elif mode == 'GRU':
            self.rnn_cell_list.append(Conv2dGRUCell(self.input_size,
                                              self.hidden_size,
                                              self.kernel_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dGRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.kernel_size,
                                                  self.bias))

        elif mode == 'RNN_TANH':
            self.rnn_cell_list.append(Conv2dRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.kernel_size,
                                                       self.bias,
                                                       "tanh"))

        elif mode == 'RNN_RELU':
            self.rnn_cell_list.append(Conv2dRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv2dRNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "relu"))
        else:
            raise ValueError("Invalid RNN mode selected.")

        self.conv = nn.Conv2d(in_channels=self.hidden_size * 2,
                             out_channels=self.output_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, sequence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))

        if torch.cuda.is_available():
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)).cuda())
        else:
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, input.size(-2), input.size(-1)))

        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
            if self.mode == 'LSTM':
                hidden_forward.append((h0[layer], h0[layer]))
            else:
                hidden_forward.append(h0[layer])

        hidden_backward = list()
        for layer in range(self.num_layers):
            if self.mode == 'LSTM':
                hidden_backward.append((hT[layer], hT[layer]))
            else:
                hidden_backward.append(hT[layer])

        for t in range(input.shape[1]):
            for layer in range(self.num_layers):

                if self.mode == 'LSTM':
                    # If LSTM
                    if layer == 0:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            input[:, t, :],
                            (hidden_forward[layer][0], hidden_forward[layer][1])
                            )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            input[:, -(t + 1), :],
                            (hidden_backward[layer][0], hidden_backward[layer][1])
                            )
                    else:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](
                            hidden_forward[layer - 1][0],
                            (hidden_forward[layer][0], hidden_forward[layer][1])
                            )
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](
                            hidden_backward[layer - 1][0],
                            (hidden_backward[layer][0], hidden_backward[layer][1])
                            )

                else:
                    # If RNN{_TANH/_RELU} / GRU
                    if layer == 0:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](input[:, t, :], hidden_forward[layer])
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](input[:, -(t + 1), :], hidden_backward[layer])
                    else:
                        # Forward net
                        h_forward_l = self.rnn_cell_list[layer](hidden_forward[layer - 1], hidden_forward[layer])
                        # Backward net
                        h_back_l = self.rnn_cell_list[layer](hidden_backward[layer - 1], hidden_backward[layer])


                hidden_forward[layer] = h_forward_l
                hidden_backward[layer] = h_back_l

            if self.mode == 'LSTM':

                outs.append(h_forward_l[0])
                outs_rev.append(h_back_l[0])

            else:
                outs.append(h_forward_l)
                outs_rev.append(h_back_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)

        out = self.conv(out)
        return out
