import torch
import torch.nn as nn


class CNN_RNN(nn.Module):
    def __init__(self, input_dim, input_size, output_dim, device, kernel_size=5, pool_kernel_size=2, dropout=0, convolution_layers=1,
                 rnn_layers=1, rnn_type='GRU', bidirectional=False, initialized=True):
        super(CNN_RNN, self).__init__()

        self.hidden_dim = [32, 64, 32, 16]
        self.ks = kernel_size
        self.device = device
        self.rnn_layers = rnn_layers
        self.convolution_layers = convolution_layers

        self.num_direction = 2 if bidirectional else 1

        def outshape(input_size, layers):
            for i in range(layers):
                input_size = (input_size - 4) / 2
            return input_size

        fc_inshape = int(outshape(input_size, convolution_layers) * self.hidden_dim[
            convolution_layers + rnn_layers - 1] * self.num_direction)

        # CNN Layers
        cnn = nn.Sequential()

        def convRelu(i, leakyRelu=False):
            nIn = input_dim if i == 0 else self.hidden_dim[i - 1]
            nOut = self.hidden_dim[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv1d(nIn, nOut, self.ks))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU())
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU())

        for j in range(convolution_layers):
            convRelu(j, True)
            cnn.add_module('pooling{0}'.format(j), nn.MaxPool1d(2))  # input_dim -> 32 L out: 96

        self.cnn = cnn

        # RNN Layers
        if rnn_layers == 1:
            self.rnn1 = nn.Sequential(
                getattr(nn, rnn_type)(self.hidden_dim[convolution_layers - 1],
                                      self.hidden_dim[convolution_layers], 1,
                                      bidirectional=bidirectional))
        else:
            self.rnn1 = getattr(nn, rnn_type)(self.hidden_dim[convolution_layers - 1],
                                              self.hidden_dim[convolution_layers], 1,
                                              bidirectional=bidirectional)
            self.rnn_dropout = nn.Dropout(p=dropout)
            self.rnn2 = getattr(nn, rnn_type)(self.num_direction * self.hidden_dim[convolution_layers],
                                              self.hidden_dim[convolution_layers + 1], 1,
                                              bidirectional=bidirectional)

        # FC Layers
        fc = nn.Sequential(
            nn.Linear(fc_inshape, int(fc_inshape / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(fc_inshape / 2), output_dim),
            nn.LeakyReLU()
        )
        self.fc = fc

        if initialized:
            self._initialize_weights()

    def _init_hidden(self, batch_size):
        self.hn1 = nn.init.kaiming_normal_(
            torch.zeros(self.num_direction * self.rnn_layers, batch_size, self.hidden_dim[self.convolution_layers],
                        device=self.device))  # (num_layers * num_directions, batch, hidden_size)
        self.hn2 = nn.init.kaiming_normal_(
            torch.zeros(self.num_direction * self.rnn_layers, batch_size, self.hidden_dim[self.convolution_layers + 1],
                        device=self.device))  # (num_layers * num_directions, batch, hidden_size)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.kaiming_normal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        CNN input shape: (batch, Cin, Lin)
        CNN output shape: (batch, Cout, Lout)

        Lout = (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

        RNN input shape: (seq_len, batch, input_size)
        RNN hidden layer shape: (num_layers * num_directions, batch, hidden_size)
        RNN output shape: (seq_len, batch, num_directions * hidden_size)

        Two FC layers
        """
        batch_size = x.shape[0]

        out = self.cnn(x)  # batch, Cout, sequence -> sequence, batch, Cout
        out = out.permute(2, 0, 1)  # sequence, batch, Cout
        self._init_hidden(out.shape[1])
        if self.rnn_layers > 1:
            out, self.hn1 = self.rnn1(out)  # (seq_len, batch, num_directions * hidden_size)
            out = self.rnn_dropout(out)
            out, self.hn2 = self.rnn2(out)
        else:
            out, self.hn1 = self.rnn1(out)
        out = out.permute(1, 0, 2)
        out = torch.reshape(out, (batch_size, -1))
        out = self.fc(out)  # (batch, seq_len, 1)
        return out
