import torch
import torch.nn as nn
from torch.autograd import Variable


class LeNet_1D(nn.Module):
    def __init__(self, hidden_dim):
        super(LeNet_1D, self).__init__()

        self.hidden_dim = hidden_dim

        def outshape(input_size, layers):
            for i in range(layers):
                input_size = (input_size - 4) / 2
            return input_size

        self.conv1 = nn.Conv1d(hidden_dim[0], hidden_dim[1], 5)
        self.conv2 = nn.Conv1d(hidden_dim[1], hidden_dim[2], 5)

        self.fc1 = nn.Linear(hidden_dim[3], 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.convolutions = nn.Sequential(
            self.conv1,  # 19 -> 32
            nn.MaxPool1d(2),
            self.conv2,  # 32 -> 64
            nn.MaxPool1d(2)
        )
        self.linears = nn.Sequential(
            self.fc1,
            nn.LeakyReLU(),
            self.fc2,
            nn.LeakyReLU(),
            self.fc3
        )

        self._initialize_weights()

    def forward(self, x):
        out = self.convolutions(x)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _to_var(self, x, requires_grad=False, volatile=False):
        """
        Varialbe type that automatically choose cpu or cuda
        """
        #     if torch.cuda.is_available():
        #         x = x.cuda()
        return Variable(x, requires_grad=requires_grad, volatile=volatile)

    def _set_mask(self, weight, mask):
        self.register_buffer('mask', mask)
        mask_var = self._to_var(mask, requires_grad=False)
        weight.data = weight.data * mask_var.data
        mask_flag = True

    def set_mask(self, mask):
        self._set_mask(self.conv1.weight, mask[0])
        self._set_mask(self.conv2.weight, mask[1])
        self._set_mask(self.fc1.weight, mask[2])
        self._set_mask(self.fc2.weight, mask[3])
        self._set_mask(self.fc3.weight, mask[4])
