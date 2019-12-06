import torch
import torch.nn as nn
from torchvision.models import alexnet


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        device,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super(ConvLSTMCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(
            input_size + hidden_size,
            4 * hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        torch.nn.init.xavier_normal_(self.Gates.weight)
        torch.nn.init.constant_(self.Gates.bias, 0)

    def forward(self, input, prev_state):
        batch_size = input.shape[0]
        spatial_size = input.shape[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size).requires_grad_().to(self.device),
                torch.zeros(state_size).requires_grad_().to(self.device)
            )

        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((input, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class ConvLSTM(nn.Module):
    def __init__(self, mem_size, device):
        super(ConvLSTM, self).__init__()
        self.mem_size = mem_size
        self.device = device
        self.conv_net = nn.Sequential(
            *list(alexnet(pretrained=True).features.children())
        )
        self.conv_lstm = ConvLSTMCell(256, self.mem_size, self.device)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.ln1 = nn.Linear(3 * 3 * self.mem_size, 1000)
        self.ln2 = nn.Linear(1000, 256)
        self.ln3 = nn.Linear(256, 10)
        self.ln4 = nn.Linear(10, 2)
        self.bn = nn.BatchNorm1d(1000)
        self.classifier = nn.Sequential(
            self.ln1, self.bn, self.relu, self.ln2, self.relu, self.ln3,
            self.relu, self.ln4
        )

    def forward(self, x):
        # consistency in dimension
        x = x.permute(2, 0, 1, 3, 4)
        state = None
        seqLen = x.size(0) - 1
        for t in range(0, seqLen):
            # diff input
            x1 = x[t] - x[t + 1]
            x1 = self.conv_net(x1)
            state = self.conv_lstm(x1, state)
        x = self.maxpool(state[0])
        x = self.classifier(x.view(x.size(0), -1))

        return x
