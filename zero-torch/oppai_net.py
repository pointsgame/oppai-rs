import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 3
inner_channels = 8
linear_first = 1024
linear_second = 512

def loss_policy(targets, outputs):
  return -torch.sum(targets * outputs) / targets.size()[0]

def loss_value(targets, outputs):
  return torch.sum((targets - outputs) ** 2) / targets.size()[0]

class OppaiNet(nn.Module):
  def __init__(self, width, height, num_channels):
    super(OppaiNet, self).__init__()

    self.width = width
    self.height = height

    self.conv1 = nn.Conv2d(num_channels, inner_channels, kernel_size, padding = 1)
    self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size, padding = 1)
    self.conv3 = nn.Conv2d(inner_channels, inner_channels, kernel_size)
    self.conv4 = nn.Conv2d(inner_channels, inner_channels, kernel_size)

    self.bn1 = nn.BatchNorm2d(inner_channels)
    self.bn2 = nn.BatchNorm2d(inner_channels)
    self.bn3 = nn.BatchNorm2d(inner_channels)
    self.bn4 = nn.BatchNorm2d(inner_channels)

    self.fc1 = nn.Linear(inner_channels * (width - 4) * (height - 4), linear_first)
    self.fc_bn1 = nn.BatchNorm1d(linear_first)

    self.fc2 = nn.Linear(linear_first, linear_second)
    self.fc_bn2 = nn.BatchNorm1d(linear_second)

    self.fc3 = nn.Linear(linear_second, width * height)
    self.fc4 = nn.Linear(linear_second, 1)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))

    x = x.view(-1, inner_channels * (self.width - 4) * (self.height - 4))

    x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p = 0.3)
    x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p = 0.3)

    policy = F.log_softmax(self.fc3(x), dim = 1).view(-1, self.width, self.height)
    value = torch.tanh(self.fc4(x)).view(-1)

    return policy, value

  def predict(self, inputs):
    policies, values = self(inputs)
    return torch.exp(policies), values

  def train_on(self, optimizer, inputs, policies, values):
    out_policies, out_values = self(inputs)

    policies_loss = loss_policy(policies, out_policies)
    values_loss = loss_value(values, out_values)
    total_loss = policies_loss + values_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
