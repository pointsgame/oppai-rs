import torch
import torch.nn as nn
import torch.nn.functional as F

INNER_CHANNELS = 192
RESIDUAL_INNER_CHANNELS = INNER_CHANNELS // 2
RESIDUAL_BLOCKS = 5
RESIDUAL_SIZE = 2
GPOOL_EVERY = 2
GPOOL_CHANNELS = 32
V1_CHANNELS = 32
P1_CHANNELS = 32
G1_CHANNELS = 32

class NormMask(nn.Module):
  def __init__(self, channels, gamma_init=False):
    super().__init__()

    self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    if gamma_init:
      self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
    else:
      self.register_parameter('gamma', None)

  def forward(self, inputs, mask):
    if self.gamma is not None:
      return (inputs * self.gamma + self.beta) * mask
    else:
      return (inputs + self.beta) * mask

def gpool_forward(inputs, mask, mask_sum_hw):
  # min size: 16, max size: 40, avg: (40 + 16) / 2 = 28
  mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 28.0

  layer_mean = torch.sum(inputs, dim=(2, 3), keepdim=True) / mask_sum_hw
  # Activation functions is always greater than -1.0, and map 0 -> 0
  layer_max = torch.amax(inputs + (mask - 1.0), dim=(2, 3), keepdim=True)

  out_pool1 = layer_mean
  out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
  out_pool3 = layer_max

  return torch.cat([out_pool1, out_pool2, out_pool3], dim=1)

class ConvAndGPool(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1r = nn.Conv2d(RESIDUAL_INNER_CHANNELS, RESIDUAL_INNER_CHANNELS - GPOOL_CHANNELS, kernel_size=3, padding='same')
    self.conv1g = nn.Conv2d(RESIDUAL_INNER_CHANNELS, GPOOL_CHANNELS, kernel_size=3, padding='same')
    self.normg = NormMask(GPOOL_CHANNELS, gamma_init=False)
    self.actg = nn.GELU()
    self.linearg = nn.Linear(3 * GPOOL_CHANNELS, RESIDUAL_INNER_CHANNELS - GPOOL_CHANNELS)

  def forward(self, inputs, mask, mask_sum_hw):
    outr = self.conv1r(inputs)

    outg = self.conv1g(inputs)
    outg = self.normg(outg, mask)
    outg = self.actg(outg)
    outg = gpool_forward(outg, mask, mask_sum_hw)
    outg = outg.squeeze(-1).squeeze(-1)
    outg = self.linearg(outg)
    outg = outg.unsqueeze(-1).unsqueeze(-1)

    return outr + outg

class ConvOrGpool(nn.Module):
  def __init__(self, gpool, in_channels, out_channels, kernel_size):
    super().__init__()

    self.is_gpool = gpool
    if gpool:
      self.layer = ConvAndGPool()
    else:
      self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')

  def forward(self, inputs, mask, mask_sum_hw):
    if self.is_gpool:
      return self.layer(inputs, mask, mask_sum_hw)
    else:
      return self.layer(inputs)

class NormActConv(nn.Module):
  def __init__(self, gamma, gpool, in_channels, out_channels, kernel_size):
    super().__init__()

    self.norm = NormMask(in_channels, gamma_init=gamma)
    self.act = nn.GELU()
    self.convgpool = ConvOrGpool(gpool, in_channels, out_channels, kernel_size)

  def forward(self, inputs, mask, mask_sum_hw):
    out = self.norm(inputs, mask)
    out = self.act(out)
    return self.convgpool(out, mask, mask_sum_hw)

class InnerResidualBlock(nn.Module):
  def __init__(self, gpool):
    super().__init__()

    self.normactconv1 = NormActConv(
      gamma=False,
      gpool=gpool,
      in_channels=RESIDUAL_INNER_CHANNELS,
      out_channels=RESIDUAL_INNER_CHANNELS,
      kernel_size=3
    )

    in_ch_2 = RESIDUAL_INNER_CHANNELS - GPOOL_CHANNELS if gpool else RESIDUAL_INNER_CHANNELS

    self.normactconv2 = NormActConv(
      gamma=True,
      gpool=False,
      in_channels=in_ch_2,
      out_channels=RESIDUAL_INNER_CHANNELS,
      kernel_size=3
    )

  def forward(self, inputs, mask, mask_sum_hw):
    out = self.normactconv1(inputs, mask, mask_sum_hw)
    out = self.normactconv2(out, mask, mask_sum_hw)
    return inputs + out

class ResidualBlock(nn.Module):
  def __init__(self, gpool):
    super().__init__()

    self.normactconvp = NormActConv(
      gamma=False,
      gpool=False,
      in_channels=INNER_CHANNELS,
      out_channels=RESIDUAL_INNER_CHANNELS,
      kernel_size=1
    )

    self.inner = nn.ModuleList([
      InnerResidualBlock(gpool=(gpool and i == 0)) for i in range(RESIDUAL_SIZE)
    ])

    self.normactconvq = NormActConv(
      gamma=True,
      gpool=False,
      in_channels=RESIDUAL_INNER_CHANNELS,
      out_channels=INNER_CHANNELS,
      kernel_size=1
    )

  def forward(self, inputs, mask, mask_sum_hw):
    out = self.normactconvp(inputs, mask, mask_sum_hw)
    for inner in self.inner:
      out = inner(out, mask, mask_sum_hw)
    out = self.normactconvq(out, mask, mask_sum_hw)
    return inputs + out

class ValueHead(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(INNER_CHANNELS, V1_CHANNELS, kernel_size=1, padding='same')
    self.bias1 = NormMask(V1_CHANNELS, gamma_init=False)
    self.act1 = nn.GELU()
    self.linear2 = nn.Linear(3 * V1_CHANNELS, 1)

  def gpool_value(self, inputs, mask_sum_hw):
    mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 28.0

    layer_mean = torch.sum(inputs, dim=(2, 3), keepdim=True) / mask_sum_hw

    out_pool1 = layer_mean
    out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
    # (sum $ map (\x -> (x - 28) ** 2) [16..40]) / (40 - 16 + 1) / 100
    out_pool3 = layer_mean * (mask_sum_hw_sqrt_offset ** 2 / 100.0 - 0.52)

    return torch.cat([out_pool1, out_pool2, out_pool3], dim=1)

  def forward(self, inputs, mask, mask_sum_hw):
    outv1 = self.conv1(inputs)
    outv1 = self.bias1(outv1, mask)
    outv1 = self.act1(outv1)

    outpooled = self.gpool_value(outv1, mask_sum_hw)
    outpooled = outpooled.reshape(inputs.size(0), -1)

    outv2 = self.linear2(outpooled)
    outv2 = outv2.squeeze(1)
    return torch.tanh(outv2)

class PolicyHead(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1p = nn.Conv2d(INNER_CHANNELS, P1_CHANNELS, kernel_size=1, padding='same')
    self.conv1g = nn.Conv2d(INNER_CHANNELS, G1_CHANNELS, kernel_size=1, padding='same')
    self.biasg = NormMask(G1_CHANNELS, gamma_init=False)
    self.actg = nn.GELU()
    self.linearg = nn.Linear(3 * G1_CHANNELS, P1_CHANNELS)
    self.bias2 = NormMask(P1_CHANNELS, gamma_init=False)
    self.act2 = nn.GELU()
    self.conv2p = nn.Conv2d(P1_CHANNELS, 1, kernel_size=1, padding='same')

  def forward(self, inputs, mask, mask_sum_hw):
    batch, _, height, width = inputs.shape

    outp = self.conv1p(inputs)

    outg = self.conv1g(inputs)
    outg = self.biasg(outg, mask)
    outg = self.actg(outg)

    outg = gpool_forward(outg, mask, mask_sum_hw)
    outg = outg.reshape(batch, -1)
    outg = self.linearg(outg)
    outg = outg.unsqueeze(-1).unsqueeze(-1)

    outp = outp + outg
    outp = self.bias2(outp, mask)
    outp = self.act2(outp)
    outp = self.conv2p(outp)

    # Masking invalid moves
    outp = outp - (1.0 - mask) * 5000.0

    outp = F.log_softmax(outp.reshape(batch, -1), dim=1)
    return outp.reshape(batch, height, width)

class Model(nn.Module):
  def __init__(self, num_channels):
    super().__init__()

    self.initial_conv = nn.Conv2d(num_channels, INNER_CHANNELS, kernel_size=3, padding='same')

    self.residuals = nn.ModuleList([
      ResidualBlock(gpool=((i + 1) % GPOOL_EVERY == 0))
      for i in range(RESIDUAL_BLOCKS)
    ])

    self.norm_trunkfinal = NormMask(INNER_CHANNELS, gamma_init=False)
    self.act_trunkfinal = nn.GELU()

    self.value_head = ValueHead()
    self.policy_head = PolicyHead()

  def forward(self, x):
    mask = x[:, 0:1, :, :]
    mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)

    x = self.initial_conv(x)

    for residual in self.residuals:
      x = residual(x, mask, mask_sum_hw)

    x = self.norm_trunkfinal(x, mask)
    x = self.act_trunkfinal(x)

    policy = self.policy_head(x, mask, mask_sum_hw)
    value = self.value_head(x, mask, mask_sum_hw)

    return policy, value

  def predict(self, inputs):
    self.eval()
    with torch.no_grad():
      policies, values = self(inputs)
      return torch.exp(policies), values

  def train_on(self, optimizer, inputs, policies, values):
    self.train()
    out_policies, out_values = self(inputs)

    diff = out_values - values
    values_loss = torch.sum(diff * diff) / inputs.size(0)

    policies_loss = -torch.sum(out_policies * policies) / inputs.size(0)

    total_loss = policies_loss + values_loss

    print(f"Loss: {total_loss.item()}")

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
