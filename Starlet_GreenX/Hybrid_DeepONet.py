import torch
import torch.nn as nn

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNet, self).__init__()
        # Input u: [Vs, Ps, CO2_source, Q_supply] -> Dim = 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrunkNet, self).__init__()
        # Input y: [x, y, z] -> Dim = 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class HybridDeepONet(nn.Module):
    def __init__(self, branch_in_dim=4, trunk_in_dim=3, latent_dim=128):
        super(HybridDeepONet, self).__init__()
        
        self.branch_net = BranchNet(branch_in_dim, 256, latent_dim)
        self.trunk_net = TrunkNet(trunk_in_dim, 256, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.output_activation = nn.Softplus()

    def forward(self, u, y):
        b = self.branch_net(u)
        t = self.trunk_net(y)
        output = torch.sum(b * t, dim=1, keepdim=True) + self.bias
        return self.output_activation(output)