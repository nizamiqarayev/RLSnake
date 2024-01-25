import torch.nn as nn
import torch.nn.functional as F

class PPOPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

class PPOValue(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PPOValue, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.value_head(x)
        return state_value