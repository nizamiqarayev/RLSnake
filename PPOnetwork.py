class PPO_Network(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
 
    def forward(self, x):
        action_prob = self.actor(x)
        state_value = self.critic(x)
        return action_prob, state_value
