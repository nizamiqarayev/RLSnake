
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, policy_model, value_model, lr, gamma=0.99, clip_param=0.2, ppo_epochs=4, critic_discount=0.5, entropy_beta=0.01):
        self.policy_model = policy_model
        self.value_model = value_model
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta

        self.policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_model.parameters(), lr=lr)

    def train_step(self, states, actions, old_log_probs, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # Calculate advantages and returns
        with torch.no_grad():
            values = self.value_model(states)
            next_values = self.value_model(next_states)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = deltas.clone()
            for t in reversed(range(len(rewards) - 1)):
                advantages[t] = advantages[t] + (self.gamma * advantages[t + 1] * (1 - dones[t]))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy and value networks
        for _ in range(self.ppo_epochs):
            log_probs, state_values, entropy = self.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2) - self.entropy_beta * entropy
            value_loss = F.mse_loss(state_values, rewards + self.gamma * next_values * (1 - dones))

            # Take optimization steps
            self.policy_optimizer.zero_grad()
            policy_loss.mean().backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def evaluate(self, states, actions):
        action_probs = self.policy_model(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        state_values = self.value_model(states)

        return log_probs, state_values.squeeze(-1), entropy
