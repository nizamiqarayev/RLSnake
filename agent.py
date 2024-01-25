import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot
import torch.nn.functional as F


from PPOModel import PPOPolicy, PPOValue
from PpoTrainer import PPOTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, index):
        self.index = index
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        
        self.epsilon = 0 

        # PPO models
        self.policy = PPOPolicy(11, 256, 3)
        self.value = PPOValue(11, 256)
        self.trainer = PPOTrainer(self.policy, self.value, lr=LR)
        
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        
    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def get_state(self, game):

        head = game.heads[self.index]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction[self.index] == Direction.LEFT
        dir_r = game.direction[self.index] == Direction.RIGHT
        dir_u = game.direction[self.index] == Direction.UP
        dir_d = game.direction[self.index] == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < head.x if game.boxCollected[0]==False else  game.boxSlots[31][0].x < head.x,  # food left
            game.food.x > head.x if game.boxCollected[0]==False else  game.boxSlots[31][0].x > head.x,  # food right
            game.food.y < head.y if game.boxCollected[0]==False else  game.boxSlots[31][0].y < head.y,  # food up
            game.food.y > head.y if game.boxCollected[0]==False else  game.boxSlots[31][0].y > head.y,  # food down
            
        ]
        
        return np.array(state, dtype=int)



    def get_action(self, state):
        
        gameMove = [0, 0, 0]
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_probs = self.policy(state)
                
        action_dist = torch.distributions.Categorical(action_probs)
        self.epsilon = 80 - self.n_games
   
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            newAction = torch.tensor(move, dtype=torch.float)

            gameMove[move]=1
            log_prob = action_dist.log_prob(newAction)
            
            self.old_log_probs.append(log_prob)
        else:
            newAction = action_dist.sample()
            log_prob = action_dist.log_prob(newAction)
            self.old_log_probs.append(log_prob)
            
            move = newAction.item()
            gameMove[move]=1
            print(gameMove)


        return newAction, gameMove

        
        


def train():
    plot_scores1 = []
    plot_mean_scores1 = []
    total_score1 = 0
    
    plot_scores2 = []
    plot_mean_scores2 = []
    total_score2 = 0
    
    record1 = 0
    record2=0
    agent1 = Agent(0)
    agent2=Agent(1)
    agents=[agent1,agent2]
    game = SnakeGameAI(agents)
    while True:
        # get old state
        state_old1 = agent1.get_state(game)
        state_old2=agent2.get_state(game)

        # get move
        final_move1,gameMove1 = agent1.get_action(state_old1)
        final_move2,gameMove2=agent2.get_action(state_old2)
        
   
        # perform move and get new state
        reward1, done1, score1 = game.play_step(0,gameMove1)
        
        reward2, done2, score2 = game.play_step(1,gameMove2)
        

        
        print("=========================================")
        print(final_move1.item())
        print(reward1)

        print(final_move2.item())
        print(reward2)

        print("=========================================")

        
        state_new1 = agent1.get_state(game)
        state_new2 = agent2.get_state(game)

        agent1.store_transition(state_old1, final_move1, reward1, state_new1, done1)
        agent2.store_transition(state_old2, final_move2,  reward2, state_new2, done2)




        if done1 or done2:
            # train long memory, plot result
            
            agent1.trainer.train_step(agent1.states, agent1.actions, agent1.old_log_probs, agent1.rewards,   agent1.next_states, agent1.dones)
            agent2.trainer.train_step(agent2.states, agent2.actions, agent2.old_log_probs, agent2.rewards,   agent2.next_states, agent2.dones)



            
            game.reset()
            
            agent1.clear_memory()
            agent2.clear_memory()

            agent1.n_games += 1
            
            agent2.n_games += 1
            
            

            if score1 > record1:
                record1 = score1
                
            if score2 > record2:
                record2 = score2
            print("==================")
            print('Game', agent1.n_games, 'Score', score1, 'Record:', record1)
            print('Game', agent1.n_games, 'Score', score1, 'Record:', record1)

            plot_scores1.append(score1)
            total_score1 += score1
            mean_score1 = total_score1 / agent1.n_games
            plot_mean_scores1.append(mean_score1)
            plot(plot_scores1, plot_mean_scores1)
            
            
            
            plot_scores2.append(score1)
            total_score2 += score2
            mean_score2 = total_score2 / agent2.n_games
            plot_mean_scores2.append(mean_score2)
            plot(plot_scores2, plot_mean_scores2)
            
            
            
   

if __name__ == '__main__':
    train()
