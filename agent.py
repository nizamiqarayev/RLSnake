import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self,index):
        self.index=index
        
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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
            game.food.x < head.x if game.boxCollected[0]==False else  game.boxSlots[0][0].x < head.x,  # food left
            game.food.x > head.x if game.boxCollected[0]==False else  game.boxSlots[0][0].x > head.x,  # food right
            game.food.y < head.y if game.boxCollected[0]==False else  game.boxSlots[0][0].y < head.y,  # food up
            game.food.y > head.y if game.boxCollected[0]==False else  game.boxSlots[0][0].y > head.y,  # food down
            
        ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


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
        final_move1 = agent1.get_action(state_old1)
        final_move2=agent2.get_action(state_old2)

        # perform move and get new state
        reward1, done1, score1 = game.play_step(0,final_move1)
        
        reward2, done2, score2 = game.play_step(1,final_move2)

        
        state_new1 = agent1.get_state(game)
        state_new2 = agent2.get_state(game)


        # train short memory
        agent1.train_short_memory(
            state_old1, final_move1, reward1, state_new1, done1)

        # remember
        agent1.remember(state_old1, final_move1, reward1, state_new1, done1)
        
        agent2.train_short_memory(
            state_old2, final_move2, reward2, state_new2, done2)

        # remember
        agent2.remember(state_old2, final_move2, reward2, state_new2, done2)





        if done1 or done2:
            # train long memory, plot result
            game.reset()
            agent1.n_games += 1
            agent1.train_long_memory()
            
            agent2.n_games += 1
            agent2.train_long_memory()

            if score1 > record1:
                record1 = score1
                agent1.model.save()
                
            if score2 > record2:
                record2 = score2
                agent2.model.save()
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
            
            
            
        # if done2:
        #     # train long memory, plot result
        #     game.reset()
        #     agent2.n_games += 1
        #     agent2.train_long_memory()

        #     if score2 > record2:
        #         record2 = score2
        #         agent2.model.save()

        #     print('Game', agent2.n_games, 'Score', score2, 'Record:', record2)

        #     plot_scores2.append(score1)
        #     total_score2 += score2
        #     mean_score2 = total_score2 / agent2.n_games
        #     plot_mean_scores2.append(mean_score2)
        #     plot(plot_scores2, plot_mean_scores2)


if __name__ == '__main__':
    train()
