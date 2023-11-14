import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:

    def __init__(self,agents, w=640, h=480,initialBoxCollected=False):
        self.w = w
        self.h = h
        self.agents=agents
        
        self.boxCollected=[initialBoxCollected,None]
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.heads = []
        for i in self.agents:
            self.heads.append(Point(self.w/2, self.h/2))
        # self.snake = [self.head,
        # # Point(self.head.x-BLOCK_SIZE, self.head.y),
        # # Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        # ]
        
        self.collectedBoxesLocation=[]

        self.score = 0
        
        self.boxCollected=[False,None]
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(10, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(10, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.heads:
            self._place_food()

    def play_step(self,whichAgent, action):
        self.currentAgent=self.agents[whichAgent]
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(whichAgent,action)  # update the head
        # 3. check if game over
        reward = 0
        game_over = False
        for index,i in enumerate(self.heads):
# or self.frame_iteration > 100*len(self.heads)
             if self.is_collision(i):
                 game_over=True
                 reward = -10
                 return reward, game_over, self.score

        # 4. place new food or just move
             if i == self.food:
            
                self.boxCollected[0]=True
                self.boxCollected[1]=index
            
                self._place_food()
             if(self.display.get_at((int(i.x), int(i.y)))==WHITE and self.boxCollected[0]):
                 # pygame.draw.rect(self.display, RED, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
                 self.score += 1
                 reward = 10
                 self.boxCollected[0]=False
                 self.boxCollected[1]=None
                 self.collectedBoxesLocation.append(Point(i.x, i.y))
            #  else:
            #      self.snake.pop()
            
       
        # if(self.boxCollected and self.head==)

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        tempVar=pt
        if pt is None:
            tempVar = self.currentAgent
        # hits boundary
        if tempVar.x > self.w - BLOCK_SIZE or tempVar.x < 0 or tempVar.y > self.h - BLOCK_SIZE or tempVar.y < 0:
            return True
        # hits itself
        # if head in self.heads[1:]:
        #     return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        x, y, size = 0, 0, 200 

        pygame.draw.rect(self.display, WHITE, (x, y, size, size), 100)
        for index,pt in enumerate(self.heads):
            # print('===============')
            
            # print(self.boxCollected[1])
            # print(pt)
            # print('===============')

            # if(self.boxCollected):
            #     pygame.draw.rect(self.display, RED, pygame.Rect(
            #     pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, RED if self.boxCollected[0] and self.boxCollected[1]==index else BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # pygame.draw.rect(self.display, BLUE2,
            # pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        for box in self.collectedBoxesLocation:
            pygame.draw.rect(self.display, RED, pygame.Rect(
                box.x, box.y, BLOCK_SIZE, BLOCK_SIZE))


        if(self.boxCollected[0]==False):
            pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

      
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self,whichAgent, action):
        # [straight, right, left]
        print(action)
        clock_wise = [Direction.RIGHT, Direction.DOWN,
        Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.heads[whichAgent].x
        y = self.heads[whichAgent].y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.heads[whichAgent] = Point(x, y)
