import pygame
import random
import math
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
GRAY = (211, 211, 211)
PEACH=(252,182,159)

GREEN = (0, 255, 0)

RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
ORANGE=(255,165,0)

BLOCK_SIZE = 20
SPEED = 40

obstaclesCoordinateArray=[Point(260, 240), Point(240, 240), Point(220, 240), Point(200, 240), Point(180, 240),
 Point(440, 240), Point(460, 240), Point(480, 240),
 Point(500, 240), Point(520, 240), Point(260, 380), Point(240, 380), Point(220, 380),
 Point(200, 380), Point(180, 380),
 Point(440, 380), Point(460, 380), Point(480, 380),
 Point(500, 380), Point(520, 380),  Point(260, 400), Point(260, 420), Point(260, 440),
 Point(260, 460),  Point(440, 400), Point(440, 420), Point(440, 440),
 Point(440, 460),  Point(260, 240), Point(260, 220), Point(260, 200),
 Point(260, 180),  Point(440, 240), Point(440, 220), Point(440, 200), Point(440, 180),
 
 Point(0, 0), Point(20, 0), Point(40, 0), Point(60, 0), Point(80, 0),
 Point(100, 0), Point(120, 0), Point(140, 0), Point(160, 0), Point(180, 0),
 Point(200, 0), Point(220, 0), Point(240, 0), Point(260, 0), Point(280, 0),
 Point(300, 0), Point(320, 0), Point(340, 0), Point(360, 0), Point(380, 0),
 Point(400, 0), Point(420, 0), Point(440, 0), Point(460, 0), Point(480, 0),
 Point(500, 0), Point(520, 0), Point(540, 0), Point(560, 0), Point(580, 0),
 Point(600, 0), Point(620, 0), Point(640, 0), Point(660, 0), Point(680, 0),
 Point(700, 0),
 
 
 Point(0, 20), Point(0, 40), Point(0, 60), Point(0, 80),
 Point(0, 100), Point(0, 120), Point(0, 140), Point(0, 160), Point(0, 180),
 Point(0, 200), Point(0, 220), Point(0, 240), Point(0, 260), Point(0, 280),
 Point(0, 300), Point(0, 320), Point(0, 340), Point(0, 360), Point(0, 380),
 Point(0, 400), Point(0, 420), Point(0, 440), Point(0, 460), Point(0, 480),
 Point(0, 500), Point(0, 520), Point(0, 540), Point(0, 560), Point(0, 580),
 Point(0, 600), Point(0, 620), Point(0, 640), Point(0, 660), Point(0, 680),
 Point(0, 700),
 
 Point(700, 20), Point(700, 40), Point(700, 60), Point(700, 80),
 Point(700, 100), Point(700, 120), Point(700, 140), Point(700, 160), Point(700, 180),
 Point(700, 200), Point(700, 220), Point(700, 240), Point(700, 260), Point(700, 280),
 Point(700, 300), Point(700, 320), Point(700, 340), Point(700, 360), Point(700, 380),
 Point(700, 400), Point(700, 420), Point(700, 440), Point(700, 460), Point(700, 480),
 Point(700, 500), Point(700, 520), Point(700, 540), Point(700, 560), Point(700, 580),
 Point(700, 600), Point(700, 620), Point(700, 640), Point(700, 660), Point(700, 680),
 Point(700, 700),
 
 Point(20, 700), Point(40, 700), Point(60, 700), Point(80, 700),
 Point(100, 700), Point(120, 700), Point(140, 700), Point(160, 700), Point(180, 700),
 Point(200, 700), Point(220, 700), Point(240, 700), Point(260, 700), Point(280, 700),
 Point(300, 700), Point(320, 700), Point(340, 700), Point(360, 700), Point(380, 700),
 Point(400, 700), Point(420, 700), Point(440, 700), Point(460, 700), Point(480, 700),
 Point(500, 700), Point(520, 700), Point(540, 700), Point(560, 700), Point(580, 700),
 Point(600, 700), Point(620, 700), Point(640, 700), Point(660, 700), Point(680, 700),
 ]

class SnakeGameAI:

    def __init__(self,agents, w=720, h=720,initialBoxCollected=False):
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
        self.direction = [Direction.RIGHT,Direction.RIGHT]
        self.distanceOfAgents=[]

        self.failed_move_counter=0

        self.heads = []
        for i in self.agents:
            self.heads.append(Point(self.w/2, self.h/2))  
        self.collectedBoxesLocation=[]
   
        self.score = 0
        
        self.boxCollected=[False,None]
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        self.boxSlots=[]
        
        
        
        tempy=12
        tempx=14
        for i in range(64):
          
            if tempx*20>420:
                tempy+=1
                tempx=14

            self.boxSlots.append([Point(tempx*BLOCK_SIZE,tempy*BLOCK_SIZE),False])
            # print(self.boxSlots)
            tempx+=1
        
        
        self.nextEmptyBoxSlotIndex=0

    def _place_food(self):
        
        
        x=420
        while x<440 and x>280:
            x=random.randint(10, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        
        
        
        y=380
        while y<400 and y>240:
            y=random.randint(10, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            
            
            
        self.food = Point(x, y)
        
        for head in self.heads:
            self.distanceOfAgents.append(self.calculate_distance(head.x,head.y,self.food.x,self.food.y))

        
        if self.food in self.heads or self.food in obstaclesCoordinateArray:
            self._place_food()

    def play_step(self,whichAgent, action):
     

        
        # print(self.nextEmptyBoxSlotIndex)

        self.currentAgent=self.agents[whichAgent]
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        
        for index,boxSlot in enumerate(self.boxSlots):
            if boxSlot[1]:
                continue
            else:
                self.nextEmptyBoxSlotIndex=index
                break
        
        
        
        
        
        self._move(whichAgent,action)
        
        reward = 0
        game_over = False
        
        
        
        if(self.boxCollected[0]):
                currentDistance=self.calculate_distance(self.heads[whichAgent].x,self.heads[whichAgent].y,self.boxSlots[self.nextEmptyBoxSlotIndex][0].x,self.boxSlots[self.nextEmptyBoxSlotIndex][0].y) 
                if currentDistance<self.distanceOfAgents[whichAgent]:
                    reward+=5
                    self.distanceOfAgents[whichAgent]=currentDistance
                else:
                    reward-=5
                    self.failed_move_counter+=1
                    # print(self.failed_move_counter)
                    if(self.failed_move_counter>400):
                        game_over=True
        else:
                currentDistance=self.calculate_distance(self.heads[whichAgent].x,self.heads[whichAgent].y,self.food.x,self.food.y) 
                if currentDistance<self.distanceOfAgents[whichAgent]:
                    reward=5
                    self.distanceOfAgents[whichAgent]=currentDistance
                else:
                    reward=-5
                    self.failed_move_counter+=1
                    # print(self.failed_move_counter)

                    if(self.failed_move_counter>400):
                        game_over=True

        
        # update the head
        # 3. check if game over
      
        for index,i in enumerate(self.heads):
            
# or self.frame_iteration > 100*len(self.heads)
             if self.is_collision(i):
                 game_over=True
                 reward = -10
                 return reward, game_over, self.score

        # 4. place new food or just move
             if i == self.food:
                self.failed_move_counter=0

                self.boxCollected[0]=True
                self.boxCollected[1]=index
                reward+=10
                self.score += 1

            
                self._place_food()
                
             for boxSlotLocation in self.boxSlots:
                 if(i==boxSlotLocation[0] and self.boxCollected[0]==True and self.boxCollected[1]==index and boxSlotLocation[1]==False):
                     self.failed_move_counter=0

                     boxSlotLocation[1]=True
                     self.score += 5
                     reward+= 20
                     self.boxCollected[0]=False
                     self.boxCollected[1]=None
                 
                   

    
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        tempVar=pt
        if pt is None:
            tempVar = self.currentAgent
        # hits boundary
        if tempVar.x > self.w - BLOCK_SIZE or tempVar.x < 0 or tempVar.y > self.h - BLOCK_SIZE or tempVar.y < 0:
            return True
        # hits itself
        if tempVar in obstaclesCoordinateArray:
            return True
        #     return True

        return False
    
    def calculate_distance(self,x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def _update_ui(self):
        self.display.fill(PEACH)
        x, y, size = 0, 0, 160 
        pygame.draw.rect(self.display, GRAY, (280, 240, 160, 160), 100)
        for obstacle in obstaclesCoordinateArray:
            # print(obstacle.x)
            pygame.draw.rect(self.display,ORANGE,(obstacle.x,obstacle.y,BLOCK_SIZE,BLOCK_SIZE),100)
        
        for box in self.boxSlots:
            pygame.draw.rect(self.display, GREEN if box[1] else GRAY, pygame.Rect(
                box[0].x, box[0].y, BLOCK_SIZE, BLOCK_SIZE))
        for index,pt in enumerate(self.heads):
            pygame.draw.rect(self.display, RED if self.boxCollected[0] and self.boxCollected[1]==index else BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        if(self.boxCollected[0]==False):
            pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self,whichAgent, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN,
        Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction[whichAgent])

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction[whichAgent] = new_dir

        x = self.heads[whichAgent].x
        y = self.heads[whichAgent].y
        if self.direction[whichAgent] == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction[whichAgent] == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction[whichAgent] == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction[whichAgent] == Direction.UP:
            y -= BLOCK_SIZE

        self.heads[whichAgent] = Point(x, y)
