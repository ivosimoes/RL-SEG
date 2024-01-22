import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque  # Double ended queue
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.8  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # if > MAX, popleft()
        self.model = Linear_QNet(input_units=11,  # len(state)
                                 hidden_units=256,  # custom hyperparameter
                                 output_units=3)  # [straight, right, left]
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]  # head of the snake

        # Points around the head of the snake
        # Note: Each block in the game has size of 20 pixels
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Retrieve current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Make the game state
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
        ]
    
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft() if MAXMEM exceeded

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)  # zip(*batch) puts each element of each tuple into a new tuple, according to their position
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # exploration (random) / explotiation (best policy) tradeoff
        self.epsilon = 150 - self.n_games  # randomness hyperparameter
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # perform a random move (random)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # perform model's best move (best policy)
            state0 = torch.tensor(state, dtype=torch.float32)
            pred = self.model(state0)  # agent will most likely return a probability distribution
            move = torch.argmax(pred).item()  # hence the need to perform argmax over the logits tensor
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # current state
        cur_state = agent.get_state(game)

        # get next move
        final_move = agent.get_action(cur_state)

        # next state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train on short memory
        agent.train_short_memory(cur_state, final_move, reward, new_state, done)

        # store data on agent
        agent.remember(cur_state, final_move, reward, new_state, done)

        if done:
            # train on long memory (on all previous experiences), plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            '''
            if score > record_score:
                record_score = score
                agent.model.save()
            '''
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record_score)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()