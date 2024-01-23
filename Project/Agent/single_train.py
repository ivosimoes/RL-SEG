import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='../DL3RN50_BDD10KBin_4R_500E_2048.pth')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--save', default='RewardV4')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ppo_torch import Agent
from environment import Environment
from tqdm import tqdm
import data
import time

seg_model = args.model
nregions = int(seg_model[21])
imgsize = int(seg_model[29:33])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cut_image(observation, action, nregions, imgsize):
    region_size = imgsize // nregions
    x = action % nregions
    y = action // nregions

    next_observation = observation.clone()
    next_observation[:, 
        y*region_size:(y+1)*region_size,
        x*region_size:(x+1)*region_size] = 0
    
    return next_observation

if __name__ == '__main__':
    env = Environment(seg_model=seg_model,
                      nregions=nregions,
                      image_size=imgsize,
                      device=device)
        
    N = 80  # Samples used in learn step
    batch_size = 16  # Batch size = Region number
    n_epochs = 4  # Epochs of learn step
    lr = 0.0003
    agent = Agent(n_actions=env.possible_actions(), 
                  batch_size=batch_size,
                  alpha=lr,
                  n_epochs=n_epochs,
                  input_units=3,
                  save_name=args.save)
    images = 150
    
    transform = A.Compose([
        A.Resize(imgsize, imgsize),
        A.Normalize(0, 1),
        ToTensorV2(),
    ])
    ds = getattr(data, args.dataset)
    ds = ds('/nas-ctm01/datasets/public', 'test', transform)

    figure_file = 'score_plot.png'

    best_score = float('-inf')
    score_history = []
    total_score = 0
    
    n_steps = 0
    learn_iters = 0

    for i in tqdm(range(images)):
        start = time.time()
        observation = ds[0]['image']  # N x C x H x W
        env.reset(image=ds[0]['image'].to(device),
                    mask=ds[0]['mask'].to(device).long()[:, None])
        done = False
        action_history = []
        avg_scores = []
        reward_history = []
        score = 0
        rep = 0
        while not done:
            action, prob, val = agent.choose_action(np.array(observation.unsqueeze(dim=0), dtype=np.float32))
            reward, done, step_score = env.step(action)
            action_history.append(action)
            reward_history.append(step_score.item())
            if reward < 0:
                rep += 1
            observation_next = cut_image(observation, action, nregions, imgsize)
            n_steps += 1
            score += step_score.item()
            agent.remember(observation.to('cpu'), action, prob, val, reward.to('cpu'), done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            #print(f"Current observation: {observation}\nNext observation: {observation_next}")
            observation = observation_next
            #print(f"Substitute observation: {observation}")
        score_history.append(score)
        total_score += score
        avg_scores.append(total_score/(i+1))

        if score > best_score:
            best_score = score
            agent.save_models()
        
        end = time.time()
        print(f"Image: {i} ({(end-start):.2f}s) | Score: {score:.3f} (Repetitions: {rep}) | Time steps: {n_steps} | Learning steps: {learn_iters}")
        print(f"First 10 actions: {action_history[:10]}")
        reward_history = ["%.2f" % elem for elem in reward_history]
        print(f"Rewards of actions: {reward_history[:10]}")
    
    plt.figure()
    plt.title('Train data')
    plt.xlabel('Number of epochs')
    plt.ylabel('Score')
    plt.plot(score_history)
    plt.plot(avg_scores)
    plt.savefig('scores.png')
    plt.close()