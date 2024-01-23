import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='../DL3RN50_BDD10KBin_4R_500E_2048.pth')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--save', default='RewardV4_10I')
args = parser.parse_args()

import numpy as np
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
    n_epochs = 6  # Epochs of learn step
    lr = 0.0003
    agent = Agent(n_actions=env.possible_actions(), 
                  batch_size=batch_size,
                  alpha=lr,
                  n_epochs=n_epochs,
                  input_units=3,
                  save_name=args.save)
    images = 30
    
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
    
    n_steps = 0
    learn_iters = 0

    for i in tqdm(range(images)):
        best_score = float('-inf')
        score_history = []
        epochs = max(200//(i+1), 30)
        start = time.time()
        
        for epoch in range(epochs):
            observation = ds[i]['image']  # N x C x H x W
            env.reset(image=ds[i]['image'].to(device),
                        mask=ds[i]['mask'].to(device).long()[:, None])
            done = False
            action_history = []
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
            
            if score > best_score:
                best_score = score
                best_actions = action_history[:]
                best_rewards = reward_history[:]
                agent.save_models()
            
            #print(f"Epoch: {epoch} ({(end-start):.2f}s) | Score: {score:.3f} (Repetitions: {rep}) | Time steps: {n_steps} | Learning steps: {learn_iters}")
            
        end = time.time()
        print(f"Image: {i} ({(end-start):.2f}s) | Score: {best_score:.3f} | Time steps: {n_steps} | Learning steps: {learn_iters}")
        print(f"First 10 actions: {best_actions[:10]}")
        best_rewards = ["%.2f" % elem for elem in best_rewards]
        print(f"Rewards of actions: {best_rewards[:10]}")
        