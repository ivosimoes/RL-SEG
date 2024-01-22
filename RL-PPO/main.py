import gym
import numpy as np
from ppo_torch import Agent
from helper import plot

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    N = 40
    batch_size = 8
    n_epochs = 4
    lr = 0.0003
    agent = Agent(n_actions=env.action_space.n, 
                  batch_size=batch_size,
                  alpha=lr,
                  n_epochs=n_epochs,
                  input_units=env.observation_space.shape)
    n_games = 300

    figure_file = 'score_plot.png'

    best_score = env.reward_range[0]
    score_history = []
    avg_scores = []
    total_score = 0
    
    n_steps = 0
    learn_iters = 0

    for i in range(1, n_games+1):
        observation = env.reset()
        env.render()
        observation = observation[0]
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(np.array(observation, dtype=np.float32))
            observation_next, reward, done, info, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            #print(f"Current observation: {observation}\nNext observation: {observation_next}")
            observation = observation_next
            #print(f"Substitute observation: {observation}")
        score_history.append(score)
        total_score += score
        avg_scores.append(total_score/i)

        if score > best_score:
            best_score = score
            agent.save_models()

        print(f"Game: {i} | Score: {score} | Time steps: {n_steps} | Learning steps: {learn_iters}")
        plot(score_history, avg_scores)