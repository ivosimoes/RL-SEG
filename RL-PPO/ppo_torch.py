import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pathlib import Path

class PPOMemory:
    ''' Class that holds previous states, probabilities, critic values, actions and dones.
        Also performs operations over the stored data. '''
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.critvals = []   # Critic/loss values
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        ''' Creates randomized indices for each batch.
            Returns the memory contents along with the batch indices. '''
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]   # Creates batches for all data

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.critvals), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, probs, critvals, reward, done):
        ''' Stores the state, action, probs, critvals, reward and done, in this order, in memory. '''
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.critvals.append(critvals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        ''' Clears everything in memory. '''
        self.states = []
        self.actions = []
        self.probs = []
        self.critvals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    ''' Model that makes decisions on the action to take. '''
    def __init__(self, n_actions, input_units, alpha, hidden_units1=256, hidden_units2=256):
        super().__init__()

        self.checkpoint_file = Path('../models/Torch_PPO_Actor.pth')
        self.actor = nn.Sequential(
            nn.Linear(*input_units, out_features=hidden_units1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        ''' Saves instance of model to models folder. '''
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        ''' Loads instance of model from models folder. '''
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    ''' Model that calculates the value of a particular state. '''
    def __init__(self, input_units, alpha, hidden_units1=256, hidden_units2=256):
        super().__init__()

        self.checkpoint_file = Path('../models/Torch_PPO_Critic.pth')
        self.critic = nn.Sequential(
            nn.Linear(*input_units, out_features=hidden_units1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=1)
        )

        self.optimizer = optim.Adam(params=self.parameters(), lr=alpha)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)
    
    def save_checkpoint(self):
        ''' Saves instance of model to models folder. '''
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        ''' Loads instance of model from models folder. '''
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    ''' Class that creates and manages the learning agent. Takes in the number of actions, 
        input units, discount rate (gamma), learning rate (alpha), policy_clip (epsilon), 
        advantage lambda, batch size, horizon/number of steps before update (N) and number of epochs. '''
    def __init__(self, n_actions, input_units, gamma=0.99, alpha=0.0003, policy_clip=0.2, 
                 gae_lambda=0.95, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        
        self.actor = ActorNetwork(n_actions, input_units, alpha)
        self.critic = CriticNetwork(input_units, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        ''' Store current state in memory (along with its data). '''
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        ''' Saves both actor and critic models. '''
        print('Saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        ''' Loads both actor and critic models. '''
        print('Loading models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def choose_action(self, observation):
        ''' Chooses the action to take given the probability distribution from the actor and
            the critic's evaluation of the state. '''
        state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)

        dist = self.actor(state)
        critval = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        critval = torch.squeeze(critval).item()

        return action, probs, critval
    
    def learn(self):
        ''' Training loop. '''
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, values, reward_arr, done_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculating the advantage estimator
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage, device=self.actor.device)

            # Calculating the total loss and training steps
            values = torch.tensor(values, device=self.actor.device)
            for batch in batches:
                # Old policies probability distribution
                states = torch.tensor(state_arr[batch], dtype=torch.float32, device=self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], dtype=torch.float32, device=self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32, device=self.actor.device)

                # New policies probability distribution
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)

                # Probability ratio
                prob_ratio = new_probs.exp() / old_probs.exp()

                # Weighted probability ratio (ratio*advantage)
                weighted_probs = advantage[batch] * prob_ratio

                # Clipped weighted probability ratio (clip(ratio, 1-epsilon, 1+epsilon)*advantage)
                # Note: torch.clamp() is essentially the same as the clip function
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                
                # Clipped surrogate objective
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Squared error loss (L(VF))
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                # Total loss (L(CLIP) + 0.5*L(VF))
                total_loss = actor_loss + 0.5*critic_loss

                # Rest of the training steps
                self.actor.zero_grad()
                self.critic.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()