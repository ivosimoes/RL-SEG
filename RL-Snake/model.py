import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

class Linear_QNet(nn.Module):
    def __init__(self, input_units, hidden_units, output_units):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_units,
                                 out_features=hidden_units)
        self.linear2 = nn.Linear(in_features=hidden_units,
                                 out_features=output_units)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, name='Snake.pth'):
        PATH = Path('../models')
        PATH.mkdir(parents=True, exist_ok=True)
        MODEL_NAME = name
        MODEL_SAVING_PATH = PATH / MODEL_NAME
        torch.save(self.state_dict(), MODEL_SAVING_PATH)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=self.lr)
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        if len(state.shape) == 1:
            # Prefered shape: (1, x), x being number of inputs
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )  # tuple of length 1
        
        # 1. Predicted Q values of current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2. Q_new = R + gamma * max(Q_value(next_pred))
        # pred.clone()
        # preds[argmax(action)] = Q_new

        # 3. Torch model train step
        self.model.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()