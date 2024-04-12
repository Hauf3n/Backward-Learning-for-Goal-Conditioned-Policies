import numpy as np
import math
import pygame
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Action_Decider(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        # prediction network
        ff_inner = 512
        network = [
            torch.nn.Linear(args.categoricals*args.classes, ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, args.num_actions)
        ]
        self.decider = torch.nn.Sequential(*network)

    def forward(self, x):    
        return self.decider(x)
    def predict(self, x):
        return F.sigmoid(self.decider(x))

class Goal_Policy(torch.nn.Module):
    
    def __init__(self, embedding_size, num_actions, device, dtype, args, goals=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_actions = num_actions
        self.goals = goals
        self.reached_goal = 0
        self.encoder = None
        self.dtype = dtype
        self.device = device
        self.rm_temperature = args.rm_temperature
        
        # actions embeddings
        self.action_embedding = torch.nn.Embedding(num_actions, embedding_size)
        
        # prediction network
        ff_inner = 128
        network = [
            torch.nn.Linear(embedding_size, ff_inner),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_inner, ff_inner),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_inner, num_actions)
        ]
        self.policy = torch.nn.Sequential(*network)
        
        if self.goals:    
            print("[POLICY] goal encodings: ", len(self.goals))
        self.encountered_states = set()
    
    def set_encoder(self, encoder, device):
        self.encoder = encoder.to(device)
    
    def forward(self, x):    
        actions_scores = self.policy(x)
        actions_probs = F.softmax(actions_scores,dim=1)
        return actions_probs
    
    def act(self, x):
        
        with torch.no_grad():
            
            if self.encoder:
                x = torch.tensor(x).to(self.device).to(self.dtype).unsqueeze(0)
                _,_,x = self.encoder.encode(x, temperature=self.rm_temperature)

            if self.goals:
                x_representation = str(x[0].cpu().numpy())
                self.encountered_states.add(x_representation)
                if x_representation in self.goals:
                    self.reached_goal += 1

            actions_scores = self.policy(x)
            actions_probs = F.softmax(actions_scores,dim=1)
            action = np.random.choice(range(self.num_actions) , 1, p=actions_probs.cpu().numpy()[0])[0]
            return action