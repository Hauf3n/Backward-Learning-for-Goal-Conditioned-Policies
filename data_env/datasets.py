import numpy as np
import bisect
from torch.utils.data import Dataset
import torch

class ImitationDataset(Dataset):
    
    def __init__(self, pairs):
        
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, index):
        state, action, weight = self.pairs[index]
        return torch.tensor(np.array(state)), torch.tensor(action), torch.tensor(weight)

class Tabular_Dataset_Backward(Dataset): 

    def __init__(self, states, actions, rewards, num_actions):

        self.states = states
        self.actions = actions
        self.rewards = rewards

        # precompute cumulative lengths
        self.cumulative_lengths = [0]
        for sequence in self.states:
            self.cumulative_lengths.append(len(sequence) + self.cumulative_lengths[-1])

    def __getitem__(self, index):

        # find the sequence index using binary search
        sequence_index = bisect.bisect_right(self.cumulative_lengths, index) - 1
        element_index = index - self.cumulative_lengths[sequence_index]

        # if element_index is 0, set it to 1 to avoid negative indexing
        element_index = max(1, element_index)

        # get state / action / target
        state = self.states[sequence_index][element_index]
        action = self.actions[sequence_index][element_index - 1]
        target = self.states[sequence_index][element_index - 1]

        return state, np.array(action), target

    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def add(self, sequence):
        self.states.append(sequence)
        self.cumulative_lengths.append(len(sequence) + self.cumulative_lengths[-1])
        
    def pop_sequence(self):
        if len(self.states) > 0:
            first_sequence_length = len(self.states[0])
            self.states.pop(0)
            self.cumulative_lengths = [cumulative_length - first_sequence_length for cumulative_length in self.cumulative_lengths[1:]]
    
