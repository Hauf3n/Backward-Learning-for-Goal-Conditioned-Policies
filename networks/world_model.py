import torch
import torch.nn.functional as F
import torch.distributions as D
   

class BWM(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.classes = args.classes
        self.categoricals = args.categoricals
        self.args = args
        
        # actions embedding
        self.action_embedding = torch.nn.Embedding(args.num_actions, args.action_emb_size)
        
        # prediction network
        ff_inner = 2048
        network = [
            torch.nn.Linear(self.classes*self.categoricals + args.action_emb_size, ff_inner, bias=False),
            torch.nn.BatchNorm1d(ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, ff_inner, bias=False),
            torch.nn.BatchNorm1d(ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, ff_inner, bias=False),
            torch.nn.BatchNorm1d(ff_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(ff_inner, self.categoricals*self.classes)
        ]
        self.prediction_network = torch.nn.Sequential(*network)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        scores = self.prediction_network(x) 
        scores = torch.reshape(scores, (batch_size, self.categoricals, self.classes))
        categoricals = F.softmax(scores, dim=2)
               
        return categoricals
    
    def sample(self, x, temperature=1.0):
        batch_size = x.shape[0]
        
        scores = self.prediction_network(x)
        scores = torch.reshape(scores, (batch_size, self.categoricals, self.classes)) / temperature
        categoricals = F.softmax(scores, dim=2)
        
        m = D.Independent(D.OneHotCategoricalStraightThrough(categoricals),1)
        sample = m.rsample()
        
        reshaped_sample = torch.reshape(sample, (batch_size, self.categoricals*self.classes))
        
        return reshaped_sample, sample
        