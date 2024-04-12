import torch
import torch.nn.functional as F
import torch.distributions as D
 
class Categorical_Representation(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.embedding_size = args.input_dim
        self.categoricals = args.categoricals
        self.classes = args.classes
        self.input_features = args.input_dim
        self.args = args
        
        # encoder
        encoder_network = [
            torch.nn.Conv2d(self.input_features, 48, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.SiLU(),
            torch.nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(256, self.categoricals*self.classes)
        ]
        
        # decoder
        decoder_network = [
            torch.nn.ConvTranspose2d(self.categoricals*self.classes, 256, kernel_size=2, stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(48, 3, kernel_size=2, stride=2, padding=0, bias=True)
        ]
        
        self.encoder = torch.nn.Sequential(*encoder_network)
        self.decoder = torch.nn.Sequential(*decoder_network)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # encode
        m, distribution, sample = self.encode(x)
        
        # decode
        decoder_sample = sample.view(batch_size, self.categoricals*self.classes, 1, 1)
        reconstruction = self.decode(decoder_sample)
        
        return reconstruction, distribution, sample, m.entropy()
    
    def encode(self, x, flatten_sample=True, temperature=1.0):
        batch_size = x.shape[0]
        
        # encode
        scores = self.encoder(x)
        scores = torch.reshape(scores, (batch_size, self.categoricals, self.classes)) / temperature
        distribution = F.softmax(scores, dim=2)
        
        # categorical bottleneck
        m = D.Independent(D.OneHotCategoricalStraightThrough(distribution), 1)
        
        sample = m.rsample()
        if flatten_sample:
            sample = torch.reshape(sample, (batch_size, self.categoricals*self.classes))
        
        return m, distribution, sample
    
    def decode(self, sample):
        return self.decoder(sample)
        
    # shape output test  
    def test(self, x):
        scores = self.encoder(x)
        return scores
    
    def test2(self, x):
        scores = self.decoder(x)
        return scores