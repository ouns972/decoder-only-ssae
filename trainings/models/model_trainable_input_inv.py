import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


@torch.no_grad()
def compute_W(X, Y, lambd, I):
    A = X.T @ X
    return torch.linalg.solve(A + lambd * I, X.T @ Y)

class Decoder(nn.Module):
    def __init__(self, n_data, n_properties, n_repeat, Y_dim, is_the_same_indices, n_features_situation, lambd=0):
        super().__init__()
        # W is not considered as a trainable parameters (no gradient descent)

        self.n_data = n_data
        self.n_properties = n_properties
        self.n_repeat = n_repeat
        self.n_features = n_properties * n_repeat
        self.Y_dim = Y_dim
        self.lambd = lambd

        self.register_buffer('I', torch.eye(self.n_features))

        self.latent = nn.Parameter(torch.rand(self.n_data, self.n_features, requires_grad = True))
        self.register_buffer('latent_with_mask', torch.zeros_like(self.latent))
        self.register_buffer('W', torch.zeros((self.n_features, self.Y_dim)))

        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, Y):
        if self.training:
            latent_with_mask = self.apply_mask()
            #latent_with_mask = self.dropout(latent_with_mask)
            self.W = compute_W(latent_with_mask, Y, self.lambd, self.I)
            return latent_with_mask @ self.W 
        else:
            return self.latent_with_mask @ self.W 

    
    def initialize_mask(self, mask):
        mask = torch.repeat_interleave(mask, self.n_repeat, dim=1)
        self.mask = mask

    def apply_mask(self):
        if self.mask is None:
            raise Exception("mask not defined...")
        
        self.latent_with_mask =  self.activation(self.latent * self.mask)
        return self.latent_with_mask
    
