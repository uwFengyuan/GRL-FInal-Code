import torch.nn as nn
import torch.nn.functional as F
from layers import GATLayer, ModifiedGATLayer, GCNLayer

class ModifiedGATModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.ReLU, dropout = 0.4):
      super().__init__()
      self.dropout = dropout
      self.attention = ModifiedGATLayer(input_dim, hidden_dim, concat = True, num_heads=8)
      self.out_attention = ModifiedGATLayer(hidden_dim, output_dim, concat = True, num_heads=1)

    def forward(self, x, adj_matrix):
      x = self.attention(x, adj_matrix)
      x = F.elu(x)
      x = self.out_attention(x, adj_matrix)
      return x

class GATModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.ReLU, dropout = 0.4):
      super().__init__()
      self.dropout = dropout
      self.attention = GATLayer(input_dim, hidden_dim, concat = True, num_heads=8)
      self.out_attention = GATLayer(hidden_dim, output_dim, concat = True, num_heads=1)

    def forward(self, x, adj_matrix):
      x = self.attention(x, adj_matrix)
      x = F.elu(x)
      x = self.out_attention(x, adj_matrix)
      return x

class GCNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.ReLU, dropout = 0.2):
      super().__init__()
      self.dropout = dropout
      self.fisrt = GCNLayer(input_dim, hidden_dim)
      self.second = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj_matrix):
      x = self.fisrt(x, adj_matrix)
      x = F.relu(x)
      x = F.dropout(x, self.dropout, training=self.training)
      x = self.second(x, adj_matrix)
      return x

class MLPModule(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.ReLU, dropout = 0.2):
      super().__init__()

      self.dropout = dropout
      self.act_fn = act_fn
      self.fisrt = nn.Linear(input_dim, hidden_dim)
      self.second = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
      x = self.fisrt(x)
      x = F.relu(x)
      x = F.dropout(x, self.dropout, training=self.training)
      x = self.second(x)
      return x