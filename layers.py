import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads=1, concat=True, alpha=0.2):
        """
        Inputs:
            input_dim - Dimensionality of input features
            output_dim - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
            dropout - Dropout parameter which is for overfitting.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        if self.concat:
            assert output_dim % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            output_dim = output_dim // num_heads

        # Create W: [output_dim * num_heads, input_dim]
        # Initialize W by torch.nn.init.xavier_uniform_(tensor, gain=1.414)
        self.W = nn.Linear(input_dim, output_dim * num_heads)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)

        # Create a: [num_heads, 2*output_dim]
        # Initialize a by torch.nn.init.xavier_uniform_(tensor, gain=1.414)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * output_dim))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # active function
        self.leakyrelu = nn.LeakyReLU(alpha)
               

    def forward(self, h, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [num_nodes, num_features] 
            adj_matrix - Adjacency matrix including self-connections. Shape: [num_nodes, num_nodes]
        """

        num_nodes = h.size(0) # 2708

        # Apply linear layer and sort nodes by head
        # [output_dim * num_heads, input_dim]*[2708, 1433] = [2708, output_dim]
        Wh = self.W(h) 

        # [2708, num_heads, output_dim]
        Wh = Wh.view(num_nodes, self.num_heads, -1) 

        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero(as_tuple=True) #[10556]

        # [10556, num_heads, 2*output_dim], [W*h_i||W*h_j] 
        # Index select returns a tensor with node_feats_flat 
        # being indexed at the desired positions along dim=0
        WhiWhj = torch.cat([
            torch.index_select(input=Wh, index=edges[0], dim=0),
            torch.index_select(input=Wh, index=edges[1], dim=0)
        ], dim=-1)

        # LeakyReLU(a*[W*h_i||W*h_j] ) 
        # [10556, num_heads, 2*output_dim], [num_heads, 2*output_dim] = [10556, num_heads]
        eij = torch.einsum('iho,ho->ih', WhiWhj, self.a) 
        eij = self.leakyrelu(eij)

        # [2708, 2708, num_heads]
        attention = eij.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15) 
        attention[adj_matrix[...,None].repeat(1,1,self.num_heads) == 1] = eij.reshape(-1)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)

        # [2708, 2708, num_heads], [2708, num_heads, output_dim] = [2708, num_heads, output_dim]
        # sum(alpha*W*h_i)
        h_prime = torch.einsum('ijh,jho->iho', attention, Wh) 

        if self.concat:
            return h_prime.reshape(num_nodes, -1)
        else:
            return h_prime.mean(dim=1)

class ModifiedGATLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads=1, concat=True, alpha=0.2, dropout = 0.6):
        """
        Inputs:
            input_dim - Dimensionality of input features
            output_dim - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
            dropout - Dropout parameter which is for overfitting.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        if self.concat:
            assert output_dim % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            output_dim = output_dim // num_heads

        # Create W: [output_dim * num_heads, input_dim]
        # Initialize W by torch.nn.init.xavier_uniform_(tensor, gain=1.414)
        self.W1 = nn.Linear(input_dim, output_dim * num_heads)
        nn.init.xavier_uniform_(self.W1.weight.data, gain=1.414)
        self.W2 = nn.Linear(input_dim, output_dim * num_heads)
        nn.init.xavier_uniform_(self.W2.weight.data, gain=1.414)
        self.W3 = nn.Linear(input_dim, output_dim * num_heads)
        nn.init.xavier_uniform_(self.W3.weight.data, gain=1.414)

        # Create a: [num_heads, 2*output_dim]
        # Initialize a by torch.nn.init.xavier_uniform_(tensor, gain=1.414)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * output_dim))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # active function
        self.leakyrelu = nn.LeakyReLU(alpha)
               

    def forward(self, h, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [num_nodes, num_features] 
            adj_matrix - Adjacency matrix including self-connections. Shape: [num_nodes, num_nodes]
        """

        num_nodes = h.size(0) # 2708

        # Apply linear layer and sort nodes by head
        # [output_dim * num_heads, input_dim]*[2708, 1433] = [2708, output_dim]
        W1h = self.W1(h) # Q
        W2h = self.W2(h) # K
        W3h = self.W3(h) # V

        # [2708, num_heads, output_dim]
        W1h = W1h.view(num_nodes, self.num_heads, -1)
        W2h = W2h.view(num_nodes, self.num_heads, -1)
        W3h = W3h.view(num_nodes, self.num_heads, -1)

        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero(as_tuple=True) #[10556]

        # [10556, num_heads, 2*output_dim], [W*h_i||W*h_j] 
        # Index select returns a tensor with node_feats_flat 
        # being indexed at the desired positions along dim=0
        WhiWhj = torch.cat([
            torch.index_select(input=W1h, index=edges[0], dim=0),
            torch.index_select(input=W2h, index=edges[1], dim=0)
        ], dim=-1)

        # a*[W*h_i||W*h_j] 
        # [10556, num_heads, 2*output_dim], [num_heads, 2*output_dim] = [10556, num_heads]
        eij = torch.einsum('iho,ho->ih', WhiWhj, self.a)
        # LeakyReLU(a*[W*h_i||W*h_j] )
        eij = self.leakyrelu(eij)

        # [2708, 2708, num_heads]
        attention = eij.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15) 
        attention[adj_matrix[...,None].repeat(1,1,self.num_heads) == 1] = eij.reshape(-1)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)

        # [2708, 2708, num_heads], [2708, num_heads, output_dim] = [2708, num_heads, output_dim]
        # sum(alpha*W*h_i)
        h_prime = torch.einsum('ijh,jhc->ihc', attention, W3h) 

        if self.concat:
            return h_prime.reshape(num_nodes, -1)
        else:
            return h_prime.mean(dim=1)

class GCNLayer(nn.Module): 
    
    def __init__(self, input_dim, output_dim):
      super().__init__()
      self.projection1 = nn.Linear(input_dim, output_dim)
      self.projection2 = nn.Linear(input_dim, output_dim, bias = False)

    def forward(self, node_feats, adj_matrix):
      """
      Inputs:
          node_feats - Tensor with node features of shape [num_nodes, c_in]
          adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[i,j]=1 else 0.
                        Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                        Shape: [num_nodes, num_nodes]
      """
      # Num neighbours = number of incoming edges
      # WH
      node_feats_self = self.projection1(node_feats)
      # A^WH
      node_feats_neig = self.projection2(node_feats)
      node_feats_neig = torch.mm(adj_matrix, node_feats_neig)
      return node_feats_self + node_feats_neig
    
      #num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
      #node_feats = self.projection(node_feats)
      #node_feats = torch.mm(adj_matrix, node_feats)
      #node_feats = node_feats / num_neighbours

      #return node_feats