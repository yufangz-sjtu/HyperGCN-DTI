import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
#from torch.nn import Module, Parameter
from torch.nn import Linear


class GraphConvolution(Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class GraphAttentionLayer(Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation for input features
        self.linear = Linear(in_features, out_features, bias=bias)
        
        # Attention parameters
        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.linear.reset_parameters()

    def forward(self, input, adj):
        # Step 1: Apply linear transformation
        h = self.linear(input)
        N = h.size()[0]  # Number of nodes
        
        # Step 2: Compute attention coefficients
        # Concatenate features for attention
        h_expanded = h.repeat(1, N).view(N, N, -1)
        a_input = torch.cat([h_expanded, h_expanded.transpose(0, 1)], dim=2)  # (N, N, 2 * out_features)
        
        # Compute attention scores
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))  # (N, N)
        
        # Step 3: Apply attention scores to adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)  # For masking
        attention = torch.where(adj > 0, e, zero_vec)  # Mask with adjacency
        
        # Step 4: Compute softmax to get attention weights
        attention = F.softmax(attention, dim=1)  # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # Dropout on attention
        
        # Step 5: Aggregate features using attention weights
        output = torch.matmul(attention, h)  # (N, out_features)
        
        if self.bias is not None:
            output += self.bias
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class SGCLayer(Module):
    """
    Simplified GCN Layer (SGC)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SGCLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Apply the linear transformation
        support = torch.mm(input, self.weight)
        # Perform the graph convolution
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'





