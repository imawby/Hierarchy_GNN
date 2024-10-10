#emb_dim=64
import torch
from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MessagePassing

class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim, edge_dim, aggr='add'):
        """Message Passing Neural Network Layer

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        ##########################################
        # MLP `\psi` for computing a message `m_ij`
        ##########################################
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        
        message_input_dim = (2 * emb_dim) + edge_dim # Node will alreay been through some layer
        message_output_dim = emb_dim
        
        self.mlp_msg = Sequential(
            Linear(message_input_dim, message_output_dim), 
            BatchNorm1d(message_output_dim), 
            ReLU(),
            Dropout(0.5),
            Linear(message_output_dim, message_output_dim), 
            BatchNorm1d(message_output_dim), 
            ReLU()
        )
        
        ##########################################
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        ##########################################
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU

        update_input_dim = (2 * emb_dim) # Old and new nodes concatenated together
        update_output_dim = emb_dim
        
        self.mlp_upd = Sequential(
            Linear(update_input_dim, update_output_dim), 
            BatchNorm1d(update_output_dim), 
            ReLU(), 
            Dropout(0.5),
            Linear(update_output_dim, update_output_dim), 
            BatchNorm1d(update_output_dim), 
            ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        """
        The forward pass updates node features via one round of message passing.

        As our MPNNLayer class inherits from the PyG MessagePassing parent class,
        we simply need to call the `propagate()` function which starts the 
        message passing procedure: `message()` -> `aggregate()` -> `update()`.

        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """Step (1) Message

        The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take 
        any arguments that were initially passed to `propagate`. Additionally, 
        we can differentiate destination nodes and source nodes by appending 
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`. 
        
        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    #def aggregate(self, inputs, index):
        """Step (2) Aggregate

        The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        #return scatter_(self.node_dim, index, inputs, reduce=self.aggr)
    
   
        
    def update(self, aggr_out, h):
        """
        Step (3) Update

        The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`, 
        as well as any optional arguments that were initially passed to 
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')   
    

################################################################################################
################################################################################################  
    
# num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
    
class MPNNModel(Module):
    def __init__(self, num_layers, emb_dim, input_dim, edge_dim):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
        """
        super().__init__()
        
        # Linear projection for initial node features
        self.input_linear = Linear(input_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        #self.pool = global_mean_pool
        
        # Linear prediction head
        # dim: d -> out_dim
        #self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.input_linear(data.x) # take us to the embedding dimension (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        #h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)
        #out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        #return out.view(-1)    
    
        return h
    
    
    
    
    
    
    
    
    
    