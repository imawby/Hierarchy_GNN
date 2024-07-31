import torch
from torch_geometric.nn import SAGEConv

# To calculate node embeddings
class GNN(torch.nn.Module):
    
    def __init__(self, node_feautres, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(node_feautres, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = func.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

# Our edge classifier applies the dot-product between source and destination
class EdgeClassifier(torch.nn.Module):
    
    def forward(self, x, edge_index) :
        # Convert node embeddings to edge-level representations:
        source_node_features = x[edge_index[0]]
        target_node_features = x[edge_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        dot_product = (source_node_features * target_node_features).sum(dim=-1)
        return torch.sigmoid(dot_product)
    
# Our node classifier
class NodeClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x) :
        x = self.fc(x)
        return x

#class Model(torch.nn.Module):
#    
#    def __init__(self, node_feautres, hidden_channels):
#        super().__init__()
#        # Instantiate homogeneous GNN:
#        self.gnn = GNN(node_feautres, hidden_channels)
#        self.classifier = Classifier()
        
#    def forward(self, x, edge_index):
#        gnnModel = self.gnn(x, edge_index)
#        pred = self.classifier(x, edge_index)
#        return pred
        