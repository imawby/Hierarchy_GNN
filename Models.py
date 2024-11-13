import torch
from torch_geometric.nn import SAGEConv

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate

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
        
def PrimaryTierGroupModel_tracks(nVariables, dropoutRate=0.5):
    
    networkInputs_0 = Input(shape=(nVariables, ))
    networkInputs_1 = Input(shape=(nVariables, ))
    
    orientationBranch_0 = OrientationBranch(networkInputs_0, dropoutRate)
    orientationBranch_1 = OrientationBranch(networkInputs_1, dropoutRate)
    
    prediction_0 = Dense(3, activation='softmax', name="orientation_0")(orientationBranch_0)
    prediction_1 = Dense(3, activation='softmax', name="orientation_1")(orientationBranch_1)
    
    combinedBranches = AllBranches_tracks(prediction_0, prediction_1, dropoutRate)
    prediction = Dense(1, activation='sigmoid', name="final_prediction")(combinedBranches)
    
    model = Model(inputs=[networkInputs_0, networkInputs_1], outputs=[prediction_0, prediction_1, prediction])
    
    return model


##########################################################################################################################
##########################################################################################################################

def OrientationBranch(branchInputs, dropoutRate):
    ################################
    # Start branch
    ################################
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(branchInputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    return x

##########################################################################################################################
##########################################################################################################################

def AllBranches_tracks(orientationBranch_0, orientationBranch_1, dropoutRate):

    x = Concatenate()([orientationBranch_0, orientationBranch_1])
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    return x