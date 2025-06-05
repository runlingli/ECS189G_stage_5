'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from base_class.method import method
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time
from Training_Visualization import Training_Visualizer

max_epochs = 220
learning_rate = 0.008

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class Method_GCN_Node_Classifier(method):
    def __init__(self, mName=None, mDescription=None):
        super(Method_GCN_Node_Classifier, self).__init__(mName, mDescription)
        self.visualizer = Training_Visualizer(save_dir='./')

    def set_data(self, graph, train_test):
        self.features = graph['X']
        self.labels = graph['y']
        self.edge_index = graph['utility']['A'].coalesce().indices()  # sparse -> edge_index
        self.idx_train = train_test['idx_train']
        self.idx_test = train_test['idx_test']

    def run(self):
        # initialize the model
        in_channels = self.features.shape[1]
        out_channels = int(self.labels.max()) + 1
        model = GCN(in_channels, 32, out_channels)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

        # training process
        train_losses = []

        model.train()
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            out = model(self.features, self.edge_index)
            loss = F.cross_entropy(out[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            # record the loss
            train_losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        # training visualization
        self.visualizer.plot_training_progress(
            train_losses, 
            title="Pubmed dataset GCN Training"
        )
        print("Training visualization saved to ./pubmed_gcn_training.png")

        model.eval()
        out = model(self.features, self.edge_index)
        pred = out.argmax(dim=1)
        self.predicted_labels = pred
        return pred[self.idx_test], self.labels[self.idx_test]
