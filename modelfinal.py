# This is the complete paper code, containing only the model and the cropping part.
# For execution, please write the training function in combination with the data.
import torch
import torch.nn as nn
import torch.optim as optim

class STCL(nn.Module):
    def __init__(self, in_features, out_features, conv_in_channels, conv_out_channels):
        super(STCL, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=(1, 1))  # 1x1卷积核

    def forward(self, x, adj):
        batch_size, num_channels, num_nodes, _ = x.size()
        adj_normalized = self.normalize_adj(adj, num_nodes)
        x = x.view(batch_size * num_channels, num_nodes, -1)
        out = torch.bmm(adj_normalized.unsqueeze(0).expand(batch_size * num_channels, -1, -1), x)
        out = self.fc(out)
        out = out.view(batch_size, num_channels, num_nodes, -1)
        out = self.conv(out)
        return out

    def normalize_adj(self, adj, num_nodes):
        adj = adj + torch.eye(num_nodes, device=adj.device)
        degree = torch.sum(adj, dim=1)+1e-6
        degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        adj_normalized = torch.mm(degree_inv_sqrt, torch.mm(adj, degree_inv_sqrt))
        return adj_normalized


class MultiLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_nodes, num_layers, conv_in_channels,
                 conv_hid_channels, conv_out_channels):
        super(MultiLayerGCN, self).__init__()
        self.adj_matrix = nn.Parameter(torch.ones(num_nodes, num_nodes), requires_grad=True)
        self.layers = nn.ModuleList()
        self.layers.append(STCL(in_features, hidden_features, conv_in_channels, conv_hid_channels))
        for _ in range(num_layers - 2):
            self.layers.append(STCL(hidden_features, hidden_features, conv_hid_channels, conv_hid_channels))
        self.layers.append(STCL(hidden_features, out_features, conv_hid_channels, conv_out_channels))
        self.fc = nn.Linear(out_features * conv_out_channels * num_nodes, 1)

    def forward(self, x, yt_1):
        batch_size, _, _, _ = x.size()
        for layer in self.layers:
            x = layer(x, self.adj_matrix)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return yt_1 + x

class Cropping(nn.Module):
    def __init__(self, window_size=2, step_size=1):
        super(Cropping, self).__init__()
        self.window_size = window_size
        self.step_size = step_size

    def forward(self, x):
        batch_size, num_channels, num_nodes, in_features = x.size()
        new_in_features = in_features - self.window_size + 1
        windows = []

        for i in range(0, new_in_features, self.step_size):
            if i + self.window_size <= in_features:
                window = x[..., i:i + self.window_size]
                windows.append(window)

        windows = torch.cat(windows, dim=2)
        return windows

in_features = 2
hidden_features = 32
out_features = 64
num_nodes = 15
learning_rate = 0.01
num_epochs = 200
num_layers = 5
conv_in_channels = 1
conv_hid_channels = 128
conv_out_channels = 1
batch_size = 32

x = Cropping()(torch.randn(batch_size, conv_in_channels, 5, 4))
yt_1 = torch.randn(batch_size, 1)
yt = torch.randn(batch_size, 1)

model = MultiLayerGCN(in_features, hidden_features, out_features, num_nodes, num_layers, conv_in_channels,
                      conv_hid_channels, conv_out_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x, yt_1)
    loss = criterion(output, yt)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training complete.")




