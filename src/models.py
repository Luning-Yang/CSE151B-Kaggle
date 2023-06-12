import torch
import torch.nn as nn
import torch.nn.functional as F
    
# Model 1
class MLP_1(nn.Module):
    """ Multilayer Perceptron. """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.05)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.05)
        x = self.fc3(x)
        return x
    
    
# Model 2
class MLP_2(nn.Module):
    """ Multilayer Perceptron. """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.node_encoder = nn.Sequential(
                    nn.Linear(input_dim, input_dim + hidden_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(input_dim + hidden_dim, hidden_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.node_encoder(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.05)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.05)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.05)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.1)
        x = self.fc5(x)
        return x
    
# Model 3
class rnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
class CNN(nn.Module):
    # Batchnorm is commented out because we found adding it make the test performance worse
    def __init__(self):
        super().__init__()
        print("test")
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(3, 3, 5)
        self.fc1 = nn.Linear(378, 380)
        #self.bn = nn.BatchNorm1d(380)
        self.fc2 = nn.Linear(380, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.bn(x)
        #x = F.dropout(x, p=0.01)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.01)
        x = F.relu(self.fc3(x))
        #x = F.dropout(x, p=0.01)
        x = F.relu(self.fc4(x))
        #x = F.dropout(x, p=0.05)
        x = self.fc5(x)
        return x