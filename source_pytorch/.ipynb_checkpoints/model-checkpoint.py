# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        #         self.embedding = nn.Embedding(output_dim, input_features, padding_idx=0)
        #         self.lstm = nn.LSTM(input_features, hidden_dim)
        #         self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.hid1 = nn.Linear(2, 4)  # 2-(4-4)-1
        self.hid2 = nn.Linear(4, 4)
        self.oupt = nn.Linear(4, 1)
        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.xavier_uniform_(self.oupt.weight)

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        z = self.tanh(self.hid1(x))
        z = self.tanh(self.hid2(z))
        z = self.sigmoid(self.oupt(z))  # necessary
        return z
    