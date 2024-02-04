from neuralNetwork import neuralNetwork

# number of inputs, hidden, output nodes and learning rate
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)