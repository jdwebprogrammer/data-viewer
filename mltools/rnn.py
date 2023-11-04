import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# Define the character set (e.g., lowercase letters)
characters = string.ascii_lowercase + ' '

# Create a mapping of characters to indices and vice versa
char_to_index = {char: i for i, char in enumerate(characters)}
index_to_char = {i: char for i, char in enumerate(characters)}

# Generate a dataset of random sequences
def generate_data(num_samples, sequence_length):
    data = []
    for _ in range(num_samples):
        sequence = ''.join(random.choice(characters) for _ in range(sequence_length))
        data.append(sequence)
    return data

# Convert sequences to tensors
def sequences_to_tensor(sequences):
    tensor = torch.zeros(len(sequences), len(sequences[0]), len(characters))
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            tensor[i, t, char_to_index[char]] = 1
    return tensor

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Define a simple RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # Set batch_first to True
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


# Define a feedforward neural network
class FeedForwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Combine the LSTM and Feedforward models
class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_output_size, ff_input_size, ff_hidden_size, ff_output_size):
        super(CombinedModel, self).__init__()
        self.lstm_model = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_output_size)
        self.ff_model = FeedForwardModel(ff_input_size, ff_hidden_size, ff_output_size)

    def forward(self, lstm_input, ff_input):
        lstm_output = self.lstm_model(lstm_input)
        ff_output = self.ff_model(ff_input)
        # You can combine the outputs or perform any other operation here
        return lstm_output, ff_output

# Example usage
lstm_input_size = 10
lstm_hidden_size = 128
lstm_output_size = 10
ff_input_size = 10
ff_hidden_size = 64
ff_output_size = 2
combined_model = CombinedModel(lstm_input_size, lstm_hidden_size, lstm_output_size, ff_input_size, ff_hidden_size, ff_output_size)


# Training parameters
input_size = len(characters)
hidden_size = 128
output_size = len(characters)
learning_rate = 0.001
num_epochs = 1000
batch_size = 32

# Create the RNN model
model = RNN(input_size, hidden_size, output_size)

# Create the LSTM model
lstm_model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for sequence generation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate training data
num_samples = 1000
sequence_length = 10
data = generate_data(num_samples, sequence_length)
data_tensor = sequences_to_tensor(data)

# Modify the training loop to match the batch size
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Generate random batch
    batch_indices = torch.randint(0, len(data), (batch_size,))
    batch_input = data_tensor[batch_indices]
    
    # Initialize hidden state for the current batch
    hidden = torch.zeros(1, batch_size, hidden_size)
    # Initialize hidden state for the current batch
    #hidden = (torch.zeros(1, batch_size, hidden_size), torch.zeros(1, batch_size, hidden_size))
    

    # Initialize loss for the current batch
    batch_loss = 0
    
    # Iterate over each time step in the sequence
    for t in range(sequence_length - 1):
        # Forward pass for one time step
        output, hidden = model(batch_input[:, t:t + 1], hidden)
        
        # Calculate loss for this time step
        target = batch_input[:, t + 1]  # Set the target to the next character
        #loss = criterion(output.squeeze(1), target.long())  # Use .long() for target
        loss = criterion(output.squeeze(1), target)

        
        # Accumulate the loss for the entire batch
        batch_loss += loss
    
    # Backpropagation
    batch_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {batch_loss.item()}')

# Generate a sample sequence from the trained model
sample_length = 20
sample_input = torch.zeros(1, sample_length, len(characters))
sample_input[0, 0, char_to_index['a']] = 1  # Starting character 'a'
hidden = torch.zeros(1, 1, hidden_size)
output_seq = ['a']

for t in range(1, sample_length):
    output, hidden = model(sample_input[:, t-1:t], hidden)
    char_probs = torch.softmax(output.squeeze(1), dim=1)  # Apply softmax
    char_index = torch.multinomial(char_probs, 1).item()  # Sample next character
    output_char = index_to_char[char_index]
    output_seq.append(output_char)

generated_sequence = ''.join(output_seq)
print(f'Generated Sequence: {generated_sequence}')