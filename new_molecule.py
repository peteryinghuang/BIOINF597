import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from collections import Counter
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

# Define the dataset URL
url = "https://github.com/peteryinghuang/BIOINF597/raw/main/qm9.csv.gz"

# Load the dataset
qm9_data = pd.read_csv(url, compression='gzip')

# Show the first 5 rows of the data
print(qm9_data.head())

from sklearn.model_selection import train_test_split

# Pad SMILES strings to maximum length
qm9_data['smiles_padded'] = qm9_data['smiles'].apply(lambda x: x.ljust(29))

# Split the dataset into training and validation sets
train_data, validate_data = train_test_split(qm9_data, test_size=0.2, random_state=42)

# Display the sizes of the divided datasets
print(f"Train dataset size: {train_data.shape[0]}")
print(f"Validation dataset size: {validate_data.shape[0]}")

# Check the padding results
print(train_data[['smiles', 'smiles_padded']].sample(50))

class SmilesTokenizer(object):
    def __init__(self):
        self.regex_pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\."
            r"|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, smiles):
        return [token for token in self.regex.findall(smiles)]

def build_vocab(smiles_list, tokenizer, max_vocab_size=50):
    tokenized_smiles = [tokenizer.tokenize(s) for s in smiles_list]
    token_counter = Counter(token for tokens in tokenized_smiles for token in tokens)
    most_common_tokens = [token for token, _ in token_counter.most_common(max_vocab_size)]
    vocab = {token: idx for idx, token in enumerate(most_common_tokens)}
    return vocab

def smiles_to_indices(smiles, tokenizer, vocab):
    unknown_token_id = len(vocab) 
    token_ids = [vocab.get(token, unknown_token_id) for token in tokenizer.tokenize(smiles)]
    return torch.tensor(token_ids, dtype=torch.long)  # Return indices as a tensor of type long

# Instantiate tokenizer
tokenizer = SmilesTokenizer()

# Build vocabulary
train_smiles = train_data['smiles_padded'].tolist()
validate_smiles = validate_data['smiles_padded'].tolist()
vocab = build_vocab(train_smiles, tokenizer)
vocab_size = len(vocab)

# Convert SMILES to indices
train_indices = [smiles_to_indices(smi, tokenizer, vocab) for smi in train_smiles]
validate_indices = [smiles_to_indices(smi, tokenizer, vocab) for smi in validate_smiles]

from torch.nn.utils.rnn import pad_sequence

max_length = 29  

train_indices_padded = pad_sequence(train_indices, batch_first=True, padding_value=len(vocab)).to(device)
validate_indices_padded = pad_sequence(validate_indices, batch_first=True, padding_value=len(vocab)).to(device)

if train_indices_padded.shape[1] < max_length:
    train_indices_padded = F.pad(train_indices_padded, (0, max_length - train_indices_padded.shape[1]), "constant", len(vocab))
if validate_indices_padded.shape[1] < max_length:
    validate_indices_padded = F.pad(validate_indices_padded, (0, max_length - validate_indices_padded.shape[1]), "constant", len(vocab))

train_loader = DataLoader(train_indices_padded, batch_size=64, shuffle=True)
validate_loader = DataLoader(validate_indices_padded, batch_size=64, shuffle=False)

class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_dim, max_length):
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # 为LSTM定义嵌入层
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2) 
        )
        
        # 解码器 LSTM
        self.decoder_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_output = nn.Linear(hidden_size, vocab_size + 1)  

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.embedding(x)
        h = self.encoder(x.mean(dim=1))  # Mean as pooling
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var).unsqueeze(1)
        z = z.repeat(1, self.max_length, 1)
        lstm_out, _ = self.decoder_rnn(z)
        recon_x = self.decoder_output(lstm_out)
        return F.log_softmax(recon_x, dim=-1), mu, log_var  # Change activation to log_softmax

# Model, optimizer, and loss
model = VAE(vocab_size, hidden_size=256, latent_dim=32, max_length=29).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def loss_function(recon_x, x, mu, log_var):
    # Flatten only the output dimensions, not the sequence length
    NLL = F.nll_loss(recon_x.transpose(1, 2), x, reduction='sum')  # Adjusted to handle sequence length properly
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return NLL + KLD   

# Check if GPU is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))
    return average_loss

def validate(model, device, validate_loader):
    model.eval()  # Switch to evaluation mode
    validate_loss = 0
    with torch.no_grad():  # No gradient computation
        for data in validate_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            validate_loss += loss.item()
    average_validate_loss = validate_loss / len(validate_loader.dataset)
    return average_validate_loss

losses = []
# Define the number of epochs
num_epoch = 50

# Call the training function and pass the device
#train(epoch, model, device, train_loader, optimizer)
# Call the training function for each epoch
start_time = time.time()

def train_and_validate(num_epochs, model, device, train_loader, validate_loader, optimizer):
    train_losses = []
    validate_losses = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        average_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(average_train_loss)

        validate_loss = validate(model, device, validate_loader)
        validate_losses.append(validate_loss)
        
        print(f'Epoch: {epoch}, Training Loss: {average_train_loss:.4f}, Validation Loss: {validate_loss:.4f}')
    
    return train_losses, validate_losses

# Call the train and validate function
train_losses, validate_losses = train_and_validate(num_epoch, model, device, train_loader, validate_loader, optimizer)

end_time = time.time()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds')

torch.save(model.state_dict(), 'model_parameters.pth')

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-')
plt.plot(validate_losses, label='Validation Loss', marker='o', linestyle='-')
plt.title('Training & Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

import torch
from rdkit import Chem
from rdkit.Chem import QED

def generate_molecule(model, device, vocab, inv_vocab, num_samples=1, max_length=29):
    model.eval()
    smiles = []
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        for i in range(num_samples):
            hidden = model.reparameterize(z[i], torch.zeros_like(z[i])).unsqueeze(0).repeat(max_length, 1, 1).transpose(0, 1)
            lstm_out, _ = model.decoder_rnn(hidden)
            recon_x = model.decoder_output(lstm_out)
            generated_smiles_indices = torch.argmax(recon_x, dim=2)
            decoded_smiles = ''.join(inv_vocab.get(idx.item(), '') for idx in generated_smiles_indices[0])
            smiles.append(decoded_smiles.strip()) 
    return smiles

def evaluate_molecules(smiles_list, training_data):
    valid_smiles = []
    validity = 0
    novelty = 0
    qed_scores = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:  # Check if the molecule is valid
            validity += 1
            valid_smiles.append(smi)
            qed_scores.append(QED.qed(mol))
            if smi not in training_data:
                novelty += 1

    validity_score = validity / len(smiles_list) if smiles_list else 0
    novelty_score = novelty / len(valid_smiles) if valid_smiles else 0
    average_qed = sum(qed_scores) / len(qed_scores) if qed_scores else 0

    return validity_score, novelty_score, average_qed, valid_smiles

# Define the reverse vocabulary for decoding indices to SMILES characters
inv_vocab = {v: k for k, v in vocab.items()}  

# Generate new molecules
num_samples = 10
generated_smiles = generate_molecule(model, device, vocab, inv_vocab, num_samples=num_samples)

# Evaluate the generated molecules
train_smiles_set = set(train_data['smiles'].tolist())  # Use set for faster lookup
validity_score, novelty_score, average_qed, valid_smiles = evaluate_molecules(generated_smiles, train_smiles_set)

print(f"Validity: {validity_score}")
print(f"Novelty: {novelty_score}")
print(f"Average QED: {average_qed}")
print("Valid SMILES:", valid_smiles)
