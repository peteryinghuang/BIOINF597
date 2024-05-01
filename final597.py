import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定义文件的URL
url = "https://github.com/peteryinghuang/BIOINF597/raw/main/qm9.csv.gz"

# 使用Pandas的read_csv()函数读取gzip压缩的文件到DataFrame中
qm9_data = pd.read_csv(url, compression='gzip')

# 显示DataFrame的前几行
print(qm9_data.head())

from sklearn.model_selection import train_test_split

# 填充 SMILES 字符串至最大长度
qm9_data['smiles_padded'] = qm9_data['smiles'].apply(lambda x: x.ljust(29))

# 划分数据集为训练集和验证集
train_data, validate_data = train_test_split(qm9_data, test_size=0.2, random_state=42)

# 计算需要保留的数据量（最接近当前大小且为256整数倍的数）
train_size_adjusted = (train_data.shape[0] // 256) * 256
validate_size_adjusted = (validate_data.shape[0] // 256) * 256

# 调整训练集和验证集的大小
train_data = train_data.iloc[:train_size_adjusted]
validate_data = validate_data.iloc[:validate_size_adjusted]

# 显示划分后的数据集大小
print(f"Train dataset size: {train_data.shape[0]}")
print(f"Validation dataset size: {validate_data.shape[0]}")

# 检查一下填充结果
print(train_data[['smiles', 'smiles_padded']].head())

import re
from collections import Counter
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

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

def smiles_to_ohe(smiles, tokenizer, vocab):
    unknown_token_id = len(vocab)  # 为不在词汇表中的符号指定一个ID
    token_ids = [vocab.get(token, unknown_token_id) for token in tokenizer.tokenize(smiles)]
    ohe = torch.eye(len(vocab) + 1)[token_ids]  # 加1因为未知符号
    return ohe

# 实例化分词器
tokenizer = SmilesTokenizer()

# 建立词汇表
train_smiles = train_data['smiles_padded'].tolist()
validate_smiles = validate_data['smiles_padded'].tolist()
vocab = build_vocab(train_smiles, tokenizer)
vocab_size = len(vocab)

# 打印词汇表信息
print(f"Vocabulary size: {len(vocab)}")
print("Sample tokens from vocabulary:", list(vocab.keys())[:10])

# 展示一些分词结果
sample_smiles = train_smiles[:5]
tokenized_samples = [tokenizer.tokenize(smi) for smi in sample_smiles]
print("\nSample tokenization:")
for smi, tokens in zip(sample_smiles, tokenized_samples):
    print(f"{smi} -> {tokens}")
    
# 转换SMILES为独热编码并打包数据
train_ohe = [smiles_to_ohe(smi, tokenizer, vocab) for smi in train_smiles]
validate_ohe = [smiles_to_ohe(smi, tokenizer, vocab) for smi in validate_smiles]

train_ohe_padded = pad_sequence(train_ohe, batch_first=True, padding_value=0)
validate_ohe_padded = pad_sequence(validate_ohe, batch_first=True, padding_value=0)


# 创建数据加载器
train_loader = DataLoader(train_ohe_padded, batch_size=64, shuffle=True)
validate_loader = DataLoader(validate_ohe_padded, batch_size=64, shuffle=False)

# 输出一些独热编码的样本
#sample_ohe = train_ohe[:5]
#print("\nSample One-Hot Encoded data:")
#for smi, ohe in zip(sample_smiles, sample_ohe):
#    print(f"{smi} -> Shape: {ohe.shape}")
#    print(ohe)

# 检查填充后的独热编码
#print("\nPadded One-Hot Encoded data sample:")
#print(train_ohe_padded[0])

# 检测是否有GPU可用，并设置PyTorch设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size + 1, hidden_size),  # 加1是因为未知符号
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2)  # 输出均值和方差
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size + 1),  # 加1是因为未知符号
            nn.Sigmoid()  # 输出概率
        )
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# Define the size of the hidden layer
hidden_size = 256
# Define the dimensionality of the latent space
latent_dim = 2

# 实例化模型并将其发送到设备
model = VAE(vocab_size, hidden_size, latent_dim).to(device)

# 损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

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
    model.eval()  # 切换到评估模式
    validate_loss = 0
    with torch.no_grad():  # 不进行梯度计算
        for data in validate_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            validate_loss += loss.item()
    average_validate_loss = validate_loss / len(validate_loader.dataset)
    return average_validate_loss

losses = []
# Define the number of epochs
num_epoch = 10

# 调用训练函数并传递设备
#train(epoch, model, device, train_loader, optimizer)
# 调用训练函数并传递设备，循环遍历每个epoch
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

# 调用训练与验证函数
train_losses, validate_losses = train_and_validate(num_epoch, model, device, train_loader, validate_loader, optimizer)

end_time = time.time()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds')
# 绘制训练与验证损失图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-')
plt.plot(validate_losses, label='Validation Loss', marker='o', linestyle='-')
plt.title('Training & Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#for epoch in range(1, num_epoch + 1):
#    avg_loss = train(epoch, model, device, train_loader, optimizer)
#    losses.append(avg_loss)
#end_time = time.time()
#print(f'Total training time: {total_time:.2f} seconds')
