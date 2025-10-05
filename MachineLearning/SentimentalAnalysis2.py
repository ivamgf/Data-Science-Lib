import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# ===============================
# 1. Pré-processamento de texto
# ===============================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

# ===============================
# 2. Dataset
# ===============================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.inputs = [self.encode(preprocess(text), vocab, max_len) for text in texts]
        self.labels = labels

    def encode(self, tokens, vocab, max_len):
        indices = [vocab.get(t, 0) for t in tokens]
        return torch.tensor(indices[:max_len] + [0]*(max_len - len(indices)), dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

# ===============================
# 3. Modelo LSTM
# ===============================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# ===============================
# 4. Treinamento
# ===============================
def train_model(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ===============================
# 5. Execução principal
# ===============================
def main():
    filepath = '../Data/posts.csv'
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    # === Ler CSV ===
    df = pd.read_csv(filepath)
    if 'texto' not in df.columns or 'sentimento' not in df.columns:
        print("O arquivo CSV deve conter as colunas 'texto' e 'sentimento'")
        return

    # === Codificação dos rótulos ===
    le = LabelEncoder()
    labels = le.fit_transform(df['sentimento'])

    # === Vocabulário ===
    all_tokens = [token for text in df['texto'] for token in preprocess(text)]
    vocab = {token: i+1 for i, token in enumerate(set(all_tokens))}
    vocab['<UNK>'] = 0

    # === Divisão treino/teste ===
    X_train, X_test, y_train, y_test = train_test_split(df['texto'], labels, test_size=0.2, random_state=42)
    train_dataset = TextDataset(X_train.tolist(), y_train, vocab)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # === Modelo e treinamento ===
    model = SentimentLSTM(vocab_size=len(vocab), embed_dim=50, hidden_dim=64, num_classes=len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, train_loader, optimizer, criterion, epochs=5)

    # === Salvar modelo ===
    torch.save(model.state_dict(), "modelo_treinado.pth")
    print("Modelo salvo como modelo_treinado.pth")

if __name__ == "__main__":
    main()
