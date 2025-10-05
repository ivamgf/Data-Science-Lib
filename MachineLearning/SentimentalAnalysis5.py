# Sentimental Analysis

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
import os

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# ---------------- Configurações ----------------
config = {
    "learning_rate": 0.001,
    "dropout": 0.3,
    "epochs": 12,
    "batch_size": 2,
    "lstm_units": 128,
    "embedding_dim": 100,
    "max_len": 20,
    "test_size": 0.2,
    "random_state": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "early_stopping_patience": 5  # parar se não melhorar por X épocas
}

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
# 2. Balanceamento de classes
# ===============================
def balance_dataset(df, target_col='sentimento'):
    dfs = []
    max_count = df[target_col].value_counts().max()
    for label in df[target_col].unique():
        df_label = df[df[target_col] == label]
        if len(df_label) == 0:
            print(f"Atenção: Classe '{label}' está ausente no dataset e será ignorada.")
            continue
        if len(df_label) < max_count:
            df_label = resample(df_label, replace=True, n_samples=max_count, random_state=42)
        dfs.append(df_label)
    if not dfs:
        raise ValueError("Nenhuma classe válida encontrada para balanceamento.")
    df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

# ===============================
# 3. Dataset
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
# 4. Modelo LSTM
# ===============================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)

# ===============================
# 5. Treinamento
# ===============================
def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    device = config["device"]
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output_val = model(x_val)
                loss_val = criterion(output_val, y_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "melhor_modelo.pth")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print("Early stopping ativado!")
                break

# ===============================
# 6. Avaliação
# ===============================
def evaluate_model(model, dataloader, device, le):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = [int(y) for y in y_true]
    y_pred = [int(y) for y in y_pred]

    print("\n=== Resultados no conjunto de teste ===")
    print(f"Acurácia:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão:  {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    labels_present = sorted(list(set(y_true) | set(y_pred)))
    target_names = [str(le.classes_[i]) for i in labels_present]

    print("\nRelatório detalhado:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=target_names,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    print("\nMatriz de confusão:")
    print(cm)

# ===============================
# 7. Execução principal
# ===============================
def main():
    filepath = '../Data/Lista_de_frases.csv'
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    df = pd.read_csv(filepath)
    if 'texto' not in df.columns or 'sentimento' not in df.columns:
        print("O arquivo CSV deve conter as colunas 'texto' e 'sentimento'")
        return

    # Balancear dataset
    df = balance_dataset(df, target_col='sentimento')

    # Label encoding
    le = LabelEncoder()
    labels = le.fit_transform(df['sentimento'])

    # Vocabulário
    all_tokens = [token for text in df['texto'] for token in preprocess(text)]
    vocab = {token: i+1 for i, token in enumerate(set(all_tokens))}
    vocab['<UNK>'] = 0

    # Divisão treino/teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        df['texto'], labels,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=labels
    )

    # Dataset e DataLoader
    train_dataset = TextDataset(X_train.tolist(), y_train, vocab, max_len=config["max_len"])
    test_dataset = TextDataset(X_test.tolist(), y_test, vocab, max_len=config["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Modelo
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=config["embedding_dim"],
        hidden_dim=config["lstm_units"],
        num_classes=len(le.classes_),
        dropout=config["dropout"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Treinamento com early stopping
    train_model(model, train_loader, test_loader, optimizer, criterion, config)

    # Avaliação final
    evaluate_model(model, test_loader, config["device"], le)

    print("\nModelo salvo como melhor_modelo.pth")

if __name__ == "__main__":
    main()
