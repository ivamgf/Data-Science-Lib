# -*- coding: utf-8 -*-
# Sentiment analysis com NLTK + Word2Vec (gensim) + LSTM (PyTorch)

import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Configurações ----------------
config = {
    "learning_rate": 0.001,
    "dropout": 0.3,
    "epochs": 50,
    "batch_size": 2,
    "lstm_units": 128,
    "embedding_dim": 100,
    "max_len": 20,
    "test_size": 0.2,
    "random_state": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ---------------- Corpus ----------------
corpus = [
    "Eu amei o filme, foi incrível e emocionante!",
    "Horrível, o pior filme que já vi na vida.",
    "Muito bom — atuação sensacional e ótima direção.",
    "Não gostei, roteiro fraco e atuações medianas.",
    "Excelente trama, recomendo para todos.",
    "O filme é decente, mas poderia ser melhor.",
    "Péssimo, perdemos nosso tempo assistindo.",
    "Adorei! Vale muito a pena ver no cinema.",
    "É ruim, não recomendo a ninguém.",
    "Bastante divertido e comédia inteligente."
]
labels = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])

# ---------------- Pré-processamento ----------------
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()


def preprocess(texts):
    processed = []
    for t in texts:
        t = t.lower()
        t = re.sub(r"http\S+|www\S+|https\S+", " ", t)
        t = re.sub(r"[^a-zà-ú\s]", " ", t)
        tokens = word_tokenize(t, language="portuguese")
        tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]
        processed.append(tokens)
    return processed


processed_tokens = preprocess(corpus)

# ---------------- Treinar Word2Vec ----------------
w2v_model = Word2Vec(sentences=processed_tokens, vector_size=config["embedding_dim"],
                     window=5, min_count=1, workers=4, sg=1)

# ---------------- Vocabulário ----------------
vocab = {word: i + 1 for i, word in enumerate(w2v_model.wv.index_to_key)}
vocab_size = len(vocab) + 1  # +1 para padding


# ---------------- Converter tokens para índices ----------------
def tokens_to_indices(tokens, vocab, max_len):
    seq = [vocab.get(tok, 0) for tok in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


X_indices = [tokens_to_indices(toks, vocab, config["max_len"]) for toks in processed_tokens]
X = torch.tensor(X_indices, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# ---------------- Criar embeddings do PyTorch ----------------
embedding_matrix = np.zeros((vocab_size, config["embedding_dim"]))
for word, idx in vocab.items():
    embedding_matrix[idx] = w2v_model.wv[word]
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

# ---------------- Dataset e DataLoader ----------------
dataset = TensorDataset(X, y)
test_size = int(config["test_size"] * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])


# ---------------- Modelo LSTM ----------------
class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, lstm_units, dropout):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


model = SentimentLSTM(embedding_matrix, config["lstm_units"], config["dropout"]).to(config["device"])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# ---------------- Treinamento ----------------
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(config["device"]), yb.to(config["device"])
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {total_loss / len(train_loader):.4f}")

# ---------------- Teste ----------------
from sklearn.metrics import precision_score, f1_score, recall_score

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(config["device"]), yb.to(config["device"])
        out = model(xb)
        preds = (out >= 0.5).float()
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n=== Resultados no conjunto de teste ===")
print("Acurácia:", accuracy_score(y_true, y_pred))
print("Precisão:", precision_score(y_true, y_pred, zero_division=1))
print("Recall:", recall_score(y_true, y_pred, zero_division=1))
print("F1-Score:", f1_score(y_true, y_pred, zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ---------------- Predição ----------------
def predict_sentiment(text):
    tokens = preprocess([text])
    seq = torch.tensor([tokens_to_indices(tokens[0], vocab, config["max_len"])], dtype=torch.long).to(config["device"])
    model.eval()
    with torch.no_grad():
        prob = model(seq).item()
    return ("Positivo" if prob >= 0.5 else "Negativo", prob)


novo_texto = "Achei o filme muito ruim e cansativo."
sentimento, prob = predict_sentiment(novo_texto)
print(f"\nTexto: {novo_texto}")
print(f"Classificação: {sentimento} (probabilidade={prob:.4f})")
