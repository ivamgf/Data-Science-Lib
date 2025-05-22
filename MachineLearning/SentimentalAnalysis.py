# Sentimental Analysis Algorithm - Algoritmo de Análise de Sentimento

# === IMPORTAÇÕES ===
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import re
import webbrowser
import pickle

# Downloads necessários para o NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')


# === 1. PRÉ-PROCESSAMENTO DE TEXTO ===
def preprocess_text(text):
    # Converte para minúsculas, remove pontuação, tokeniza, remove stopwords e aplica stemmer
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return tokens


# === 2. CARGA E TRATAMENTO DOS DADOS ===
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    df.dropna(inplace=True)
    df['tokens'] = df['texto'].apply(preprocess_text)  # Aplica o pré-processamento
    return df

# Cria vocabulário baseado na frequência mínima dos tokens
def build_vocab(token_lists, min_freq=1):
    freq = {}
    for tokens in token_lists:
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
    vocab = {word: i + 1 for i, (word, count) in enumerate(freq.items()) if count >= min_freq}
    vocab['<UNK>'] = 0  # Token desconhecido
    return vocab

# Codifica uma lista de tokens para uma sequência de índices do vocabulário
def encode_tokens(tokens, vocab, max_len=50):
    indices = [vocab.get(t, 0) for t in tokens]
    return indices[:max_len] + [0] * max(0, max_len - len(indices))


# === 3. CLASSE DE DATASET PERSONALIZADA ===
class TextDataset(Dataset):
    def __init__(self, df, vocab, label_encoder):
        self.inputs = [encode_tokens(t, vocab) for t in df['tokens']]
        self.labels = label_encoder.transform(df['sentimento'])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])


# === 4. DEFINIÇÃO DO MODELO ===
# Rede neural feedforward simples (MLP — Perceptron Multicamadas)
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)  # Média dos embeddings (pooling)
        dropped = self.dropout(pooled)
        return self.fc(dropped)


# === 5. TREINAMENTO DO MODELO ===
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, label_encoder):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validação
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                outputs = model(x_val)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_val.numpy())

        epoch_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {epoch_acc:.2f}")
        print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

    # Relatório final
    final_report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    final_accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    html_report = classification_report_to_html(final_report, label_encoder.classes_, final_accuracy, conf_matrix)

    html_path = 'resultado_validacao.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)

    print(f'Relatório final salvo em {html_path}')
    webbrowser.open(f'file://{os.path.abspath(html_path)}')

    # Salva modelo treinado
    torch.save(model.state_dict(), 'modelo_treinado.pth')
    print("Modelo salvo como modelo_treinado.pth")


# === CONVERSÃO DE RELATÓRIO PARA HTML ===
def classification_report_to_html(report_dict, class_names, accuracy, conf_matrix):
    html = f"""
    <html><head><title>Relatório de Validação</title></head>
    <body>
    <h1>Relatório de Classificação</h1>
    <h2>Acurácia Final: {accuracy:.2f}</h2>

    <h3>Métricas por Classe</h3>
    <table border="1" cellpadding="8">
    <tr><th>Classe</th><th>Precisão</th><th>Recall</th><th>F1-score</th><th>Suporte</th></tr>
    """
    for label in class_names:
        metrics = report_dict[label]
        html += f"<tr><td>{label}</td><td>{metrics['precision']:.2f}</td><td>{metrics['recall']:.2f}</td><td>{metrics['f1-score']:.2f}</td><td>{metrics['support']}</td></tr>"
    html += "</table>"

    # Matriz de confusão
    html += "<h3>Matriz de Confusão</h3><table border='1' cellpadding='8'><tr><th></th>"
    for label in class_names:
        html += f"<th>{label}</th>"
    html += "</tr>"
    for i, row in enumerate(conf_matrix):
        html += f"<tr><th>{class_names[i]}</th>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</table>"

    html += "</body></html>"
    return html


# === 6. EXECUÇÃO PRINCIPAL ===
def main():
    filepath = '../Data/posts.csv'  # Caminho para seu dataset
    df = load_data(filepath)

    vocab = build_vocab(df['tokens'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['sentimento'])

    dataset = TextDataset(df, vocab, label_encoder)

    # Divisão simples: primeiros 16 para treino, 4 para validação
    train_indices = list(range(16))
    val_indices = list(range(16, 20))

    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=len(label_encoder.classes_),
        dropout=0.4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=32, device=device,
                label_encoder=label_encoder)

    # Salva vocabulário
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # Salva codificador de labels
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Vocabulário salvo como vocab.pkl")
    print("LabelEncoder salvo como label_encoder.pkl")


if __name__ == "__main__":
    main()