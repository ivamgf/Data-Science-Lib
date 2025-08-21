import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
import nltk
import os

# Downloads necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# === Função de pré-processamento ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return tokens

# === Codificação de tokens para índices ===
def encode_tokens(tokens, vocab, max_len=50):
    indices = [vocab.get(t, 0) for t in tokens]
    return indices[:max_len] + [0] * max(0, max_len - len(indices))

# === Modelo LSTM (igual ao usado no treino) ===
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

# === Verificações e carregamento dos arquivos salvos ===
if not os.path.exists("modelo_sentimentos.pth"):
    print("Arquivo do modelo 'modelo_sentimentos.pth' não encontrado.")
    exit()

if not os.path.exists("vocab.pkl") or not os.path.exists("label_encoder.pkl"):
    print("Arquivos 'vocab.pkl' ou 'label_encoder.pkl' não encontrados.")
    exit()

# === Carregamento do vocabulário e do codificador de rótulos ===
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Parâmetros devem bater com o modelo treinado ===
embed_dim = 50
hidden_dim = 64
num_classes = len(label_encoder.classes_)
vocab_size = len(vocab)

# === Dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Inicializa e carrega o modelo ===
model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load("modelo_sentimentos.pth", map_location=device))
model.to(device)
model.eval()

# === Entrada do usuário ===
user_text = input("Digite um texto para análise de sentimento: ")

# === Pré-processamento e codificação ===
tokens = preprocess_text(user_text)
encoded = encode_tokens(tokens, vocab)
input_tensor = torch.tensor([encoded]).to(device)

# === Predição ===
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    predicted_label = label_encoder.inverse_transform([prediction])[0]

# === Resultado ===
print(f"Sentimento previsto: {predicted_label}")
