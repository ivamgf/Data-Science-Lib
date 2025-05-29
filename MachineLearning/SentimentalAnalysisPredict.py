# Sentimental Analysis Predict

# Imports principais para manipulação do modelo, processamento de texto e serialização
import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re
import nltk

# Baixar os recursos do NLTK necessários para tokenização, stopwords e stemming em português
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# === Função para pré-processar o texto ===
def preprocess_text(text):
    # Converte texto para minúsculas
    text = text.lower()
    # Remove caracteres que não são letras ou espaços (pontuação, símbolos, etc)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokeniza o texto em palavras (tokens)
    tokens = word_tokenize(text, language='portuguese')
    # Define stopwords (palavras comuns a serem ignoradas)
    stop_words = set(stopwords.words('portuguese'))
    # Inicializa o stemmer para reduzir palavras à raiz (ex: "correr", "correndo" -> "corr")
    stemmer = RSLPStemmer()
    # Aplica stemming e remove stopwords
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return tokens

# === Função para converter tokens em índices numéricos usando o vocabulário ===
def encode_tokens(tokens, vocab, max_len=50):
    # Para cada token, pega seu índice no vocabulário; se não existir, usa 0 (token desconhecido)
    indices = [vocab.get(t, 0) for t in tokens]
    # Garante tamanho fixo max_len, cortando ou preenchendo com zeros (padding)
    return indices[:max_len] + [0] * max(0, max_len - len(indices))


# === Definição da arquitetura do modelo (deve ser igual ao modelo treinado) ===
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3):
        super().__init__()
        # Camada de embedding que transforma índices em vetores densos
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Camada dropout para evitar overfitting
        self.dropout = nn.Dropout(dropout)
        # Camada linear final que gera saída com dimensão igual ao número de classes
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Passa input pela camada embedding: shape (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # Aplica média ao longo da dimensão seq_len para agregar informações da frase
        pooled = embedded.mean(dim=1)
        # Aplica dropout
        dropped = self.dropout(pooled)
        # Passa pela camada linear para saída final (logits)
        return self.fc(dropped)


# === Carregar recursos treinados: vocabulário, label encoder e modelo ===
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define se usará GPU (se disponível) ou CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cria a instância do modelo com os parâmetros usados no treinamento
model = SentimentClassifier(
    vocab_size=len(vocab),
    embed_dim=100,
    num_classes=len(label_encoder.classes_),
    dropout=0.4
)

# Carrega pesos do modelo treinado, adaptando para CPU ou GPU conforme disponível
model.load_state_dict(torch.load('modelo_treinado.pth', map_location=device))
model.to(device)
# Coloca o modelo em modo avaliação (desativa dropout e batchnorm)
model.eval()

# === Entrada do usuário ===
user_text = input("Digite um texto para análise de sentimento: ")
# Pré-processa o texto (tokeniza, limpa, stemmer, etc)
tokens = preprocess_text(user_text)
# Codifica tokens para índices conforme vocabulário
encoded = encode_tokens(tokens, vocab)
# Converte para tensor e envia para dispositivo (CPU ou GPU)
input_tensor = torch.tensor([encoded]).to(device)

# === Predição ===
with torch.no_grad():  # Desliga cálculo de gradiente para acelerar e economizar memória
    output = model(input_tensor)  # Passa input pelo modelo
    prediction = torch.argmax(output, dim=1).item()  # Pega índice da classe com maior score
    predicted_label = label_encoder.inverse_transform([prediction])[0]  # Converte índice para rótulo legível

# Exibe o resultado da predição
print(f"Sentimento previsto: {predicted_label}")
