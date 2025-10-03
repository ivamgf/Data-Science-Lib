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

# ===============================
# Função de pré-processamento
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

# ===============================
# Codificação de tokens
# ===============================
def encode_tokens(tokens, vocab, max_len=50):
    indices = [vocab.get(t, 0) for t in tokens]  # 0 para tokens desconhecidos
    return indices[:max_len] + [0] * max(0, max_len - len(indices))

# ===============================
# Modelo LSTM
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
# Verificações de arquivos
# ===============================
if not os.path.exists("melhor_modelo.pth"):
    print("Arquivo 'melhor_modelo.pth' não encontrado.")
    exit()
if not os.path.exists("vocab.pkl") or not os.path.exists("label_encoder.pkl"):
    print("Arquivos 'vocab.pkl' ou 'label_encoder.pkl' não encontrados.")
    exit()

# ===============================
# Carregamento de vocabulário e label encoder
# ===============================
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ===============================
# Parâmetros devem bater com o modelo treinado
# ===============================
embed_dim = 100   # deve ser igual ao do treino
hidden_dim = 128  # deve ser igual ao do treino
num_classes = len(label_encoder.classes_)
vocab_size = len(vocab)

# ===============================
# Inicializa modelo e carrega pesos
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, num_classes)

# Carregamento parcial para evitar erro de mismatch
state_dict = torch.load("melhor_modelo.pth", map_location=device)
model_dict = model.state_dict()
# Atualiza apenas parâmetros compatíveis
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.to(device)
model.eval()

# ===============================
# Predição de sentimento
# ===============================
user_text = input("Digite um texto para análise de sentimento: ")

tokens = preprocess_text(user_text)
encoded = encode_tokens(tokens, vocab)
input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    predicted_label = label_encoder.inverse_transform([prediction])[0]

print(f"Sentimento previsto: {predicted_label}")
