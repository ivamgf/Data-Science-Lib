# -*- coding: utf-8 -*-
# Sentiment analysis com NLTK (stopwords, tokenização, stemmer) + Embedding + LSTM

import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------- Configurações ----------------
config = {
    "learning_rate": 0.001,
    "dropout": 0.3,
    "epochs": 30,
    "batch_size": 2,
    "lstm_units": 256,
    "max_words": 5000,
    "max_len": 20,
    "test_size": 0.2,
    "random_state": 42,
    "verbose": 1
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
labels = np.array([1,0,1,0,1,0,0,1,0,1])

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
        processed.append(" ".join(tokens))
    return processed

processed = preprocess(corpus)

# ---------------- Tokenização + Padding ----------------
tokenizer = Tokenizer(num_words=config["max_words"])
tokenizer.fit_on_texts(processed)
X_seq = tokenizer.texts_to_sequences(processed)
X_pad = pad_sequences(X_seq, maxlen=config["max_len"])
y = labels

# ---------------- Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=config["test_size"],
    random_state=config["random_state"], stratify=y
)

# ---------------- Modelo LSTM ----------------
model = Sequential()
model.add(Embedding(input_dim=config["max_words"], output_dim=128, input_length=config["max_len"]))
model.add(LSTM(config["lstm_units"], return_sequences=True))
model.add(Dropout(config["dropout"]))
model.add(LSTM(64))
model.add(Dropout(config["dropout"]))
model.add(Dense(32, activation="relu"))
model.add(Dropout(config["dropout"]))
model.add(Dense(1, activation="sigmoid"))

opt = Adam(learning_rate=config["learning_rate"])
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# ---------------- Treinamento ----------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    verbose=config["verbose"]
)

# ---------------- Teste ----------------
from sklearn.metrics import precision_score, f1_score

y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== Resultados no conjunto de teste ===")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nResultados por exemplo:")
for i, (true, pred, prob) in enumerate(zip(y_test, y_pred, y_pred_prob)):
    print(f"Exemplo {i+1}: True={true}, Pred={pred}, Probabilidade={prob:.4f}")

# ---------------- Predição ----------------
def predict_sentiment(text):
    proc = preprocess([text])
    seq = tokenizer.texts_to_sequences(proc)
    pad = pad_sequences(seq, maxlen=config["max_len"])
    prob = model.predict(pad).ravel()[0]
    return ("Positivo" if prob >= 0.5 else "Negativo", prob)

novo_texto = "Achei o filme muito ruim e cansativo."
sentimento, prob = predict_sentiment(novo_texto)
print(f"\nTexto: {novo_texto}")
print(f"Classificação: {sentimento} (probabilidade={prob:.4f})")
