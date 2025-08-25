# Text mining, stemming, Bag of Words and Word2Vec

import os
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from gensim.models import Word2Vec
from collections import Counter

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Inicializar stemmer e stopwords
stemmer = RSLPStemmer()
stop_words = set(stopwords.words("portuguese"))

# Função para processar PDF
def process_pdf(filepath):
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return None, [], []

    # Ler PDF
    with open(filepath, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    words = text.split()
    occurrences = []
    labels = []
    processed_contexts = []  # armazena listas de tokens processados

    for i, word in enumerate(words):
        if word in ["<revisao-sistematica>", "<revisao-narrativa>"]:
            start = max(0, i - 10)
            end = min(len(words), i + 11)
            context_words = words[start:i] + words[i+1:end]

            # Tokenização
            tokens = nltk.word_tokenize(" ".join(context_words), language="portuguese")

            # Stemização e remoção de stopwords
            processed_tokens = [
                stemmer.stem(token.lower())
                for token in tokens
                if token.isalpha() and token.lower() not in stop_words
            ]

            if processed_tokens:
                processed_contexts.append(processed_tokens)

            tag_name = "revisao-sistematica" if word == "<revisao-sistematica>" else "revisao-narrativa"
            occurrences.append((tag_name, " ".join(processed_tokens)))
            labels.append(1 if tag_name == "revisao-sistematica" else 0)

    # Construir Bag of Words
    all_tokens = [token for context in processed_contexts for token in context]
    bow = Counter(all_tokens)

    # Treinar Word2Vec
    # CBOW => sg=0
    # SkipGram => sg=1
    if processed_contexts:
        w2v_model = Word2Vec(sentences=processed_contexts, vector_size=100, window=5, min_count=1, workers=4, sg=0)
    else:
        w2v_model = None

    return text, labels, bow, w2v_model, occurrences

# Função principal
def main():
    filepath = '../Data/resumos.pdf'
    text, labels, bow, w2v_model, occurrences = process_pdf(filepath)

    print("Vetor de labels final:", labels)
    print("Bag of Words:", bow.most_common(10))  # exibir 10 palavras mais frequentes
    if w2v_model:
        print("Exemplo de embedding Word2Vec para a primeira palavra do vocabulário:")
        first_word = list(w2v_model.wv.index_to_key)[0]
        print(first_word, w2v_model.wv[first_word])
    else:
        print("Modelo Word2Vec não foi treinado.")

    # Exibir contextos processados
    for tag, context in occurrences:
        print(f"Tag: {tag} | Contexto: {context}")

if __name__ == "__main__":
    main()
