# Algorithm to text mining
# Preparation with stemmer

# Imports
import os
import re
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer  # stemmer para português

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Functions
def preprocess_text(text):
    """Executa tokenização, limpeza, remoção de stopwords e stemmização"""
    # Tokenizar
    tokens = nltk.word_tokenize(text, language="portuguese")

    # Remover símbolos e números
    tokens = [re.sub(r'[^a-zA-Zá-úÁ-Ú]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]  # remove strings vazias

    # Remover stopwords
    stop_words = set(stopwords.words("portuguese"))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Aplicar stem
    stemmer = RSLPStemmer()
    stemmed = [stemmer.stem(token.lower()) for token in tokens]

    # Remover "revisaosistema" (ou variações com maiúsculas/minúsculas)
    stemmed = [token for token in stemmed if token != "revisaosistema"]

    return stemmed


def main():
    filepath = '../Data/artigo.pdf'
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return

    # Open PDF
    with open(filepath, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)

        # Read all pages
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            if page.extract_text():
                text += page.extract_text() + "\n"

    # Dividir em palavras para localizar a tag
    words = text.split()
    target = "<revisao-sistematica>"

    found = False
    for i, word in enumerate(words):
        if word == target:
            start = max(0, i - 10)
            end = min(len(words), i + 11)
            context = words[start:end]
            trecho_original = " ".join(context)

            # Pré-processar apenas o trecho
            trecho_processado = preprocess_text(trecho_original)

            print("Trecho encontrado (original):")
            print(trecho_original)
            print("Trecho processado (stemmização + stopwords removidas):")
            print(trecho_processado)
            print("=" * 80)

            found = True

    if not found:
        print("Nenhum trecho com a tag <revisao-sistematica> foi encontrado.")

if __name__ == "__main__":
    main()
