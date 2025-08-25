# Text mining and labeling with NLTK WordNetLemmatizer

import os
import nltk
import PyPDF2
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar lematizador e stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))  # ajustar para 'portuguese' se seu texto for em português

# Função para processar PDF
def process_pdf(filepath):
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return None, []

    # Ler PDF
    with open(filepath, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Separar palavras
    words = text.split()

    occurrences = []
    labels = []

    # Processar cada tag
    for i, word in enumerate(words):
        if word in ["<revisao-sistematica>", "<revisao-narrativa>"]:
            start = max(0, i - 10)
            end = min(len(words), i + 11)
            context_words = words[start:i] + words[i+1:end]

            # Tokenização simples
            tokens = nltk.word_tokenize(" ".join(context_words))

            # Lematização e remoção de stopwords
            processed_tokens = [
                lemmatizer.lemmatize(token.lower())
                for token in tokens
                if token.isalpha() and token.lower() not in stop_words
            ]

            tag_name = "revisao-sistematica" if word == "<revisao-sistematica>" else "revisao-narrativa"
            occurrences.append((tag_name, " ".join(processed_tokens)))
            labels.append(1 if tag_name == "revisao-sistematica" else 0)

    # Exibir resultados
    if occurrences:
        print(f"Ocorrências no arquivo {os.path.basename(filepath)}:")
        for tag, context in occurrences:
            print(f"Tag: {tag}")
            print(f"Contexto processado: {context}")
            print("-" * 60)
    else:
        print(f"Nenhuma tag encontrada no arquivo {os.path.basename(filepath)}.")

    return text, labels

# Função principal
def main():
    filepath = '../Data/resumos.pdf'
    text, labels = process_pdf(filepath)

    print("Vetor de labels final:")
    print(labels)

if __name__ == "__main__":
    main()
