# Algorithm to text mining and labeling
# Extraction and labeling with tokenization, stopwords removal and stemming

import os
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Função para processar um único arquivo PDF
def process_pdf(filepath):
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        return None, []

    # Abrir PDF
    with open(filepath, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)

        # Ler todas as páginas
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Dividir o texto em palavras (pré-tokenização simples)
    words = text.split()

    # Inicializar lista de ocorrências e vetor de rótulos
    occurrences = []
    labels = []

    # Preparar recursos de PLN
    stop_words = set(stopwords.words("portuguese"))
    stemmer = RSLPStemmer()

    # Procurar todas as ocorrências das tags
    for i, word in enumerate(words):
        if word in ["<revisao-sistematica>", "<revisao-narrativa>"]:
            context_start = max(0, i - 10)
            context_end = min(len(words), i + 11)
            context_words = words[context_start:i] + words[i+1:context_end]  # remove a tag

            # Tokenização
            tokens = nltk.word_tokenize(" ".join(context_words), language="portuguese")

            # Remoção de stopwords e stemming
            processed_tokens = [
                stemmer.stem(token.lower())
                for token in tokens
                if token.isalpha() and token.lower() not in stop_words
            ]

            # Guardar ocorrência processada
            tag = "revisao-sistematica" if word == "<revisao-sistematica>" else "revisao-narrativa"
            occurrences.append((tag, " ".join(processed_tokens)))
            labels.append(1 if tag == "revisao-sistematica" else 0)

    # Imprimir resultados
    if occurrences:
        print(f"Ocorrências encontradas no arquivo {os.path.basename(filepath)}:")
        for tag, context in occurrences:
            print(f"Tag: {tag}")
            print(f"Contexto processado: {context}")
            print("-" * 80)
    else:
        print(f"Nenhuma tag encontrada no arquivo {os.path.basename(filepath)}.")

    return text, labels

# Função principal
def main():
    filepath = '../Data/resumos.pdf'  # caminho do PDF
    text, labels = process_pdf(filepath)

    # Exibir vetor final de labels
    print("Vetor de labels final:")
    print(labels)

if __name__ == "__main__":
    main()
