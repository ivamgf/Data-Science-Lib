# Algorithm to text mining and labeling
# Extraction and labeling

import os
import PyPDF2

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

    # Dividir o texto em palavras
    words = text.split()

    # Inicializar lista de ocorrências e vetor de rótulos
    occurrences = []
    labels = []

    # Procurar todas as ocorrências das tags
    for i, word in enumerate(words):
        if word == "<revisao-sistematica>":
            context_start = max(0, i - 10)
            context_end = min(len(words), i + 11)
            context_words = words[context_start:i] + words[i+1:context_end]  # remove a tag
            occurrences.append(("revisao-sistematica", " ".join(context_words)))
            labels.append(1)
        elif word == "<revisao-narrativa>":
            context_start = max(0, i - 10)
            context_end = min(len(words), i + 11)
            context_words = words[context_start:i] + words[i+1:context_end]  # remove a tag
            occurrences.append(("revisao-narrativa", " ".join(context_words)))
            labels.append(0)

    # Imprimir resultados
    if occurrences:
        print(f"Ocorrências encontradas no arquivo {os.path.basename(filepath)}:")
        for tag, context in occurrences:
            print(f"Tag: {tag}")
            print(f"Contexto: {context}")
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
