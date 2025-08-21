# Algorithm to text mining
# Extraction

# Imports
import os
import PyPDF2

# Functions
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

    # Dividir o texto em palavras simples
    words = text.split()

    # Tag alvo
    target = "<revisao-sistematica>"

    found = False
    for i, word in enumerate(words):
        if word == target:
            start = max(0, i - 10)
            end = min(len(words), i + 11)  # inclui a tag + 10 palavras depois
            context = words[start:end]
            print("Trecho encontrado:")
            print(" ".join(context))
            print("=" * 80)
            found = True

    if not found:
        print("Nenhum trecho com a tag <revisao-sistematica> foi encontrado.")

if __name__ == "__main__":
    main()
