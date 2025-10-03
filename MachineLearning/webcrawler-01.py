import requests

# Exemplo: buscar repositórios sobre Python
url = "https://api.github.com/search/repositories?q=python&sort=stars&order=desc"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    repos = data["items"]

    print("Top 10 repositórios sobre Python no GitHub:\n")
    for repo in repos[:10]:
        print(f"Nome: {repo['name']}")
        print(f"Dono: {repo['owner']['login']}")
        print(f"Estrelas: {repo['stargazers_count']}")
        print(f"Descrição: {repo['description']}")
        print(f"Link: {repo['html_url']}")
        print("-" * 50)
else:
    print("Erro na requisição:", response.status_code)
