import json

# Carregar o arquivo JSON
with open('data/fr.sputniknews.africa--20220630--20230630.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extração das palavras-chave (kws) do JSON
keywords = data['metadata']['all']['kws']

# Exibir as primeiras 10 palavras-chave para verificar
for key, value in list(keywords.items())[:10]:
    print(f"Keyword: {key}, Count: {value}")
