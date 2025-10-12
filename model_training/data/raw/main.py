import requests
import pandas as pd
import time
import re

BEARER_TOKEN = ''
URL = "https://api.x.com/2/tweets/search/recent"
QUERY = '"a" lang:pt'
MAX_RESULTS = 100    
TOTAL_POSTS = 10000

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def limpar_texto(texto):
    # Remove RT (retweets)
    texto = re.sub(r'\bRT\b', '', texto)
    # Remove menções de usuários (@usuario)
    texto = re.sub(r'@\w+', '', texto)
    return texto

def buscar_posts(qtd_total):
    params = {"query": QUERY, "max_results": MAX_RESULTS}
    posts = []
    next_token = None

    try:
        while len(posts) < qtd_total:
            if next_token:
                params["next_token"] = next_token
            data = connect_to_endpoint(URL, params)
            batch = data.get("data", [])
            posts.extend(batch)
            next_token = data.get("meta", {}).get("next_token", None)
            if not next_token or not batch:
                break
            print(f"Coletados {len(posts)} posts até agora...")
            time.sleep(15)  
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    finally:
        mensagens = [limpar_texto(p['text']) for p in posts]
        df = pd.DataFrame(mensagens, columns=['Mensagem'])
        df.to_excel('mensagens_X_coletadas.xlsx', index=False)
        print(f"Salvo {len(posts)} mensagens coletadas em mensagens_X_coletadas.xlsx")

    return posts

posts = buscar_posts(TOTAL_POSTS)
