import requests

def fetch_news(keywords, secret="057734477b67461a91969fee3df2635f"):
    
    url = 'https://newsapi.org/v2/everything?'

    topics = '|'.join(keywords)

    language = 'en'

    parameters = {
        'q': topics,  
        'pageSize': 10,  
        'apiKey': secret,  
        'language': language  
    }

    response = requests.get(url, params=parameters)

    response_json = response.json()

    articles = response_json['articles']
    
    news = [{'content': article['content']} for article in articles]
    print(news)
    return news