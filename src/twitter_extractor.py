# importação das bibliotecas

import tweepy as tw
import pandas as pd

# configuração das credenciais de acesso a API do twitter

api_key = "API_KEY_GOES_HERE"
secret_key = "SECRET_KEY_GOES_HERE"
access_token = "ACCESS_TOKEN_GOES_HERE"
access_token_secret = "ACCESS_TOKEN_SECRET_GOES_HERE"

auth = tw.OAuthHandler(api_key, secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth, wait_on_rate_limit=True)

# definição do período de extração e da chave que será buscada
# stayhome and quarantine
# extracted in 10/01/2021

search_words = "#stayhome -filter:retweets"
date_since = "2021-01-05"
date_until = "2021-01-06"

# busca os dados na API do Twitter

tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since,
              tweet_mode="extended",
              until=date_until).items(10000)
tweets

# Criação de uma lista com os dados extraídos

tweets_data = [[tweet.full_text, tweet.created_at] for tweet in tweets]

tweets_data

# criação de um dataframe do Pandas a partir da lista

tweets_df = pd.DataFrame(data=tweets_data, columns=['text', "date"])

tweets_df.head(3)

tweets_df.to_csv('stayhome_twitter_data_05_01_2020.csv', index=False, sep='|')
