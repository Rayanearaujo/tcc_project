# Análise de tweets referentes ao Covid19

#### Esse notebook apresenta o código, escrito na linguagem de programação Python, desenvolvido como parte do trabalho de conclusão de curso apresentado pela aluna Rayane de Araújo Nunes ao Curso de Especialização em Ciência de Dados e Big Data como requisito parcial à obtenção do título de especialista. 
#### Nesse notebook, encontra-se o código referente ao tratamento e processamento dos dados para análise de tweets referentes ao COVID-19, visando analisar o que as pessoas estavam falando sobre o isolamento social aplicado como forma de contenção à pandemia.

### Processamento / tratamento de dados

# importação das bibliotecas

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer
import glob

# leitura do arquivo csv contendo o dataset extraído do Kaggle transformando-o em um dataframe do pandas

df1 = pd.read_csv('dataset/archive/covid19_tweets.csv')

# leitura do arquivo csv contendo o dataset extraído pela Twitter API transformando-o em um dataframe do pandas

path = r'C:\Users\Rayane\Documents\PUC\TCC\twitter_data'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, sep='|')
    li.append(df)

df2 = pd.concat(li, axis=0, ignore_index=True)

# impressão do número de linhas e colunas presentes no dataframe 1

print(df1.shape)

# impressão do número de linhas e colunas presentes no dataframe 1

print(len(df1.index))

# impressão do número de linhas e colunas presentes no dataframe 2

print(len(df2.index))

# impressão de informações sobre os dataframes

df1.info()

df2.info()

# visualização de uma amostra dos dados do dataframe 1

df1

# visualização de uma amostra dos dados do dataframe 2

df2

# extração das hashtags do dataframe 2

def extract_hashtags(row):
    result_text = []
    text = row['text']
    hashtags = [word for word in text.split() if word[0] == "#"]
    for hashtag in hashtags:
        result_text.append(hashtag)
    return str(result_text)

df2['hashtags'] = df2.apply(extract_hashtags, axis=1)
df2

# verificação da presença da hashtag lockdown no dataframe 1

initial_df1 = df1.copy()

def has_lockdown(row):
    has_lockdown = 0
    text = row['hashtags']
    hashtags_list = text.replace("[", "").replace("]", "").replace("'", "").replace(",", "")
    for word in hashtags_list.split():
        if word.lower() == 'lockdown' or word.lower() == 'stayhome' or word.lower() == 'quarantine' or \
            word.lower() == 'socialdistancing' or word.lower() == 'socialdistance':
            has_lockdown = 1
    return has_lockdown

df1 = df1.dropna(subset=['hashtags'])
df1['has_lockdown_hashtag'] = df1.apply(has_lockdown, axis=1)
df1

df1 = df1.loc[df1['has_lockdown_hashtag'] == 1]
len(df1.index)

# extrair 15 mil registros do df1

df1 = initial_df1.copy()

df1 = df1[:15000]

# merge dataframes

df = pd.concat([df1,df2])

print(len(df1.index) + len(df2.index))
print(len(df.index))

df

# verificação de linhas duplicadas

unique_df = np.unique(df[["user_name", "user_location", "user_description", "user_created", 
                          "user_followers", "user_friends", "user_favourites", "user_verified", 
                          "date", "text", "hashtags", "source", "is_retweet"]], axis=0)

# exibição do número de linhas

len(unique_df.index)

# verificação do número de linhas duplicadas usando drop_duplicates

unique_df = df.drop_duplicates(["user_name", "user_location", "user_description", "user_created", 
                                "user_followers", "user_friends", "user_favourites", "user_verified", 
                                "date", "text", "hashtags", "source", "is_retweet"])

# exibição do número de linhas

print('Número de linhas duplicadas: ', len(df.index) - len(unique_df.index))

print('Novo tamanho do dataframe: ', len(unique_df.index))

# verificação do número de usuários únicos

unique_users_df = unique_df.drop_duplicates(["user_name"])

# exibição do número de linhas

len(unique_users_df.index)

# verificação do número de tweets com texto único
old_unique_df_size = len(unique_df.index)

unique_df = unique_df.drop_duplicates(["text"])

# exibição do número de linhas

print('Número de linhas duplicadas: ', old_unique_df_size - len(unique_df.index))

print('Novo tamanho do dataframe: ', len(unique_df.index))

## cálculo do número de tweets com mensagens repetidas

print('tweets com mensagens repetidas: ', len(df.index) - len(unique_df.index))

# remoção de registros nulos

not_null = unique_df.dropna(subset=['text'])

# substituição de NaN nas hashtags

not_null['hashtags'] = not_null['hashtags'].fillna('')

print('tweets nulos: ', len(unique_df.index) - len(not_null.index))

# remover colunas não necessárias
cleaned_df = not_null.drop(columns=["user_name", "user_description", 
                            "user_created", "user_followers", "user_friends", 
                            "user_favourites", "user_verified", "source", "is_retweet"])

print(len(cleaned_df.index))
cleaned_df

# conversão dos datatypes

cleaned_df.info()

cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

cleaned_df.info()

# quebrar coluna de data em dia, mês, ano, data e dia da semana

cleaned_df['day'] = pd.to_datetime(cleaned_df['date']).dt.day
cleaned_df['month'] = pd.to_datetime(cleaned_df['date']).dt.month
cleaned_df['year'] = pd.to_datetime(cleaned_df['date']).dt.year
cleaned_df['date'] = pd.to_datetime(cleaned_df['date']).dt.date
cleaned_df['week_day'] = pd.to_datetime(cleaned_df['date']).dt.day_name()

print(len(cleaned_df.index))
cleaned_df.head(3)

cleaned_df.info()

# convert to lower case

cleaned_df['text'] = cleaned_df['text'].str.lower()
cleaned_df['user_location'] = cleaned_df['user_location'].str.lower()
cleaned_df['hashtags'] = cleaned_df['hashtags'].str.lower()

cleaned_df

## remoção de palavras que começam com @ and #

def remove_mention_and_hashtags(row):
    text = row['text']
    result_text = [word for word in text.split() if word[0] != "@" and word[0] != "#"]
    result_text = ' '.join(result_text)
    return result_text
    
cleaned_df["text"] = cleaned_df.apply(remove_mention_and_hashtags, axis=1)

# removeção de links

cleaned_df['text'] = cleaned_df['text'].str.replace('http\S+|www.\S+', '', case=False)


# tokenização

# nltk.download('punkt')

tokenized_df = cleaned_df.copy()

def tokenize_data(row):
    text = row['text']
    tokens = nltk.word_tokenize(text)
    
    token_words = [word for word in tokens if word.isalpha()] # ignore punctuation
    return token_words

tokenized_df['tokenized_text'] = cleaned_df.apply(tokenize_data, axis=1)

tokenized_df

# stemming

stemming = PorterStemmer()

def stem_list(row):
    text_list = row['tokenized_text']
    stemmed_list = [stemming.stem(word) for word in text_list]
    return (stemmed_list)

tokenized_df['stemmed_text'] = tokenized_df.apply(stem_list, axis=1)

tokenized_df

# remoção de stop words de stemmed_text

stops = set(stopwords.words("english"))                  

def remove_stop_words(row):
    list_without_stop_words = row['stemmed_text']
    meaningful_words = [word for word in list_without_stop_words if not word in stops]
    return (meaningful_words)

tokenized_df['cleaned_stemmed_text'] = tokenized_df.apply(remove_stop_words, axis=1)

tokenized_df

# remoção de stop words from tokenized_text

stops = set(stopwords.words("english"))                  
# stops.update(['https', 'http', 'amp', 'u'])

def remove_stop_words(row):
    list_without_stop_words = row['tokenized_text']
    meaningful_words = [word for word in list_without_stop_words if not word in stops]
    return (meaningful_words)

tokenized_df['cleaned_tokenized_text'] = tokenized_df.apply(remove_stop_words, axis=1)

tokenized_df

# criação de coluna unificando as palavras após tokenização

def rejoin_words(row):
    words_list = row['cleaned_tokenized_text'] # foi criada a partir do campo antes do stemming para facilitar visualização das palavras na nuvem de palavras
    joined_words = ( " ".join(words_list))
    return joined_words

tokenized_df['transformed_text'] = tokenized_df.apply(rejoin_words, axis=1)

tokenized_df

# foi criada a partir do campo após o stemming para ser utilizado para gerar a análise de freqência via bag of words

# criação de coluna unificando as palavras após stemming

def rejoin_words(row):
    words_list = row['cleaned_stemmed_text']
    joined_words = ( " ".join(words_list))
    return joined_words

tokenized_df['transformed_stemmed_text'] = tokenized_df.apply(rejoin_words, axis=1)

tokenized_df

### Análise e Exploração dos Dados

# palavras mais utilizadas / frequência de palavras (word cloud)

# criação de string contendo todas as palavras de todos os tweets, após o tratamento

def join_rows(df, column_name):
    text = " ".join(text for text in df[column_name])
    return text

all_words = join_rows(tokenized_df, 'transformed_stemmed_text')


# exibição de nuvem de palavras

stopwords = set(STOPWORDS)
#stopwords.update(['https', 'amp', 'u']) # algumas stopwords ainda presentes precisaram ser removidas

wordcloud = WordCloud(max_font_size=50, max_words=100, stopwords=stopwords, background_color="white").generate(all_words)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# análise de palavras em 2020 (alguns meses após início da pandemia) e em 2021 (quase 1 ano depois do início)

# criação dos dataframes
df_2020 = tokenized_df.loc[tokenized_df['year'] == 2020]
df_2021 = tokenized_df.loc[tokenized_df['year'] == 2021]

# verificação de perda de dados / erro
print(len(df_2020.index) + len(df_2021.index) != len(tokenized_df.index))

# criação da nuven de palavras de 2020
all_2020_words = join_rows(df_2020, 'transformed_stemmed_text')
stopwords = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, width=400, max_words=100, stopwords=stopwords, background_color="white") \
    .generate(all_2020_words)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# criação da nuven de palavras de 2021
all_2021_words = join_rows(df_2021, 'transformed_stemmed_text')
stopwords = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, width=400, max_words=100, stopwords=stopwords, background_color="white") \
    .generate(all_2021_words)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# distribuição de posts por localização

tokenized_df['user_location'] = tokenized_df['user_location'].str.lower()

tokenized_df['user_location'].nunique()

# muitas localizações diferentes, seria necessária um tratamento mais detalhado em cima desse campo para agrupar 
# locais similares e remover irrelevantes

tokenized_df['user_location'].head(20)

# distribuição de tweets por dia da semana

ax = tokenized_df['week_day'].value_counts().plot(kind='bar', figsize=(14,8), \
          title="Distribuição de tweets por dis da semana", color='lightblue')
ax.set_xlabel("Dia da semana")
ax.set_ylabel("Quantidade")
plt.show()

# análise de termos e frequências (bag of words)

# extração dos termos mais comuns (n-grams: bigrams e trigrams)

word_vectorizer = CountVectorizer(ngram_range=(2,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(tokenized_df['transformed_stemmed_text'])
frequencies = sum(sparse_matrix).toarray()[0]

result = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency']) \
    .sort_values(by='frequency', ascending=False)

result.head(20)

result.iloc[:20].plot.barh(width=.9, figsize=(12, 8), title="Frequência de termos", color='lightblue')
plt.ylabel('Bigrams e trigrams')
plt.xlabel('Número de ocorrências')

# relevância dos termos (tf-idf)

tfIdfTransformer = TfidfTransformer(use_idf=True)
newTfIdf = tfIdfTransformer.fit_transform(sparse_matrix)
df = pd.DataFrame(newTfIdf[0].T.todense(), index=word_vectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(50))

# hashtags frequentemente associadas as hashtags extraídas (utilizadas em conjunto)

def find_hashtags(row):
    hashtags_list = row['hashtags']
    if not (type(hashtags_list) is not str and math.isnan(hashtags_list)):
        hashtags_list = hashtags_list.replace("[", "").replace("]", "").replace("'", "")
    return hashtags_list

def join_hashtags(df, column_name):
    hashtags_list = df[column_name]
    
    text = ",".join(text.replace(' ', '') for text in df[column_name])
    
    text = text.replace(', stayhome', '').replace('stayhome', '') \
        .replace(', quarantine', '').replace('quarantine', '') \
        .replace(', covid19', '').replace('covid19', '')
        
        
    return text

all_words = join_rows(tokenized_df, 'transformed_text')

result = tokenized_df[["hashtags"]].copy()

result["hashtags"] = result.apply(find_hashtags, axis=1)

all_words = join_hashtags(result, "hashtags")

# show word cloud of associated hashtags
stopwords = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, max_words=50, stopwords=stopwords, background_color="white").generate(all_words)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# identificação de outliers de acordo com o tamanho da mensagem

final_df = tokenized_df.copy()
final_df['text_size'] = final_df.transformed_text.apply(lambda x: len(x))

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(final_df['text_size'], color="lightblue", axlabel='tamanho do texto')
plt.title('Distribuição do tamanho dos tweets')

plt.figure(figsize=(10,7), dpi= 80)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(y=final_df['text_size'])
plt.title('Estatísticas dos tweets')

final_df['text_size'].describe()

# analisar min

final_df[final_df['text_size'] == 0]

# analisar max

final_df[final_df['text_size'] > 230]

# analisar outliers

final_df[final_df['text_size'] > 140]

print(len(final_df[final_df['text_size'] < 20].index))

final_df[final_df['text_size'] < 20]

# remoção de textos com menos de 20 caracteres

print(len(final_df.index))
print(len(tokenized_df.index))

final_df = final_df.loc[final_df['text_size'] > 20]

print(len(final_df.index))

final_df

### Criação de Modelos de Machine Learning

# análise de termos e frequências (bag of words)

word_vectorizer = CountVectorizer(max_features=10000, analyzer='word')
bow_vector = word_vectorizer.fit_transform(final_df['transformed_text'])
feature_names = word_vectorizer.get_feature_names()

tfIdfTransformer = TfidfTransformer().fit(bow_vector)
newTfIdf = tfIdfTransformer.transform(bow_vector)

# impessão de informação (linhas e colunas) da matriz esparsa gerada pelo modelo tf-idf

newTfIdf

# redução de dimensionalidade da matriz esparsa

svd = TruncatedSVD(n_components=3000)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(newTfIdf)
print(svd.explained_variance_ratio_.sum())

# explained_variance_ratio deve ser maior que 60%, idealmente mais que 80% para evitar overfitting

# cálculo do k ideal utilizando o elbow method

model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(4,12), metric='calinski_harabasz', timings=False, locate_elbow=True
)

visualizer.fit(X)
visualizer.show()

### kmeans

# Criação do modelo de clusterização usando K-means

true_k = 7
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=0)
model.fit(X)
y_pred = model.predict(X)
final_df["cluster"] = model.fit_predict(X)

# mostra clusters distintos

different_clusters_df = final_df[['cluster']].copy()

different_clusters_df = different_clusters_df.drop_duplicates(["cluster"])

different_clusters_df

# termos mais frequentes por cluster

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = word_vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        try:
            print(' %s' % terms[ind]) # end="\n"
        except:
            print('index not found: ', ind)
    print

# termos mais frequentes do cluster e seus tf-idf

bow_transformer = word_vectorizer.fit(final_df['transformed_text'])

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms_list = []
terms = word_vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        cluster = i
        term = feature_names[ind]
        value = tfIdfTransformer.idf_[bow_transformer.vocabulary_[term]]
        print(term, ": ", value)
        terms_list.append([i, term, value])

df_terms = pd.DataFrame(terms_list, columns=['cluster', 'term', 'tf_idf'])
df_terms

# plot the cluster assignments and cluster centers

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
plt.scatter(model.cluster_centers_[:, 0],   
            model.cluster_centers_[:, 1],
            marker='^', 
            c=[0, 1, 2, 3, 4, 5, 6], 
            s=100, 
            linewidth=2,
            cmap="plasma")

### mini batch kmeans

# Criação do modelo de clusterização usando mini batch K-means

true_k = 7
mbk_model = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
mbk_model.fit(X)
mbk_y_pred = mbk_model.predict(X)
final_df["mbk_cluster"] = mbk_model.fit_predict(X)

# mostra clusters distintos

different_clusters_df = final_df[['mbk_cluster']].copy()

different_clusters_df = different_clusters_df.drop_duplicates(["mbk_cluster"])

different_clusters_df

# termos mais frequentes por cluster

print("Top terms per cluster:")
order_centroids = mbk_model.cluster_centers_.argsort()[:, ::-1]

terms = word_vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        try:
            print(' %s' % terms[ind]) # end="\n"
        except:
            print('index not found: ', ind)
    print

# termos mais frequentes do cluster e seus tf-idf

bow_transformer = word_vectorizer.fit(final_df['transformed_text'])

print("Top terms per cluster:")
order_centroids = mbk_model.cluster_centers_.argsort()[:, ::-1]
terms_list = []
terms = word_vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        cluster = i
        term = feature_names[ind]
        value = tfIdfTransformer.idf_[bow_transformer.vocabulary_[term]]
        print(term, ": ", value)
        terms_list.append([i, term, value])

df_mbk_terms = pd.DataFrame(terms_list, columns=['cluster', 'term', 'tf_idf'])
df_mbk_terms

# plot the cluster assignments and cluster centers

plt.scatter(X[:, 0], X[:, 1], c=mbk_y_pred, cmap="plasma")
plt.scatter(mbk_model.cluster_centers_[:, 0],   
            mbk_model.cluster_centers_[:, 1],
            marker='^', 
            c=[0, 1, 2, 3, 4, 5, 6], 
            s=100, 
            linewidth=2,
            cmap="plasma")

## Apresentação dos Resultados

### mini batch kmeans

# entre -1 e 1, quanto mais perto de 1 melhor. Valores próximos de 0 significam overlapping clusters

print("Silhouette Coefficient: %0.2f" % metrics.silhouette_score(X, mbk_model.labels_, sample_size=1000))

# a partir de 0, sendo 0 o melhor valor

print("Davies Bouldin Score: %0.2f" % metrics.davies_bouldin_score(X, mbk_model.labels_))

### kmeans

# entre -1 e 1, quanto mais perto de 1 melhor. Valores próximos de 0 significam overlapping clusters

print("Silhouette Coefficient: %0.2f" % metrics.silhouette_score(X, model.labels_, sample_size=1000))

# a partir de 0, sendo 0 o melhor valor

print("Davies Bouldin Score: %0.2f" % metrics.davies_bouldin_score(X, model.labels_))


# número de mensagens por cluster

result_count_df = final_df['cluster'].value_counts()

result_count_df

# quantidade de tweet por cluster

ax = final_df['cluster'].value_counts().plot(kind='bar', figsize=(14,8), title="Total de tweets por cluster", \
         color='lightblue')
ax.set_xlabel("cluster")
ax.set_ylabel("Número de tweets")
plt.show()

# quantidade de mensagens por tamanho de tweet por cluster

average_df = DataFrame({'average' : final_df.groupby('cluster')['text_size'].mean()}).reset_index()

sns.barplot(x="cluster", y="average", data=average_df, palette="Blues_d")
plt.title("Média do tamanho dos tweets por cluster")

# quantidade de mensagens por dia da semana por cluster

plt.subplots(figsize=(14,8))
sns.countplot(final_df['cluster'], hue=final_df['week_day'], palette="Blues_d")
plt.title("Quantidade de mensagens por dia da semana por cluster")
plt.show()

# quantidade de mensagens por ano por cluster

plt.subplots(figsize=(14,8))
sns.countplot(final_df['cluster'], hue=final_df['year'], palette="Blues_d")
plt.title("Quantidade de mensagens por ano por cluster")
plt.show()

# termos mais importantes de cada cluster, tf-idf

def plot_frequent_terms_and_its_tfidf(df, cluster):
    tfidf_cluster_df = df.loc[df['cluster'] == cluster]
    tfidf_cluster_df = tfidf_cluster_df[["term", "tf_idf"]].copy()

    sns.barplot(x="tf_idf", y="term", data=tfidf_cluster_df, palette="Blues_d")
    plt.title("tf-idf dos termos mais frequentes no cluster {}".format(cluster))
    plt.show()
    
for i in range(0, true_k):
    plot_frequent_terms_and_its_tfidf(df_terms, i)

# nuvem de palavras por cluster

def word_cloud_by_cluster(cluster):
    df_cluster = final_df.loc[final_df['cluster'] == cluster]

    all_cluster_words = join_rows(df_cluster, 'transformed_text')
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_font_size=50, width=400, max_words=50, stopwords=stopwords, background_color="white") \
        .generate(all_cluster_words)
    print('Word cloud for cluster: ', cluster)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

for i in range(0, true_k):
    word_cloud_by_cluster(i)
