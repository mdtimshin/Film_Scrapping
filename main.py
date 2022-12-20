import numpy as np
import pandas as pd
import requests
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import re
import matplotlib.pyplot as plt

tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

headers = {
    'X-API-KEY': "ae27cd7b-5c7b-43b1-befe-b20a1e4bb0ad",
    'accept': "application/json"
}
fig, axs = plt.subplots(nrows=1, ncols=5)
#region titanic
result1 = requests.get(
    'https://kinopoiskapiunofficial.tech/api/v2.2/films/2213',
    headers=headers
).json()
titanic = {'name': result1['nameRu'], 'rating': result1['ratingKinopoisk']}
print(titanic)

result2 = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/2213/reviews?page=1&order=DATE_DESC',
        headers=headers
).json()

for i in range(2, 11):
    result = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/2213/reviews?page={i}&order=DATE_DESC',
        headers=headers
    ).json()
    items = result['items']
    result2['items'] += items

titanic = {**titanic,
           'totalPositiveReviews': result2['totalPositiveReviews'],
           'totalNegativeReviews': result2['totalNegativeReviews'],
           'totalNeutralReviews': result2['totalNeutralReviews'],
           'reviews': list(map(lambda x: {'positiveRating': x['positiveRating'],
                                          'negativeRating': x['negativeRating'],
                                          'description': re.sub(r'\<[^>]*\>', '', x['description'])}, result2['items']))}
print(titanic)

reviews = list(map(lambda x: x['description'], titanic['reviews']))
results = model.predict(reviews, k=1)
total_score = {'reviews': reviews, 'sentiment': results}
print(total_score)

titanic_reviews = titanic['reviews']
df = pd.json_normalize(titanic_reviews)
df['sentiment'] = list(map(lambda x: list(x.keys())[0], results))
df['sentiment-value'] = list(map(lambda x: x[list(x.keys())[0]], results))
print(df)

X = np.array(df['positiveRating'])
Y = np.array(df['negativeRating'])
Z = np.array(df['sentiment-value'])
data_XYZ_df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Z': Z
})

categories = np.unique(df['sentiment'])
colors = ['red', 'blue', 'green']

plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
axs[0] = plt.axes(projection='3d')


for i, category in enumerate(categories):
    axs[0].scatter3D('positiveRating', 'negativeRating', 'sentiment-value',
                data=df.loc[df.sentiment == category, :],
                s=20, c=colors[i], label=str(category))

plt.gca().set(xlabel='Positive Rating', ylabel='negativeRating', zlabel='Sentiment Value', label='Титаник')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16)

# plt.show()
#endregion
#region intouchables
result1 = requests.get(
    'https://kinopoiskapiunofficial.tech/api/v2.2/films/535341',
    headers=headers
).json()
intouchables = {'name': result1['nameRu'], 'rating': result1['ratingKinopoisk']}
print(intouchables)

result2 = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/2213/reviews?page=1&order=DATE_DESC',
        headers=headers
).json()

for i in range(2, 11):
    result = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/535341/reviews?page={i}&order=DATE_DESC',
        headers=headers
    ).json()
    items = result['items']
    result2['items'] += items

intouchables = {**intouchables,
           'totalPositiveReviews': result2['totalPositiveReviews'],
           'totalNegativeReviews': result2['totalNegativeReviews'],
           'totalNeutralReviews': result2['totalNeutralReviews'],
           'reviews': list(map(lambda x: {'positiveRating': x['positiveRating'],
                                          'negativeRating': x['negativeRating'],
                                          'description': re.sub(r'\<[^>]*\>', '', x['description'])}, result2['items']))}
print(intouchables)

reviews = list(map(lambda x: x['description'], intouchables['reviews']))
results = model.predict(reviews, k=1)
total_score = {'reviews': reviews, 'sentiment': results}
print(total_score)

intouchables_reviews = intouchables['reviews']
df = pd.json_normalize(intouchables_reviews)
df['sentiment'] = list(map(lambda x: list(x.keys())[0], results))
df['sentiment-value'] = list(map(lambda x: x[list(x.keys())[0]], results))
print(df)

X = np.array(df['positiveRating'])
Y = np.array(df['negativeRating'])
Z = np.array(df['sentiment-value'])
data_XYZ_df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Z': Z
})

categories = np.unique(df['sentiment'])
colors = ['red', 'blue', 'green']

plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
axs[1] = plt.axes(projection='3d')


for i, category in enumerate(categories):
    axs[1].scatter3D('positiveRating', 'negativeRating', 'sentiment-value',
                data=df.loc[df.sentiment == category, :],
                s=20, c=colors[i], label=str(category))

plt.gca().set(xlabel='Positive Rating', ylabel='negativeRating', zlabel='Sentiment Value', label='1 + 1')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16)

# plt.show()
#endregion
#region inTime
result1 = requests.get(
    'https://kinopoiskapiunofficial.tech/api/v2.2/films/517988',
    headers=headers
).json()
inTime = {'name': result1['nameRu'], 'rating': result1['ratingKinopoisk']}
print(inTime)

result2 = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/517988/reviews?page=1&order=DATE_DESC',
        headers=headers
).json()

for i in range(2, 11):
    result = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/517988/reviews?page={i}&order=DATE_DESC',
        headers=headers
    ).json()
    items = result['items']
    result2['items'] += items

inTime = {**inTime,
           'totalPositiveReviews': result2['totalPositiveReviews'],
           'totalNegativeReviews': result2['totalNegativeReviews'],
           'totalNeutralReviews': result2['totalNeutralReviews'],
           'reviews': list(map(lambda x: {'positiveRating': x['positiveRating'],
                                          'negativeRating': x['negativeRating'],
                                          'description': re.sub(r'\<[^>]*\>', '', x['description'])}, result2['items']))}
print(inTime)

reviews = list(map(lambda x: x['description'], inTime['reviews']))
results = model.predict(reviews, k=1)
total_score = {'reviews': reviews, 'sentiment': results}
print(total_score)

inTime_reviews = inTime['reviews']
df = pd.json_normalize(inTime_reviews)
df['sentiment'] = list(map(lambda x: list(x.keys())[0], results))
df['sentiment-value'] = list(map(lambda x: x[list(x.keys())[0]], results))
print(df)

X = np.array(df['positiveRating'])
Y = np.array(df['negativeRating'])
Z = np.array(df['sentiment-value'])
data_XYZ_df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Z': Z
})

categories = np.unique(df['sentiment'])
colors = ['red', 'blue', 'green']

plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
axs[2] = plt.axes(projection='3d')


for i, category in enumerate(categories):
    axs[2].scatter3D('positiveRating', 'negativeRating', 'sentiment-value',
                data=df.loc[df.sentiment == category, :],
                s=20, c=colors[i], label=str(category))

plt.gca().set(xlabel='Positive Rating', ylabel='negativeRating', zlabel='Sentiment Value', label='Время')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16)

# plt.show()
#endregion
#region interstellar
result1 = requests.get(
    'https://kinopoiskapiunofficial.tech/api/v2.2/films/258687',
    headers=headers
).json()
interstellar = {'name': result1['nameRu'], 'rating': result1['ratingKinopoisk']}
print(interstellar)

result2 = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/258687/reviews?page=1&order=DATE_DESC',
        headers=headers
).json()

for i in range(2, 11):
    result = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/258687/reviews?page={i}&order=DATE_DESC',
        headers=headers
    ).json()
    items = result['items']
    result2['items'] += items

interstellar = {**interstellar,
           'totalPositiveReviews': result2['totalPositiveReviews'],
           'totalNegativeReviews': result2['totalNegativeReviews'],
           'totalNeutralReviews': result2['totalNeutralReviews'],
           'reviews': list(map(lambda x: {'positiveRating': x['positiveRating'],
                                          'negativeRating': x['negativeRating'],
                                          'description': re.sub(r'\<[^>]*\>', '', x['description'])}, result2['items']))}
print(interstellar)

reviews = list(map(lambda x: x['description'], interstellar['reviews']))
results = model.predict(reviews, k=1)
total_score = {'reviews': reviews, 'sentiment': results}
print(total_score)

interstellar_reviews = interstellar['reviews']
df = pd.json_normalize(interstellar_reviews)
df['sentiment'] = list(map(lambda x: list(x.keys())[0], results))
df['sentiment-value'] = list(map(lambda x: x[list(x.keys())[0]], results))
print(df)

X = np.array(df['positiveRating'])
Y = np.array(df['negativeRating'])
Z = np.array(df['sentiment-value'])
data_XYZ_df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Z': Z
})

categories = np.unique(df['sentiment'])
colors = ['red', 'blue', 'green']

plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
axs[3] = plt.axes(projection='3d')


for i, category in enumerate(categories):
    axs[3].scatter3D('positiveRating', 'negativeRating', 'sentiment-value',
                data=df.loc[df.sentiment == category, :],
                s=20, c=colors[i], label=str(category))

plt.gca().set(xlabel='Positive Rating', ylabel='negativeRating', zlabel='Sentiment Value', label='Интерстеллар')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16)

# plt.show()
#endregion
#region shrek
result1 = requests.get(
    'https://kinopoiskapiunofficial.tech/api/v2.2/films/430',
    headers=headers
).json()
shrek = {'name': result1['nameRu'], 'rating': result1['ratingKinopoisk']}
print(shrek)

result2 = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/430/reviews?page=1&order=DATE_DESC',
        headers=headers
).json()

for i in range(2, 11):
    result = requests.get(
        f'https://kinopoiskapiunofficial.tech/api/v2.2/films/430/reviews?page={i}&order=DATE_DESC',
        headers=headers
    ).json()
    items = result['items']
    result2['items'] += items

shrek = {**shrek,
           'totalPositiveReviews': result2['totalPositiveReviews'],
           'totalNegativeReviews': result2['totalNegativeReviews'],
           'totalNeutralReviews': result2['totalNeutralReviews'],
           'reviews': list(map(lambda x: {'positiveRating': x['positiveRating'],
                                          'negativeRating': x['negativeRating'],
                                          'description': re.sub(r'\<[^>]*\>', '', x['description'])}, result2['items']))}
print(shrek)

reviews = list(map(lambda x: x['description'], shrek['reviews']))
results = model.predict(reviews, k=1)
total_score = {'reviews': reviews, 'sentiment': results}
print(total_score)

shrek_reviews = shrek['reviews']
df = pd.json_normalize(shrek_reviews)
df['sentiment'] = list(map(lambda x: list(x.keys())[0], results))
df['sentiment-value'] = list(map(lambda x: x[list(x.keys())[0]], results))
print(df)

X = np.array(df['positiveRating'])
Y = np.array(df['negativeRating'])
Z = np.array(df['sentiment-value'])
data_XYZ_df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Z': Z
})

categories = np.unique(df['sentiment'])
colors = ['red', 'blue', 'green']

plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
axs[3] = plt.axes(projection='3d')


for i, category in enumerate(categories):
    axs[3].scatter3D('positiveRating', 'negativeRating', 'sentiment-value',
                data=df.loc[df.sentiment == category, :],
                s=20, c=colors[i], label=str(category))

plt.gca().set(xlabel='Positive Rating', ylabel='negativeRating', zlabel='Sentiment Value', label='Шрэк')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16)

plt.show()
#endregion

