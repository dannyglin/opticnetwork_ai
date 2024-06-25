import pandas as pd
from openai import OpenAI
import os
import ast
import numpy as np 
import pdb

question = input("What is your question?")

response = client.chat.completion.create(
    model = "gpt-3.5-turbo",
    messages = [{"role": "user", "content": "You are an expert in Optic Network and you are an assistant who is helping to answer questions. Please help us answer question as accurate and without any errors"},
                {"role": "user", "content": question}])

response.choices[0].message.content

question = "Do you have parking"
question

# Insert Path if the pdfs
df = pd.read_csv('insert path')
df.head


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

get_embedding(df['text'].iloc[0])

%%time
df['text'].head(5).apply(get_embedding)
%%time


get_embedding(df['text'].applying(get_embedding))
df.to_csv('./website-with-embeddings.csv', index=False)
df.to_pickle('./website-with-embedding.pkl')

%%time

question_embedding = get_embedding(question)
question, question_embedding[0:10], "..."

def fn(page_embedding):
    return np.dot(page_embedding, question_embedding)

df['distance'] = df['embedding'].apply(fn)
df.head()

df.sort_values('distance', ascending=False, inplace=True)


response

client = OpenAI()



