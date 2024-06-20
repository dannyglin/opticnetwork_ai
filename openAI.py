import pandas as pd
from openai import OpenAI
import os
import ast
import numpy as np 
import pdb

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embedding.create(input = text)

question = input("What is your question?")

response = client.chat.completion.create(
    model = "gpt-3.5-turbo",
    messages = [{"role": "user", "content": "You are an expert in Optic Network and you are an assistant who is helping to answer questions. Please help us answer question as accurate and without any errors"},
                {"role": "user", "content": question}])

response

response.choices[0].message.content


