import pandas as pd
from openai import OpenAI
import numpy as np
import os

# Initialize OpenAI client
client = OpenAI(api_key="your_api_key")  # Ensure you set your OpenAI API key

# Function to get embedding
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Load the dataset
df = pd.read_csv('path_to_your_csv')  # Replace 'path_to_your_csv' with your actual file path

# Generate embeddings for the text in the dataframe
df['embedding'] = df['text'].apply(get_embedding)

# Save the dataframe with embeddings
df.to_csv('./website-with-embeddings.csv', index=False)
df.to_pickle('./website-with-embeddings.pkl')

# Function to handle queries
def query(question):
    # Get the embedding for the question
    question_embedding = get_embedding(question)

    # Calculate cosine similarity between question and page embeddings
    def cosine_similarity(page_embedding):
        return np.dot(page_embedding, question_embedding) / (np.linalg.norm(page_embedding) * np.linalg.norm(question_embedding))
    
    # Apply similarity function to embeddings and get top 4 matches
    df['distance'] = df['embedding'].apply(cosine_similarity)
    top_four_indices = df.nlargest(4, 'distance').index

    # Combine the top 4 context texts
    context = "\n\n".join(df.loc[top_four_indices, 'text'])

    # Generate response using OpenAI's chat completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant who is helping the AT&T Optic Network Team respond to their questions. Use first person (e.g., 'We') to refer to the AT&T Optic Networks Team."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Use this information from the AT&T Optic Network Team website as context to answer the user question: {context}. Please stick to this context when answering these questions."}
        ]
    )
    
    return response.choices[0].message.content

# Example query
print(query("What is optic networks?"))
