from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="your_api_key")  # Ensure you set your OpenAI API key

# Load the dataframe with embeddings
df = pd.read_pickle('./website-with-embeddings.pkl')

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.json
    question = data['question']
    answer = query(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
