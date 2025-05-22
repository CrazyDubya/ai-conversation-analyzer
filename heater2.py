# heater2.py
import json
import multiprocessing
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    data = json.load(file)


def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def analyze_similarity(user_token_range, assistant_token_range):
    user_tokens = []
    assistant_tokens = []
    print(
        f"Preprocessing conversations for user_token_range={user_token_range}, assistant_token_range={assistant_token_range}...")
    start_time = time.time()
    for conversation in tqdm(data, desc="Conversations", unit="conv"):
        for message in conversation['mapping'].values():
            if message.get('message'):
                content_parts = message['message'].get('content', {}).get('parts', [])
                text = ' '.join([part for part in content_parts if isinstance(part, str)])
                tokens = preprocess(text)
                if message['message']['author']['role'] == 'user':
                    user_token_start, user_token_end = user_token_range
                    if user_token_end is None:
                        user_token_end = len(tokens)
                    elif user_token_end < 0:
                        user_token_end = len(tokens) + user_token_end
                    user_tokens.append(tokens[user_token_start:user_token_end])
                elif message['message']['author']['role'] == 'assistant':
                    assistant_token_start, assistant_token_end = assistant_token_range
                    if assistant_token_end is None:
                        assistant_token_end = len(tokens)
                    elif assistant_token_end < 0:
                        assistant_token_end = len(tokens) + assistant_token_end
                    assistant_tokens.append(tokens[assistant_token_start:assistant_token_end])
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
    # Flatten the token lists
    user_texts = [' '.join(tokens) for tokens in user_tokens]
    assistant_texts = [' '.join(tokens) for tokens in assistant_tokens]

    # Vectorize the text data
    print("Vectorizing text data...")
    start_time = time.time()
    vectorizer = TfidfVectorizer()
    user_tfidf_matrix = vectorizer.fit_transform(user_texts)
    assistant_tfidf_matrix = vectorizer.transform(assistant_texts)
    print(f"Vectorization completed in {time.time() - start_time:.2f} seconds.")

    # Calculate cosine similarity between user and assistant tokens
    print("Calculating cosine similarity...")
    start_time = time.time()
    similarity_matrix = cosine_similarity(user_tfidf_matrix, assistant_tfidf_matrix)
    print(f"Cosine similarity calculation completed in {time.time() - start_time:.2f} seconds.")

    # Create a heatmap
    print("Creating heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_matrix, cmap='coolwarm', cbar=True)
    plt.title(
        f'Heatmap of Similarity Between User Message Tokens and Assistant Response Tokens\nUser Tokens: {user_token_range}, Assistant Tokens: {assistant_token_range}')
    plt.xlabel('Assistant Response Tokens')
    plt.ylabel('User Message Tokens')

    # Get the base directory and filename
    base_dir = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/'
    base_filename = f'similarity_heatmap_user_{user_token_range[0]}_{user_token_range[1]}_assistant_{assistant_token_range[0]}_{assistant_token_range[1]}'

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the sequential filename
    sequential_filename = f"{base_filename}_{timestamp}.png"

    # Combine the base directory and sequential filename to get the output path
    output_path = os.path.join(base_dir, sequential_filename)

    # Save the file using the output path
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Define the token ranges for user and assistant messages
    user_token_ranges = [(0, 16), (0, 32), (0, 64), (0, 128), (0, 256), (0, 512), (0, 1024)]
    assistant_token_ranges = [(0, 16), (0, 32), (0, 64), (0, 128), (0, 256), (0, 512), (0, 1024)]

    # Additional variations
    user_token_ranges_middle = [(0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]
    user_token_ranges_end = [(-0.1, None), (-0.25, None), (-0.5, None)]

    assistant_token_ranges_middle = [(0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]
    assistant_token_ranges_end = [(-0.1, None), (-0.25, None), (-0.5, None)]

    # Create a list to store the tasks
    tasks = []

    # Iterate over the token ranges and create tasks
    for user_token_range in user_token_ranges:
        for assistant_token_range in assistant_token_ranges:
            tasks.append((user_token_range, assistant_token_range))

    # Middle user and end assistant
    for user_token_range in user_token_ranges_middle:
        for assistant_token_range in assistant_token_ranges_end:
            tasks.append((user_token_range, assistant_token_range))

    # End user and end assistant
    for user_token_range in user_token_ranges_end:
        for assistant_token_range in assistant_token_ranges_end:
            tasks.append((user_token_range, assistant_token_range))

    # Beginning user and middle assistant
    for user_token_range in user_token_ranges:
        for assistant_token_range in assistant_token_ranges_middle:
            tasks.append((user_token_range, assistant_token_range))

    # Beginning user and end assistant
    for user_token_range in user_token_ranges:
        for assistant_token_range in assistant_token_ranges_end:
            tasks.append((user_token_range, assistant_token_range))

    # Create a process pool
    with multiprocessing.Pool() as pool:
        # Map the tasks to the process pool
        pool.starmap(analyze_similarity, tasks)
