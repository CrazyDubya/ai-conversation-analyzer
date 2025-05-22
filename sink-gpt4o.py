import json
import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import IsolationForest
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from textblob import TextBlob
from wordcloud import WordCloud

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Define functions
def print_time_notice(task):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{task} started at {current_time}")


def load_conversations(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(10)]


def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])


def analyze_conversations(conversations):
    analyzed_data = []
    for conversation in conversations:
        conversation_id = conversation['id']
        title = conversation.get('title', 'Untitled')
        messages = []
        full_text = ''
        for node in conversation['mapping'].values():
            if node.get('message') and node['message']['content']:
                message_content = node['message']['content'].get('parts', [])
                author_role = node['message']['author']['role']
                text_parts = [part if isinstance(part, str) else part.get('text', '') for part in message_content]
                text = ' '.join(text_parts)
                full_text += ' ' + text
                messages.append({'author': author_role, 'content': text})
        keywords = extract_keywords(full_text)
        summary = generate_summary(full_text)
        analyzed_data.append({
            'conversation_id': conversation_id,
            'title': title,
            'messages': messages,
            'keywords': keywords,
            'summary': summary
        })
    return analyzed_data


def save_analyzed_conversations(analyzed_conversations, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for conversation in analyzed_conversations:
        output_file = os.path.join(output_dir, f"{conversation['title'][:10]}_{conversation['conversation_id']}.json")
        with open(output_file, 'w') as file:
            json.dump(conversation, file, indent=4)
    print(f"Analyzed conversations have been saved to {output_dir}")


def visualize_message_lengths(user_messages):
    message_lengths = [len(message) for message in user_messages]
    max_length = max(message_lengths)
    min_length = min(message_lengths)
    mean_length = np.mean(message_lengths)
    median_length = np.median(message_lengths)
    std_dev_length = np.std(message_lengths)

    print("Message Length Statistics:")
    print(f"Longest message: {max_length} characters")
    print(f"Shortest message: {min_length} characters")
    print(f"Mean message length: {mean_length:.2f} characters")
    print(f"Median message length: {median_length:.2f} characters")
    print(f"Standard deviation of message length: {std_dev_length:.2f} characters")

    plt.figure(figsize=(10, 6))
    plt.hist(message_lengths, bins=20, edgecolor='black')
    plt.title('Distribution of User Message Lengths')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Frequency')
    plt.savefig('user_message_length_distribution.png')
    plt.show()

    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(message_lengths, percentiles)
    print("\nMessage Length Percentiles:")
    for p, v in zip(percentiles, percentile_values):
        print(f"{p}th percentile: {v:.2f} characters")

    sorted_lengths = np.sort(message_lengths)
    cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lengths, cumulative_prob)
    plt.title('Cumulative Distribution of User Message Lengths')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Cumulative Probability')
    plt.savefig('user_message_length_cumulative_distribution.png')
    plt.show()


def detect_anomalies(conversations):
    num_messages = [len(conversation['mapping']) for conversation in conversations]
    model = IsolationForest(contamination=0.1)
    num_messages = np.array(num_messages).reshape(-1, 1)
    model.fit(num_messages)
    anomalies = model.predict(num_messages)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(num_messages)), num_messages, c=anomalies, cmap='viridis')
    plt.title('Anomaly Detection in Conversations')
    plt.xlabel('Conversation Index')
    plt.ylabel('Number of Messages')
    plt.savefig('anomaly_detection.png')
    plt.show()


def analyze_sentiment_trends(conversations):
    plt.figure(figsize=(14, 7))
    for i, conversation in enumerate(conversations[:10]):
        timestamps = []
        sentiments = []
        for node in conversation['mapping'].values():
            message = node.get('message')
            if message and 'content' in message:
                content_parts = message['content'].get('parts', [])
                text = ' '.join([part for part in content_parts if isinstance(part, str)])
                sentiment = TextBlob(text).sentiment.polarity
                timestamps.append(message['create_time'])
                sentiments.append(sentiment)
        if timestamps:
            plt.plot(timestamps, sentiments, label=f'Conversation {i + 1}')
    plt.title('Sentiment Trends in Conversations')
    plt.xlabel('Timestamp')
    plt.ylabel('Sentiment Polarity')
    plt.legend()
    plt.savefig('sentiment_trend.png')
    plt.show()


def visualize_word_frequencies(user_messages):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(' '.join(user_messages))
    filtered_words = [w for w in word_tokens if not w.lower() in stop_words and w.isalnum()]
    word_freq = Counter(filtered_words)

    word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency',
                                                                                              ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=word_freq_df.head(20))
    plt.title('Top 20 Most Common Words in User Messages')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig('top_20_user_words.png')
    plt.show()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of User Messages')
    plt.savefig('user_wordcloud.png')
    plt.show()


def main():
    file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
    print_time_notice("Loading conversation data")
    conversations = load_conversations(file_path)

    print_time_notice("Extracting user messages")
    user_messages = [part for conversation in conversations for node in conversation['mapping'].values()
                     if node.get('message') and node['message']['author']['role'] == 'user'
                     for part in node['message']['content'].get('parts', []) if isinstance(part, str)]

    print_time_notice("Analyzing conversations")
    analyzed_conversations = analyze_conversations(conversations)

    output_dir = './analyzed_conversations-2'
    print_time_notice(f"Saving analyzed conversations to {output_dir}")
    save_analyzed_conversations(analyzed_conversations, output_dir)

    print_time_notice("Visualizing message lengths")
    visualize_message_lengths(user_messages)

    print_time_notice("Detecting anomalies in conversations")
    detect_anomalies(conversations)

    print_time_notice("Analyzing sentiment trends in conversations")
    analyze_sentiment_trends(conversations)
4
    print_time_notice("Visualizing word frequencies in user messages")
    visualize_word_frequencies(user_messages)

    print_time_notice("Script completed")


if __name__ == "__main__":
    main()
