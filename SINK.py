import json
import os

import nltk
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)


# Function to extract keywords from text
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(10)]


# Function to generate a summary from text using LexRank
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])


# Extract and analyze conversation data
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
                messages.append({
                    'author': author_role,
                    'content': text
                })

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


# Define the output directory
output_dir = './analyzed_conversations-2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Analyze the conversations and save to new JSON files
analyzed_conversations = analyze_conversations(conversations_data)
for conversation in analyzed_conversations:
    output_file = os.path.join(output_dir, f"{conversation['title'][:10]}_{conversation['conversation_id']}.json")
    with open(output_file, 'w') as file:
        json.dump(conversation, file, indent=4)

print(f"Analyzed conversations have been saved to {output_dir}")
import json

import nltk

# Ensure the stopwords and other resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
user_messages = []
assistant_messages = []

for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            author_role = node['message']['author']['role']
            message_content = node['message']['content'].get('parts', [])
            if author_role == 'user':
                user_messages.extend(message_content)
            elif author_role == 'assistant':
                assistant_messages.extend(message_content)


# Function to perform text analysis
def analyze_text(messages, title):
    # Filter out non-text messages
    text_messages = [message for message in messages if isinstance(message, str)]

    # Join all messages into a single string
    all_text = ' '.join(text_messages)

    # Tokenize the text by words, converting to lowercase
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))

    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Count the total number of words and characters
    total_words = len(words)
    total_characters = sum(len(word) for word in words)

    # Count the frequency of each word
    word_counts = Counter(filtered_words)

    # Count the number of unique words
    unique_word_count = len(word_counts)

    # Get the 20 most common words
    common_words = word_counts.most_common(20)

    # Generate n-grams
    def get_ngrams(words, n):
        ngrams_list = ngrams(words, n)
        return Counter(ngrams_list).most_common(20)

    common_2grams = get_ngrams(filtered_words, 2)
    common_3grams = get_ngrams(filtered_words, 3)
    common_4grams = get_ngrams(filtered_words, 4)

    # Perform POS tagging
    pos_tags = pos_tag(filtered_words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    common_pos_tags = pos_counts.most_common(20)

    # Create and display a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Perform sentiment analysis on each text message
    sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

    # Calculate the overall sentiment
    overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    print(f"Overall sentiment for {title}: {overall_sentiment}")

    # Save the results to a file
    with open(f'analysis_results_{title}.txt', 'w') as result_file:
        result_file.write(f"Most common words for {title}:\n")
        for word, count in common_words:
            result_file.write(f"{word}: {count}\n")
        result_file.write(f"\nNumber of unique words: {unique_word_count}\n")
        result_file.write(f"\nMost common 2-grams:\n")
        for gram, count in common_2grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 3-grams:\n")
        for gram, count in common_3grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 4-grams:\n")
        for gram, count in common_4grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common POS tags:\n")
        for tag, count in common_pos_tags:
            result_file.write(f"{tag}: {count}\n")
        result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")
        result_file.write(f"\nTotal number of words: {total_words}\n")
        result_file.write(f"\nTotal number of characters: {total_characters}\n")


# Analyze user messages
analyze_text(user_messages, "User")

# Analyze assistant messages
analyze_text(assistant_messages, "Assistant")
import json

import nltk

# Ensure the stopwords and other resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
user_messages = []
assistant_messages = []

for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            author_role = node['message']['author']['role']
            message_content = node['message']['content'].get('parts', [])
            if author_role == 'user':
                user_messages.extend(message_content)
            elif author_role == 'assistant':
                assistant_messages.extend(message_content)


# Function to perform text analysis
def analyze_text(messages, title):
    # Filter out non-text messages
    text_messages = [message for message in messages if isinstance(message, str)]

    # Join all messages into a single string
    all_text = ' '.join(text_messages)

    # Tokenize the text by words, converting to lowercase
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))

    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Count the total number of words and characters
    total_words = len(words)
    total_characters = sum(len(word) for word in words)

    # Count the frequency of each word
    word_counts = Counter(filtered_words)

    # Count the number of unique words
    unique_word_count = len(word_counts)

    # Get the 20 most common words
    common_words = word_counts.most_common(20)

    # Generate n-grams
    def get_ngrams(words, n):
        ngrams_list = ngrams(words, n)
        return Counter(ngrams_list).most_common(20)

    common_2grams = get_ngrams(filtered_words, 2)
    common_3grams = get_ngrams(filtered_words, 3)
    common_4grams = get_ngrams(filtered_words, 4)

    # Perform POS tagging
    pos_tags = pos_tag(filtered_words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    common_pos_tags = pos_counts.most_common(20)

    # Create and display a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Perform sentiment analysis on each text message
    sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

    # Calculate the overall sentiment
    overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    print(f"Overall sentiment for {title}: {overall_sentiment}")

    # Word diversity (unique words / total words)
    word_diversity = unique_word_count / total_words if total_words else 0

    # Average sentence length in words
    sentences = re.split(r'[.!?]', all_text)
    avg_sentence_length = sum(len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences) / len(
        sentences) if sentences else 0

    # Sentiment distribution
    sentiment_distribution = Counter(round(sentiment, 1) for sentiment in sentiments)

    # Unique words per message
    unique_words_per_message = [len(set(re.findall(r'\b\w+\b', message.lower()))) for message in text_messages]
    avg_unique_words_per_message = sum(unique_words_per_message) / len(
        unique_words_per_message) if unique_words_per_message else 0

    # Keyword extraction using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf_matrix = vectorizer.fit_transform(text_messages)
    keywords = vectorizer.get_feature_names_out()

    # Save the results to a file
    with open(f'analysis_results_{title}.txt', 'w') as result_file:
        result_file.write(f"Most common words for {title}:\n")
        for word, count in common_words:
            result_file.write(f"{word}: {count}\n")
        result_file.write(f"\nNumber of unique words: {unique_word_count}\n")
        result_file.write(f"\nMost common 2-grams:\n")
        for gram, count in common_2grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 3-grams:\n")
        for gram, count in common_3grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 4-grams:\n")
        for gram, count in common_4grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common POS tags:\n")
        for tag, count in common_pos_tags:
            result_file.write(f"{tag}: {count}\n")
        result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")
        result_file.write(f"\nTotal number of words: {total_words}\n")
        result_file.write(f"\nTotal number of characters: {total_characters}\n")
        result_file.write(f"\nWord diversity: {word_diversity}\n")
        result_file.write(f"\nAverage sentence length: {avg_sentence_length} words\n")
        result_file.write(f"\nSentiment distribution:\n")
        for sentiment, count in sentiment_distribution.items():
            result_file.write(f"{sentiment}: {count}\n")
        result_file.write(f"\nAverage unique words per message: {avg_unique_words_per_message}\n")
        result_file.write(f"\nTop keywords (TF-IDF):\n")
        for keyword in keywords:
            result_file.write(f"{keyword}\n")


# Analyze user messages
analyze_text(user_messages, "User")

# Analyze assistant messages
analyze_text(assistant_messages, "Assistant")
import json

import nltk
from nltk import pos_tag
from nltk.util import ngrams

# Ensure the stopwords and other resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
user_messages = []
assistant_messages = []

for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            author_role = node['message']['author']['role']
            message_content = node['message']['content'].get('parts', [])
            if author_role == 'user':
                user_messages.extend(message_content)
            elif author_role == 'assistant':
                assistant_messages.extend(message_content)


# Function to perform text analysis
def analyze_text(messages, title):
    # Filter out non-text messages
    text_messages = [message for message in messages if isinstance(message, str)]

    # Join all messages into a single string
    all_text = ' '.join(text_messages)

    # Tokenize the text by words, converting to lowercase
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))

    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Count the frequency of each word
    word_counts = Counter(filtered_words)

    # Count the number of unique words
    unique_word_count = len(word_counts)

    # Get the 20 most common words
    common_words = word_counts.most_common(20)

    # Generate n-grams
    def get_ngrams(words, n):
        ngrams_list = ngrams(words, n)
        return Counter(ngrams_list).most_common(20)

    common_2grams = get_ngrams(filtered_words, 2)
    common_3grams = get_ngrams(filtered_words, 3)
    common_4grams = get_ngrams(filtered_words, 4)

    # Perform POS tagging
    pos_tags = pos_tag(filtered_words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    common_pos_tags = pos_counts.most_common(20)

    # Create and display a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Perform sentiment analysis on each text message
    sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

    # Calculate the overall sentiment
    overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    print(f"Overall sentiment for {title}: {overall_sentiment}")

    # Save the results to a file
    with open(f'analysis_results_{title}.txt', 'w') as result_file:
        result_file.write(f"Most common words for {title}:\n")
        for word, count in common_words:
            result_file.write(f"{word}: {count}\n")
        result_file.write(f"\nNumber of unique words: {unique_word_count}\n")
        result_file.write(f"\nMost common 2-grams:\n")
        for gram, count in common_2grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 3-grams:\n")
        for gram, count in common_3grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 4-grams:\n")
        for gram, count in common_4grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common POS tags:\n")
        for tag, count in common_pos_tags:
            result_file.write(f"{tag}: {count}\n")
        result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")


# Analyze user messages
analyze_text(user_messages, "User")

# Analyze assistant messages
analyze_text(assistant_messages, "Assistant")
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# Load data
with open('conversations.json', 'r') as file:
    data = json.load(file)

# Extract features (e.g., length of messages)
features = [len(conversation['mapping']) for conversation in data]

# Fit Isolation Forest
model = IsolationForest(contamination=0.1)
preds = model.fit_predict(np.array(features).reshape(-1, 1))

# Plot results
plt.scatter(range(len(features)), features, c=preds)
plt.title('Anomaly Detection in Conversations')
plt.xlabel('Conversation Index')
plt.ylabel('Number of Messages')
plt.show()

# Save the anomaly detection plot
plt.savefig('anomaly_detection.png')


def load_conversation(file_path):
    with open(file_path, 'r') as file:
        conversation_data = json.load(file)
    return conversation_data


def format_conversation(conversation):
    formatted_conversation = f"Conversation ID: {conversation['conversation_id']}\n"
    formatted_conversation += f"Title: {conversation['title']}\n\n"

    for message in conversation['messages']:
        author = message['author']
        content_parts = message['content']

        # Handle content parts which might be dicts or strings
        if content_parts and isinstance(content_parts[0], dict):
            content = "\n".join([part.get('text', '') for part in content_parts])
        else:
            content = "\n".join(content_parts)

        create_time = message['create_time']
        update_time = message['update_time']

        formatted_conversation += f"Author: {author}\n"
        formatted_conversation += f"Content: {content}\n"
        formatted_conversation += f"Create Time: {create_time}\n"
        formatted_conversation += f"Update Time: {update_time}\n"
        formatted_conversation += "-" * 40 + "\n"

    return formatted_conversation


def main():
    # Prompt the user for the path to the gptlogs directory
    gptlogs_dir = input("Enter the path to the gptlogs directory: ").strip()

    # List the JSON files in the gptlogs directory
    json_files = [f for f in os.listdir(gptlogs_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the specified directory.")
        return

    # Display the available JSON files
    print("Available JSON files:")
    for idx, json_file in enumerate(json_files):
        print(f"{idx + 1}. {json_file}")

    # Prompt the user to select a JSON file
    file_index = int(input("Enter the number of the JSON file to load: ")) - 1

    if file_index < 0 or file_index >= len(json_files):
        print("Invalid selection.")
        return

    # Load the selected JSON file
    selected_file = json_files[file_index]
    file_path = os.path.join(gptlogs_dir, selected_file)

    conversation = load_conversation(file_path)

    # Format the conversation for readability
    formatted_conversation = format_conversation(conversation)

    # Output the formatted conversation
    print(formatted_conversation)


if __name__ == "__main__":
    main()
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract conversation summaries
summaries = []
titles = []
for conversation in conversations_data:
    titles.append(conversation['title'])
    summary = ' '.join(
        [
            part
            for node in conversation['mapping'].values()
            if node.get('message') and 'parts' in node['message']['content']
            for part in node['message']['content']['parts']
            if isinstance(part, str)
        ]
    )
    summaries.append(summary)

# Vectorize the summaries
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(summaries)

# Perform KMeans clustering
num_clusters = 32
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Create a DataFrame for the clustering results
cluster_df = pd.DataFrame({'Title': titles, 'Cluster': labels})

# Plot the clustering results
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=cluster_df)
plt.title('Conversation Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Conversations')
plt.savefig('conversation_clusters.png')
plt.show()
import json
import re
from collections import Counter

import matplotlib.pyplot as plt

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)


# Function to detect code language
def detect_code_language(code):
    if re.search(r'\bimport\b|\bdef\b|\bclass\b|\bprint\b', code):
        return 'Python'
    elif re.search(r'<\/?\w+>', code):
        return 'HTML'
    elif re.search(r'\bfunction\b|\bvar\b|\bconsole\.log\b', code):
        return 'JavaScript'
    elif re.search(r'\bif\b|\bfi\b|\b#!/bin/bash\b|\bdo\b', code):
        return 'Bash'
    elif re.search(r'\bcolor\b|\bbackground\b|\bfont\b|\bpadding\b|\bmargin\b', code):
        return 'CSS'
    elif re.search(r'<\?php\b|\b\$[a-zA-Z_]\w*\b', code):
        return 'PHP'
    elif re.search(r'\bSELECT\b|\bFROM\b|\bWHERE\b', code):
        return 'SQL'
    elif re.search(r'\{\s*"\w+"\s*:', code):
        return 'JSON'
    elif re.search(r'---\s*\n', code):
        return 'YAML'
    elif re.search(r'<\?xml\b', code):
        return 'XML'
    elif re.search(r'#\s*\w+', code):
        return 'Markdown'
    elif re.search(r'\bsh\b|\bexport\b|\bunset\b', code):
        return 'Shell'
    elif re.search(r'\bdef\b|\bputs\b|\bend\b', code):
        return 'Ruby'
    elif re.search(r'\b(sub|print|end)\b|\b(main|cout|int)\b|\b(procedure|data division)\b', code):
        return 'Ancient'
    elif re.search(r'\bpublic\b|\bclass\b|\bvoid\b|\bnew\b', code):
        return 'Java'
    elif re.search(r'\bfunc\b|\bpackage\b|\bimport\b', code):
        return 'Go'
    elif re.search(r'\binterface\b|\bimplements\b|\bconst\b|\bimport\b', code):
        return 'TypeScript'
    elif re.search(r'\bval\b|\bvar\b|\bdef\b|\bobject\b', code):
        return 'Scala'
    elif re.search(r'\bfun\b|\bval\b|\bvar\b|\bimport\b', code):
        return 'Kotlin'
    elif re.search(r'\blibrary\b|\bdata\b|\bprint\b', code):
        return 'R'
    elif re.search(r'\bfunction\b|\bend\b|\bplot\b|\bmatrix\b', code):
        return 'MATLAB'
    else:
        return 'Other'


# Extract and analyze assistant messages
language_counts = []
tool_counts = 0
for conversation in conversations_data:
    conversation_id = conversation['id']
    lang_counter = Counter()
    for node in conversation['mapping'].values():
        if node.get('message') and node['message']['content']:
            author_role = node['message']['author']['role']
            if author_role == 'assistant':
                message_content = node['message']['content'].get('parts', [])
                for part in message_content:
                    code_snippets = re.findall(r'```(.*?)```', part, re.DOTALL)
                    for snippet in code_snippets:
                        lang = detect_code_language(snippet)
                        lang_counter[lang] += 1
            elif author_role == 'tool':
                tool_counts += 1
    language_counts.append((conversation_id, lang_counter))

# Count the total number of code snippets by language
total_counts = Counter()
for _, lang_counter in language_counts:
    total_counts.update(lang_counter)

# Add tool count to total counts
total_counts['Tool'] = tool_counts

# Write the results to a file
with open('code_snippet_counts.txt', 'w') as result_file:
    result_file.write("Code Snippet Counts by Language for Each Conversation\n")
    result_file.write("=" * 50 + "\n")
    for conversation_id, lang_counter in language_counts:
        result_file.write(f"\nConversation ID: {conversation_id}\n")
        for lang, count in lang_counter.items():
            result_file.write(f"  {lang}: {count}\n")
    result_file.write("\n\nTotal code snippets by language:\n")
    for lang, count in total_counts.items():
        result_file.write(f"  {lang}: {count}\n")

# Combine languages with fewer than 26 occurrences into 'Other' for the pie chart
pie_counts = total_counts.copy()
for lang, count in list(pie_counts.items()):
    if count < 26:
        pie_counts['Other'] += count
        del pie_counts[lang]

# Generate a pie chart
labels = [lang for lang in pie_counts if lang != 'None']
sizes = [count for lang, count in pie_counts.items() if lang != 'None']
plt.figure(figsize=(10, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribution of Code Snippets by Language')
plt.savefig('code_snippets_pie_chart.png')
plt.show()

# Print the totals for each language
print("Total code snippets by language:")
for lang, count in total_counts.items():
    print(f"{lang}: {count}")
import json
from collections import Counter

import matplotlib.pyplot as plt
from textblob import TextBlob

# Load data
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Analyze context of specific keywords
keyword = 'help'
contexts = []

for conversation in data:
    for node_id, node in conversation['mapping'].items():
        message = node.get('message')
        if message and 'content' in message:
            content_parts = message['content'].get('parts', [])
            if isinstance(content_parts, list) and all(isinstance(part, str) for part in content_parts):
                text = ' '.join(content_parts)
                if keyword in text:
                    contexts.append(text)
            else:
                print(f"Unexpected content format in node {node_id}: {message['content']}")

# Print keyword contexts
for context in contexts:
    print(context)
    print()

# Count the occurrences of each keyword context
context_counter = Counter(contexts)
print(context_counter)

# Save the keyword context counts to a file
with open('/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/keyword_contexts.json', 'w') as file:
    json.dump(context_counter, file)

# Analyze sentiment trends
for i, conversation in enumerate(data):
    timestamps = []
    sentiments = []
    for node_id, node in conversation['mapping'].items():
        message = node.get('message')
        if message and 'content' in message:
            content_parts = message['content'].get('parts', [])
            if isinstance(content_parts, list) and all(isinstance(part, str) for part in content_parts):
                text = ' '.join(content_parts)
                sentiment = TextBlob(text).sentiment.polarity
                timestamps.append(message['create_time'])
                sentiments.append(sentiment)
            else:
                print(f"Unexpected content format in node {node_id}: {message['content']}")
    if timestamps:
        plt.plot(timestamps, sentiments, label=f'Conversation {i + 1}')

plt.title('Sentiment Trends in Conversations')
plt.xlabel('Timestamp')
plt.ylabel('Sentiment Polarity')
plt.legend()
plt.savefig('/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/sentiment_trend.png')
plt.show()
import json
import os

import nltk
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)


# Function to extract keywords from text
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(10)]


# Function to generate a summary from text using sumy
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])


# Extract and analyze conversation data
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
                messages.append({
                    'author': author_role,
                    'content': text
                })

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


# Define the output directory
output_dir = './analyzed_conversations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Analyze the conversations and save to new JSON files
analyzed_conversations = analyze_conversations(conversations_data)
for conversation in analyzed_conversations:
    output_file = os.path.join(output_dir, f"{conversation['title'][:10]}_{conversation['conversation_id']}.json")
    with open(output_file, 'w') as file:
        json.dump(conversation, file, indent=4)

print(f"Analyzed conversations have been saved to {output_dir}")
import json

import nltk

# Ensure the stopwords and other resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
user_messages = []
assistant_messages = []

for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            author_role = node['message']['author']['role']
            message_content = node['message']['content'].get('parts', [])
            if author_role == 'user':
                user_messages.extend(message_content)
            elif author_role == 'assistant':
                assistant_messages.extend(message_content)


# Function to perform text analysis
def analyze_text(messages, title):
    # Filter out non-text messages
    text_messages = [message for message in messages if isinstance(message, str)]

    # Join all messages into a single string
    all_text = ' '.join(text_messages)

    # Tokenize the text by words, converting to lowercase
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))

    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Calculate lexical richness (type-token ratio)
    unique_words = set(filtered_words)
    lexical_richness = len(unique_words) / len(filtered_words) if filtered_words else 0

    # Identify the longest and shortest messages
    longest_message = max(text_messages, key=lambda msg: len(re.findall(r'\b\w+\b', msg)))
    shortest_message = min(text_messages, key=lambda msg: len(re.findall(r'\b\w+\b', msg)))

    # Count the number of questions asked
    question_count = sum(1 for msg in text_messages if '?' in msg)

    # Emotion analysis
    emotions = {
        'joy': 0,
        'sadness': 0,
        'anger': 0,
        'fear': 0,
        'disgust': 0
    }

    # Use a basic dictionary-based approach for emotion analysis
    emotion_words = {
        'joy': ['happy', 'joy', 'pleased', 'delighted', 'excited', 'glad'],
        'sadness': ['sad', 'unhappy', 'sorrow', 'down', 'depressed', 'miserable'],
        'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
        'fear': ['fear', 'afraid', 'scared', 'terrified', 'worried', 'nervous'],
        'disgust': ['disgust', 'revolted', 'nauseated', 'repulsed', 'sickened', 'distaste']
    }

    for word in filtered_words:
        for emotion, words in emotion_words.items():
            if word in words:
                emotions[emotion] += 1

    # Save the results to a file
    with open(f'fun_analysis_results_{title}.txt', 'w') as result_file:
        result_file.write(f"Lexical richness (TTR) for {title}: {lexical_richness}\n")
        result_file.write(f"\nLongest message for {title}:\n{longest_message}\n")
        result_file.write(f"\nShortest message for {title}:\n{shortest_message}\n")
        result_file.write(f"\nNumber of questions asked by {title}: {question_count}\n")
        result_file.write(f"\nEmotion analysis for {title}:\n")
        for emotion, count in emotions.items():
            result_file.write(f"{emotion}: {count}\n")


# Analyze user messages
analyze_text(user_messages, "User")

# Analyze assistant messages
analyze_text(assistant_messages, "Assistant")
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
from textblob import TextBlob

from wordcloud import WordCloud

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
messages = []
for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            message_content = node['message']['content'].get('parts', [])
            messages.extend(message_content)

# Filter out non-text messages
text_messages = [message for message in messages if isinstance(message, str)]

# Join all messages into a single string
all_text = ' '.join(text_messages)

# Tokenize the text by words, converting to lowercase
words = re.findall(r'\b\w+\b', all_text.lower())

# Count the frequency of each word
word_counts = Counter(words)

# Get the 20 most common words
common_words = word_counts.most_common(20)

# Display the most common words
print("Most common words:")
for word, count in common_words:
    print(f"{word}: {count}")

# Create and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Perform sentiment analysis on each text message
sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

# Calculate the overall sentiment
overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
print(f"Overall sentiment: {overall_sentiment}")

# Save the results to a file
with open('analysis_results.txt', 'w') as result_file:
    result_file.write("Most common words:\n")
    for word, count in common_words:
        result_file.write(f"{word}: {count}\n")
    result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

# Ensure the stopwords are downloaded
nltk.download('stopwords')

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
messages = []
for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            message_content = node['message']['content'].get('parts', [])
            messages.extend(message_content)

# Filter out non-text messages
text_messages = [message for message in messages if isinstance(message, str)]

# Join all messages into a single string
all_text = ' '.join(text_messages)

# Tokenize the text by words, converting to lowercase
words = re.findall(r'\b\w+\b', all_text.lower())

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

# Filter out stop words
filtered_words = [word for word in words if word not in stop_words]

# Count the frequency of each word
word_counts = Counter(filtered_words)

# Get the 20 most common words
common_words = word_counts.most_common(1000)

# Display the most common words
print("Most common words:")
for word, count in common_words:
    print(f"{word}: {count}")

# Create and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Perform sentiment analysis on each text message
sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

# Calculate the overall sentiment
overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
print(f"Overall sentiment: {overall_sentiment}")

# Save the results to a file
with open('analysis_results.txt', 'w') as result_file:
    result_file.write("Most common words:\n")
    for word, count in common_words:
        result_file.write(f"{word}: {count}\n")
    result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")
import json

import graphviz

# Load data
file_path = 'conversations.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Create a flowchart for a single conversation
conversation = data[0]
dot = graphviz.Digraph(comment='Conversation Flow')

for node_id, node in conversation['mapping'].items():
    message = node.get('message')
    if message:
        author = message['author']['role']
        content = ' '.join(message['content'].get('parts', []))
        dot.node(node_id, f"{author}: {content}")

    parent_id = node.get('parent')
    if parent_id:
        dot.edge(parent_id, node_id)

# Save and render the flowchart
dot.render('conversation_flowchart', format='png')

# Display the flowchart
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('conversation_flowchart.png')
plt.imshow(image)
plt.axis('off')
plt.show()
import math

import cv2


def load_images(image_dir, size=(200, 300)):
    images = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, size)
            images.append(resized_image)
    return images


def create_image_grid(images, image_size=(200, 300)):
    img_width, img_height = image_size
    num_images = len(images)

    # Calculate the grid size
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    # Create a blank canvas
    stitched_image = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype=np.uint8)

    # Place images on the canvas
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        stitched_image[y_start:y_end, x_start:x_end] = image

    return stitched_image


def main():
    image_dir = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/bookcovers'  # Replace with your image directory
    image_size = (200, 300)  # Define the size to which each image will be resized

    images = load_images(image_dir, size=image_size)

    # Check if we have enough images for the grid
    if not images:
        print("No images found in the directory.")
        return

    stitched_image = create_image_grid(images, image_size=image_size)

    # Save the stitched image
    output_path = 'stitched_bookshelf.jpg'
    cv2.imwrite(output_path, stitched_image)
    print(f'Stitched image saved as {output_path}')

    # Display the stitched image
    cv2.imshow('Stitched Image', stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# heater.py
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import seaborn as sns
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


# Preprocess text
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


# Extract user and assistant tokens
user_tokens = []
assistant_tokens = []

print("Preprocessing conversations...")
start_time = time.time()
for conversation in tqdm(data, desc="Conversations", unit="conv"):
    for message in conversation['mapping'].values():
        if message.get('message'):
            content_parts = message['message'].get('content', {}).get('parts', [])
            text = ' '.join([part for part in content_parts if isinstance(part, str)])
            tokens = preprocess(text)
            if message['message']['author']['role'] == 'user':
                user_tokens.append(tokens[:512])  # First 5 tokens of user messages
            elif message['message']['author']['role'] == 'assistant':
                mid_index = len(tokens) // 2
                assistant_tokens.append(
                    tokens[mid_index - 256:mid_index + 255])  # Middle 5 tokens of assistant responses
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
plt.title('Heatmap of Similarity Between User Message Tokens and Assistant Response Tokens')
plt.xlabel('Assistant Response Tokens')
plt.ylabel('User Message Tokens')

# Get the base directory and filename
base_dir = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/'
base_filename = 'similarity_heatmap'

# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the sequential filename
sequential_filename = f"{base_filename}_{timestamp}.png"

# Combine the base directory and sequential filename to get the output path
output_path = os.path.join(base_dir, sequential_filename)

# Save the file using the output_path
plt.savefig(output_path)
print(f"Heatmap saved to {output_path}")

plt.show()
# heater2.py
import json
import multiprocessing

import nltk
from nltk.stem import WordNetLemmatizer

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
# heater.py

import multiprocessing
import os

import nltk
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the directory containing JSON files
directory = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations-2'


# Preprocess the text data
def preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)


# Function to calculate sentiment
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Function to calculate complexity
def calculate_complexity(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    total_words = len(words)
    noun_count = len([word for word, pos in pos_tags if pos.startswith('NN')])
    verb_count = len([word for word, pos in pos_tags if pos.startswith('VB')])
    if total_words > 0:
        complexity_score = (noun_count + verb_count) / total_words
    else:
        complexity_score = 0
    return complexity_score


# Analyze the data
def analyze_data(file_path, result_queue):
    with open(file_path, 'r') as file:
        conversation = json.load(file)

    user_message = ''
    assistant_response = ''
    for message in conversation.get('messages', []):
        content = ' '.join(message.get('content', []))
        if message.get('author') == 'user':
            user_message += content + ' '
        elif message.get('author') == 'assistant':
            assistant_response += content + ' '

    sentiment_score = calculate_sentiment(user_message)
    complexity_score = calculate_complexity(user_message)
    creativity_score = len(set(preprocess(user_message))) / len(preprocess(user_message)) if len(
        preprocess(user_message)) > 0 else 0
    conversation_length = len(preprocess(user_message)) + len(preprocess(assistant_response))

    result_queue.put((user_message.strip(), assistant_response.strip(), sentiment_score, complexity_score,
                      creativity_score, conversation_length))


# Create the heatmaps
def create_heatmaps(user_messages, assistant_responses, sentiment_scores, complexity_scores, creativity_scores,
                    conversation_lengths):
    # Sentiment vs. Complexity
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([sentiment_scores, complexity_scores]).T, cmap='coolwarm', cbar=True)
    plt.title('Sentiment vs. Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Sentiment')
    plt.savefig('sentiment_vs_complexity.png')
    plt.close()

    # Creativity vs. Conversation Length
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([creativity_scores, conversation_lengths]).T, cmap='coolwarm', cbar=True)
    plt.title('Creativity vs. Conversation Length')
    plt.xlabel('Conversation Length')
    plt.ylabel('Creativity')
    plt.savefig('creativity_vs_conversation_length.png')
    plt.close()

    # Sentiment vs. Creativity vs. Complexity
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sentiment_scores, creativity_scores, complexity_scores, c=conversation_lengths, cmap='coolwarm')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Creativity')
    ax.set_zlabel('Complexity')
    plt.title('Sentiment vs. Creativity vs. Complexity')
    plt.savefig('sentiment_vs_creativity_vs_complexity.png')
    plt.close()

    # Topic Similarity vs. Conversation Length
    vectorizer = TfidfVectorizer()
    user_tfidf_matrix = vectorizer.fit_transform(user_messages)
    assistant_tfidf_matrix = vectorizer.transform(assistant_responses)
    similarity_scores = cosine_similarity(user_tfidf_matrix, assistant_tfidf_matrix).diagonal()

    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([similarity_scores, conversation_lengths]).T, cmap='coolwarm', cbar=True)
    plt.title('Topic Similarity vs. Conversation Length')
    plt.xlabel('Conversation Length')
    plt.ylabel('Topic Similarity')
    plt.savefig('topic_similarity_vs_conversation_length.png')
    plt.close()

    # Creativity vs. Complexity
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([creativity_scores, complexity_scores]).T, cmap='coolwarm', cbar=True)
    plt.title('Creativity vs. Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Creativity')
    plt.savefig('creativity_vs_complexity.png')
    plt.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Create a shared queue using Manager
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    # Get the list of JSON files
    json_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]

    # Create a process pool
    with multiprocessing.Pool() as pool:
        # Process the files in parallel
        pool.starmap(analyze_data, [(file, result_queue) for file in tqdm(json_files, desc="Processing files")])

    # Collect the results from the queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Combine the results
    user_messages = []
    assistant_responses = []
    sentiment_scores = []
    complexity_scores = []
    creativity_scores = []
    conversation_lengths = []

    for result in results:
        user_messages.append(result[0])
        assistant_responses.append(result[1])
        sentiment_scores.append(result[2])
        complexity_scores.append(result[3])
        creativity_scores.append(result[4])
        conversation_lengths.append(result[5])

    # Create the heatmaps
    create_heatmaps(user_messages, assistant_responses, sentiment_scores, complexity_scores, creativity_scores,
                    conversation_lengths)
import json
import time
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from transformers import pipeline

# Load data
with open('/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json', 'r') as file:
    data = json.load(file)

# Load intent recognition model
nlp = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Define possible intents
candidate_labels = ["question", "request", "feedback", "greeting"]


def process_message(message):
    msg = message.get('message')
    if msg and 'content' in msg:
        content_parts = msg['content'].get('parts', [])
        text = ' '.join([part for part in content_parts if isinstance(part, str)])
        if text.strip():
            result = nlp(text, candidate_labels)
            return {'text': text, 'intent': result['labels'][0]}
    return None


def process_conversation(conversation):
    intents = []
    for message in conversation['mapping'].values():
        intent = process_message(message)
        if intent:
            intents.append(intent)
    return intents


# Parallel processing
if __name__ == "__main__":
    start_time = time.time()

    try:
        # Use a context manager to initialize tqdm
        with Pool(cpu_count()) as pool:
            results = list(
                tqdm(pool.imap(process_conversation, data), total=len(data), desc="Processing Conversations"))

        # Flatten results
        intents = [item for sublist in results for item in sublist]

        # Print intents (optional, may be skipped to save time)
        for item in intents:
            print(f"Text: {item['text']}")
            print(f"Intent: {item['intent']}")
            print()

        # Save the intent recognition results to a file
        with open('intent_recognition.json', 'w') as file:
            json.dump(intents, file)

    except Exception as e:
        print(f"An error occurred: {e}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract user messages and analyze message length
user_messages = []
for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message']['author']['role'] == 'user':
            message_content = node['message']['content'].get('parts', [])
            for part in message_content:
                if isinstance(part, str):  # Ensure the content part is a string
                    user_messages.append(part)

# Calculate message length statistics
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

# Create a DataFrame for message lengths
message_lengths_df = pd.DataFrame({'Message Length': message_lengths})

# Plot the distribution of message lengths
plt.figure(figsize=(10, 6))
plt.hist(message_lengths, bins=20, edgecolor='black')
plt.title('Distribution of User Message Lengths')
plt.xlabel('Message Length (characters)')
plt.ylabel('Frequency')
plt.savefig('user_message_length_distribution.png')
plt.show()

# Calculate percentiles of message lengths
percentiles = [25, 50, 75, 90, 95, 99]
percentile_values = np.percentile(message_lengths, percentiles)

print("\nMessage Length Percentiles:")
for p, v in zip(percentiles, percentile_values):
    print(f"{p}th percentile: {v:.2f} characters")

# Plot the cumulative distribution of message lengths
sorted_lengths = np.sort(message_lengths)
cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

plt.figure(figsize=(10, 6))
plt.plot(sorted_lengths, cumulative_prob)
plt.title('Cumulative Distribution of User Message Lengths')
plt.xlabel('Message Length (characters)')
plt.ylabel('Cumulative Probability')
plt.savefig('user_message_length_cumulative_distribution.png')
plt.show()
import csv
import heapq
import json
from datetime import datetime

# Load data
file_path = 'conversations.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a min-heap to keep track of the top 10 longest assistant messages
top_assistant_messages = []

for conversation in data:
    for message in conversation['mapping'].values():
        if message.get('message') and message['message']['author']['role'] == 'assistant':
            message_id = message['message']['id']
            text_parts = message['message']['content'].get('parts', [])
            if text_parts:
                text = ' '.join([part for part in text_parts if isinstance(part, str)])
                char_count = len(text)
                if len(top_assistant_messages) < 500:
                    heapq.heappush(top_assistant_messages, (char_count, message_id, text))
                else:
                    heapq.heappushpop(top_assistant_messages, (char_count, message_id, text))

# Convert heap to a sorted list of messages
top_assistant_messages.sort(reverse=True, key=lambda x: x[0])

# Save the top 10 longest assistant messages to a JSON file
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_path_json = f'top_10_assistant_messages_{timestamp}.json'
with open(output_path_json, 'w') as file:
    json.dump(top_assistant_messages, file)
print(f"Top 10 longest assistant messages saved to {output_path_json}")

# Save the top 10 longest assistant messages to a CSV file
output_path_csv = f'top_10_assistant_messages_{timestamp}.csv'
with open(output_path_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Rank', 'Message ID', 'Character Count'])
    for rank, (char_count, message_id, text) in enumerate(top_assistant_messages, start=1):
        csv_writer.writerow([rank, message_id, char_count])
print(f"Top 10 longest assistant messages saved to {output_path_csv}")
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/top_10_assistant_messages_20240520214620.csv'
df = pd.read_csv(file_path)

# Sort the dataframe by character count in descending order
df_sorted = df.sort_values(by='Character Count', ascending=False).head(500)

# Visualize the top 500 assistant messages by character count in a bar chart
plt.figure(figsize=(12, 18))
plt.barh(df_sorted['Message ID'], df_sorted['Character Count'], color='skyblue')

# Set y-ticks to show every 50th message ID
plt.yticks(df_sorted['Message ID'][::50])

plt.xlabel('Character Count')
plt.ylabel('Message ID')
plt.title('Top 500 Assistant Messages by Character Count')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot as an image
output_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/top_500_assistant_messages.png'
plt.savefig(output_path)
plt.show()

output_path
import json

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)


# Function to calculate response quality
def calculate_response_quality(messages):
    qualities = []
    for message in messages:
        if message['author'] == 'assistant':
            for part in message['content']:
                polarity = TextBlob(part).sentiment.polarity
                qualities.append(polarity)
    return np.mean(qualities) if qualities else 0


import json
import time
import xml.etree.ElementTree as ET
from collections import Counter

import plotly.graph_objects as go
import umap
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set the path to the XML file
xml_file_path = "/Users/puppuccino/PycharmProjects/inner_mon/archive_key_sum.xml"

# Initialize lists to store conversation data
conversations = []
soft_errors = []
hard_errors = []

# Parse the XML file
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Define custom stop words
custom_stop_words = [
    'render', 'user', 'requests', 'generate', 'images', 'content', 'policy', 'explain', 'unable', 'please'
]

# Iterate through each conversation in the XML file
for conv in tqdm(root.findall('Conversation'), desc="Processing XML data"):
    try:
        conversation_id = conv.find('ConversationID').text if conv.find('ConversationID') is not None else ""
        keywords = conv.find('Keywords').text.split(', ') if conv.find('Keywords') is not None else []
        summary = conv.find('Summary').text if conv.find('Summary') is not None else ""

        conversation = {
            "conversation_id": conversation_id,
            "title": "",  # Title is not provided in the XML, so we set it as an empty string
            "messages": [],  # Messages are not provided in the XML, so we set it as an empty list
            "keywords": keywords,
            "summary": summary
        }

        # Check for missing fields and log soft errors
        missing_fields = []
        if not conversation["conversation_id"]:
            missing_fields.append('conversation_id')
        if not conversation["keywords"]:
            missing_fields.append('keywords')
        if not conversation["summary"]:
            missing_fields.append('summary')

        if missing_fields:
            soft_errors.append(f"Missing {', '.join(missing_fields)} in conversation with ID {conversation_id}")

        conversations.append(conversation)

    except Exception as e:
        hard_errors.append(f"Error processing conversation: {str(e)}")

# Extract features from conversations
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
features = vectorizer.fit_transform([f"{conv['summary']} {' '.join(conv['keywords'])}" for conv in conversations])

# Calculate similarity scores
similarity_matrix = cosine_similarity(features)

# Normalize the similarity matrix
scaler = StandardScaler(with_mean=True)  # with_mean=False to avoid dense matrix creation
normalized_similarity_matrix = scaler.fit_transform(similarity_matrix)

# Use UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_components=3)
start_time = time.time()
reduced_features = umap_reducer.fit_transform(normalized_similarity_matrix)
end_time = time.time()
reduction_time = end_time - start_time

# Perform clustering on the reduced features
n_clusters = 32  # Adjust the number of clusters based on your data
kmeans = KMeans(n_clusters=n_clusters, random_state=40)
labels = kmeans.fit_predict(reduced_features)

# Add the cluster labels to the conversations
for i, conversation in enumerate(conversations):
    conversation['cluster'] = int(labels[i])

# Analyze the centroids to understand the main topics or themes
centroids = kmeans.cluster_centers_

# Find the top keywords in each cluster
cluster_keywords = []
for i in range(n_clusters):
    cluster_conversations = [conv for conv in conversations if conv['cluster'] == i]
    all_keywords = [keyword for conv in cluster_conversations for keyword in conv.get('keywords', [])]
    if all_keywords:
        top_keywords = Counter(all_keywords).most_common(500)  # Adjust the number of top keywords
    else:
        top_keywords = []
    cluster_keywords.append(top_keywords)

# Print the top keywords for each cluster
for i, keywords in enumerate(cluster_keywords):
    print(f"Cluster {i} Keywords: {keywords}")

# Create a 3D scatter plot with cluster labels and top keywords
fig = go.Figure()

for i in range(n_clusters):
    cluster_points = reduced_features[labels == i]
    hover_text = [f"Cluster {i}<br>Keywords: {', '.join([kw for kw, _ in cluster_keywords[i]])}" for _ in
                  range(len(cluster_points))]
    fig.add_trace(go.Scatter3d(
        x=cluster_points[:, 0],
        y=cluster_points[:, 1],
        z=cluster_points[:, 2],
        mode='markers',
        marker=dict(size=3),
        name=f"Cluster {i}",
        text=hover_text,
        hoverinfo='text'
    ))

# Set plot layout
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Save the plot as an HTML file
timestamp = time.strftime("%Y%m%d_%H%M%S")
plot_filename = f"clustered_conversation_plot_{timestamp}.html"
fig.write_html(plot_filename)

# Save the reduced features as a JSON file
reduced_features_filename = f"reduced_features_{timestamp}.json"
with open(reduced_features_filename, "w") as file:
    json.dump(reduced_features.tolist(), file)

# Save the soft errors to a file
if soft_errors:
    soft_error_filename = f"soft_errors_{timestamp}.txt"
    with open(soft_error_filename, "w") as file:
        file.write("\n".join(soft_errors))

# Save the hard errors to a file
if hard_errors:
    hard_error_filename = f"hard_errors_{timestamp}.txt"
    with open(hard_error_filename, "w") as file:
        file.write("\n".join(hard_errors))

# Print summary information
print(f"Processed {len(conversations)} conversations.")
print(f"Dimensionality reduction time: {reduction_time:.2f} seconds.")
print(f"Clustered plot saved as: {plot_filename}")
print(f"Reduced features saved as: {reduced_features_filename}")
if soft_errors:
    print(f"Soft errors saved as: {soft_error_filename}")
if hard_errors:
    print(f"Hard errors saved as: {hard_error_filename}")
import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from textblob import TextBlob
from transformers import pipeline

# Load data
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Sidebar for selecting the analysis type
analysis_type = st.sidebar.selectbox("Select Analysis Type",
                                     ["Anomaly Detection", "Keyword Context", "Sentiment Trend", "Intent Recognition"])

# Anomaly Detection
if analysis_type == "Anomaly Detection":
    num_messages = [len(conversation['mapping']) for conversation in data]

    # Fit the isolation forest model
    model = IsolationForest(contamination=0.1)
    num_messages = [[num] for num in num_messages]
    model.fit(num_messages)
    anomalies = model.predict(num_messages)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(num_messages)), [num[0] for num in num_messages], c=anomalies, cmap='viridis')
    plt.title('Anomaly Detection in Conversations')
    plt.xlabel('Conversation Index')
    plt.ylabel('Number of Messages')
    st.pyplot(plt)

# Keyword Context Analysis
elif analysis_type == "Keyword Context":
    keyword = st.text_input("Enter keyword for context analysis", "help")
    if st.button("Analyze Keyword Context"):
        contexts = []
        for conversation in data:
            for message in conversation['mapping'].values():
                msg = message.get('message')
                if msg and 'content' in msg:
                    content_parts = msg['content'].get('parts', [])
                    text = ' '.join([part for part in content_parts if isinstance(part, str)])
                    if keyword in text:
                        contexts.append(text)
        context_counter = Counter(contexts)
        st.write(context_counter)
        st.json(context_counter)

# Sentiment Trend Analysis
elif analysis_type == "Sentiment Trend":
    max_conversations = st.slider("Select number of conversations to plot", 1, 20, 10)
    if st.button("Analyze Sentiment Trends"):
        plot_number = 1
        conversations_per_plot = []
        for i, conversation in enumerate(data[:max_conversations]):
            timestamps = []
            sentiments = []
            for node_id, node in conversation['mapping'].items():
                message = node.get('message')
                if message and 'content' in message:
                    content_parts = message['content'].get('parts', [])
                    text = ' '.join([part for part in content_parts if isinstance(part, str)])
                    sentiment = TextBlob(text).sentiment.polarity
                    timestamps.append(message['create_time'])
                    sentiments.append(sentiment)
            if timestamps:
                df = pd.DataFrame({'timestamp': timestamps, 'sentiment': sentiments})
                df = df.sort_values(by='timestamp')
                df['rolling_sentiment'] = df['sentiment'].rolling(window=5, min_periods=1).mean()
                conversations_per_plot.append((df, f'Conversation {i + 1}'))

        plt.figure(figsize=(14, 7))
        for df, label in conversations_per_plot:
            plt.plot(df['timestamp'], df['rolling_sentiment'], label=label, alpha=0.7)
        plt.title(f'Sentiment Trends in Conversations')
        plt.xlabel('Timestamp')
        plt.ylabel('Sentiment Polarity')
        plt.legend()
        st.pyplot(plt)

# Intent Recognition
elif analysis_type == "Intent Recognition":
    nlp = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    candidate_labels = ["question", "request", "feedback", "greeting"]
    if st.button("Analyze Intents"):
        intents = []
        for conversation in data:
            for message in conversation['mapping'].values():
                msg = message.get('message')
                if msg and 'content' in msg:
                    content_parts = msg['content'].get('parts', [])
                    text = ' '.join([part for part in content_parts if isinstance(part, str)])
                    if text.strip():
                        result = nlp(text, candidate_labels)
                        intents.append({'text': text, 'intent': result['labels'][0]})
        st.json(intents)

# Upload new data
st.sidebar.markdown("## Upload New Data")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = json.load(uploaded_file)
    st.sidebar.success("Data Loaded!")
import json

import matplotlib.pyplot as plt
import pandas as pd

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)


# Function to calculate text length
def calculate_text_length(messages, role):
    total_length = 0
    for message in messages:
        if message['author'] == role:
            for part in message['content']:
                total_length += len(part)
    return total_length


# Extract and analyze conversation data
conversation_stats = []
for conversation in conversations_data:
    conversation_id = conversation['id']
    title = conversation.get('title', 'Untitled')
    messages = []
    for node in conversation['mapping'].values():
        if node.get('message') and node['message']['content']:
            message_content = node['message']['content'].get('parts', [])
            author_role = node['message']['author']['role']
            messages.append({
                'author': author_role,
                'content': message_content
            })

    interaction_count = len(messages)
    assistant_text_length = calculate_text_length(messages, 'assistant')
    user_text_length = calculate_text_length(messages, 'user')

    conversation_stats.append({
        'conversation_id': conversation_id,
        'title': title,
        'interaction_count': interaction_count,
        'assistant_text_length': assistant_text_length,
        'user_text_length': user_text_length
    })

# Convert to DataFrame
df = pd.DataFrame(conversation_stats)

# Get top 25 longest conversations by number of interactions
top_25_interactions = df.nlargest(25, 'interaction_count')

# Get top 25 longest conversations by assistant text length
top_25_assistant_text = df.nlargest(25, 'assistant_text_length')

# Get top 25 longest conversations by user text length
top_25_user_text = df.nlargest(25, 'user_text_length')

# Save to CSV files
top_25_interactions.to_csv('top_25_interactions.csv', index=False)
top_25_assistant_text.to_csv('top_25_assistant_text.csv', index=False)
top_25_user_text.to_csv('top_25_user_text.csv', index=False)

# Plot the results
plt.figure(figsize=(14, 7))

# Plot top 25 longest conversations by number of interactions
plt.subplot(131)
plt.barh(top_25_interactions['title'], top_25_interactions['interaction_count'])
plt.xlabel('Number of Interactions')
plt.title('Top 25 Conversations by Interactions')

# Plot top 25 longest conversations by assistant text length
plt.subplot(132)
plt.barh(top_25_assistant_text['title'], top_25_assistant_text['assistant_text_length'])
plt.xlabel('Assistant Text Length')
plt.title('Top 25 Conversations by Assistant Text Length')

# Plot top 25 longest conversations by user text length
plt.subplot(133)
plt.barh(top_25_user_text['title'], top_25_user_text['user_text_length'])
plt.xlabel('User Text Length')
plt.title('Top 25 Conversations by User Text Length')

plt.tight_layout()
plt.savefig('top_25_conversations_analysis.png')
plt.show()
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract timestamps for user messages
timestamps = []
for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message']['author']['role'] == 'user':
            timestamp = node['message']['create_time']
            if timestamp:
                timestamps.append(datetime.fromtimestamp(timestamp))

# Create a DataFrame with hour and day of week
df = pd.DataFrame({
    'Hour': [ts.hour for ts in timestamps],
    'DayOfWeek': [ts.strftime('%A') for ts in timestamps]
})

# Pivot table for heatmap
heatmap_data = df.pivot_table(index='DayOfWeek', columns='Hour', aggfunc='size', fill_value=0)

# Reorder the days of the week
heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='d')
plt.title('User Activity Heatmap')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.savefig('user_activity_heatmap.png')
plt.show()
import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Load the conversations.json file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract user messages and analyze behavior patterns
user_messages = []
for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message']['author']['role'] == 'user':
            message_content = node['message']['content'].get('parts', [])
            for part in message_content:
                if isinstance(part, str):  # Ensure the content part is a string
                    user_messages.append(part)

# Word frequency analysis
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(' '.join(user_messages))
filtered_words = [w for w in word_tokens if not w.lower() in stop_words and w.isalnum()]
word_freq = Counter(filtered_words)

# Create a DataFrame for word frequency
word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency',
                                                                                          ascending=False)

# Plot the top 20 most common words
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=word_freq_df.head(20))
plt.title('Top 20 Most Common Words in User Messages')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.savefig('top_20_user_words.png')
plt.show()

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of User Messages')
plt.savefig('user_wordcloud.png')
plt.show()
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the input and output directories
input_dir = './gptlogs'
output_file = 'gptlogs_vector_db.faiss'

# Initialize lists to hold data
vectors = []
metadata = []

# Load each JSON file and convert the text data to vectors
for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r') as file:
            conversation = json.load(file)
            summary = conversation['summary']
            vector = model.encode(summary)
            vectors.append(vector)
            metadata.append({
                'conversation_id': conversation['conversation_id'],
                'title': conversation['title'],
                'file_name': file_name
            })

# Convert lists to numpy arrays
vectors = np.array(vectors).astype('float32')

# Create a FAISS index and add vectors to it
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Save the FAISS index to a file
faiss.write_index(index, output_file)

# Save the metadata to a JSON file
with open('metadata.json', 'w') as file:
    json.dump(metadata, file, indent=4)

print(f"Vector database and metadata have been saved.")


# Function to perform a similarity search
def search_similar_conversations(query, top_k=5):
    query_vector = model.encode(query).astype('float32')
    distances, indices = index.search(np.array([query_vector]), top_k)

    results = []
    for i in range(top_k):
        result = metadata[indices[0][i]]
        result['distance'] = float(distances[0][i])
        results.append(result)

    return results


# Example usage
query = "Find conversations about AI and machine learning."
results = search_similar_conversations(query)

print(f"Top {len(results)} similar conversations for the query '{query}':")
for result in results:
    print(result)
