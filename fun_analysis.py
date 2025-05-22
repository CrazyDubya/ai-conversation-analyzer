import json
import re

import nltk
from nltk.corpus import stopwords

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
