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
