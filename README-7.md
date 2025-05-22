
# chatgptarchive.py

## Overview

`chatgptarchive.py` is a Python script designed to parse conversations from a JSON file (e.g., a ChatGPT conversation archive) and save them to individual JSON files. This tool is helpful for managing and analyzing conversation data more efficiently.

## Prerequisites

Before you run `chatgptarchive.py`, you need to have Python installed on your system. This script is compatible with Python 3.6 or later.

## Getting Your ChatGPT Conversation Archive

### Requesting the Archive

1. **OpenAI Users:** Log in to your ChatGPT account, navigate to settings/data controls, and request your data. The data will delivered to you via email as a downloadable zip file.

### Unzipping the Archive

- **On macOS:**
  1. Open Terminal.
  2. Navigate to the directory containing the downloaded zip file.
  3. Use the command `unzip [filename].zip` - replace `[filename]` with the name of your downloaded file.

- **On Windows:**
  1. Navigate to the folder containing the zip file in File Explorer.
  2. Right-click on the zip file.
  3. Select "Extract All..." and follow the instructions to extract the files.

- **On Linux:**
  1. Open Terminal.
  2. Navigate to the directory containing the downloaded zip file.
  3. Use the command `unzip [filename].zip`.

## Installation

No additional libraries are required for this script. It uses only the built-in `json`, `os`, and `re` modules available in Python's standard library.

## Usage

To run `chatgptarchive.py`:

1. Open your command line interface (Terminal on macOS and Linux, Command Prompt or PowerShell on Windows).
2. Navigate to the directory containing `chatgptarchive.py`.
3. Run the script using the command:
   ```
   python chatgptarchive.py
   ```
4. When prompted, enter the full path to your unzipped `conversations.json` file.
5. The script will process the data and save each conversation as a separate JSON file in a directory named `gptlogs-DDMMYY` (with the current date) within the same directory as your `conversations.json`.

## Output

After running the script, each conversation from the `conversations.json` will be saved as an individual JSON file in the `gptlogs-DDMMYY` directory. These files are named based on the title of the conversation or assigned a default name if the title is not available.

## Additional Scripts

### load_and_format_conversation.py

This script helps you load and format a conversation from the saved JSON files.

```python
import json
import os

def load_conversation(file_path):
    with open(file_path, 'r') as file:
        conversation_data = json.load(file)
    return conversation_data

def format_conversation(conversation):
    formatted_conversation = f"Conversation ID: {conversation['conversation_id']}
"
    formatted_conversation += f"Title: {conversation['title']}

"

    for message in conversation['messages']:
        author = message['author']
        content_parts = message['content']

        # Handle content parts which might be dicts or strings
        if content_parts and isinstance(content_parts[0], dict):
            content = "
".join([part.get('text', '') for part in content_parts])
        else:
            content = "
".join(content_parts)

        create_time = message['create_time']
        update_time = message['update_time']

        formatted_conversation += f"Author: {author}
"
        formatted_conversation += f"Content: {content}
"
        formatted_conversation += f"Create Time: {create_time}
"
        formatted_conversation += f"Update Time: {update_time}
"
        formatted_conversation += "-" * 40 + "
"

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
```

### analyze_conversations.py

This script analyzes the conversations to find common words and perform sentiment analysis.

```python
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
```

## Troubleshooting

If you encounter any issues regarding file paths or permissions, ensure that the path you input corresponds exactly to where your `conversations.json` file is located and that you have appropriate read/write permissions for the directory.

---

This README provides a comprehensive guide to obtaining, preparing, and processing your ChatGPT conversation data using `chatgptarchive.py`.
