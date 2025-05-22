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
