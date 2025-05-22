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
