import csv
import json
import os

import anthropic

client = anthropic.Anthropic(
    api_key=os.environ['ANTHROPIC_API_KEY'],
)

# Define the directory containing JSON files
directory = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations-2'


# Function to process a JSON file with Claude-haiku
def process_with_claude_haiku(json_content):
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0.2,
            system="""analyze json and respond using this guide. Only Repond in XML, no extra - 
<ConversationAnalysis>
<Keywords>
<Keyword>example</Keyword>
<Keyword>test</Keyword>
</Keywords>
<Summary>This is a summary of the conversation. 5 sentence Max</Summary>
<SentimentTrends>
<Trend>
<ConversationId>1234</ConversationId>
<Timestamp>2024-05-24T12:00:00</Timestamp>
<Sentiment>0.5</Sentiment>
</Trend>
<!-- Repeat <Trend> for each conversation's sentiment trend -->
</SentimentTrends>
<WordFrequencies>
<Word>
<Text>example</Text>
<Frequency>5</Frequency>
</Word>
<!-- Repeat <Word> for each word -->
</WordFrequencies>
<Anomalies>
<Anomaly>
<ConversationId>1234</ConversationId>
<MessageCount>10</MessageCount>
<IsAnomaly>true</IsAnomaly>
</Anomaly>
<!-- Repeat <Anomaly> for each detected anomaly -->
</Anomalies>
<Statistics>
<MessageLengths>
<Longest>1234</Longest>
<Shortest>10</Shortest>
<Mean>50</Mean>
<Median>45</Median>
<StdDev>20</StdDev>
</MessageLengths>
</Statistics>
</ConversationAnalysis>
""",
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(json_content)
                }
            ]
        )
        return message.content
    except Exception as e:
        print(f"Error processing conversation: {e}")
        return None


# Process all JSON files in the directory
xml_results = []
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename), 'r') as file:
            json_content = json.load(file)
            xml_content = process_with_claude_haiku(json_content)
            if xml_content:
                xml_results.append({
                    'conversation_id': json_content.get('conversation_id', 'unknown'),
                    'xml': xml_content
                })

# Save the results to a CSV file
csv_file = 'conversation_analysis.csv'
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['conversation_id', 'xml']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in xml_results:
        writer.writerow(result)

print(f"Results saved to {csv_file}")
