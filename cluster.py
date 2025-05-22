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
