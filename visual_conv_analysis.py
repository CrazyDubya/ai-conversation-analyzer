import re
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords from nltk
nltk.download('stopwords')

# Load the formatted CSV file
formatted_file_path = 'formatted_conversation_analysis.csv'
data = pd.read_csv(formatted_file_path)

# Filter out invalid rows in Keywords column
data = data[data['Keywords'].apply(lambda x: isinstance(x, str))]


# Function to clean keywords
def clean_keywords(keyword_list):
    stop_words = set(stopwords.words('english'))
    # Remove stop words, numbers, and correct common typos
    cleaned_keywords = []
    for keyword in keyword_list:
        # Remove numbers
        keyword = re.sub(r'\b\d+\b', '', keyword)
        # Remove stop words
        if keyword.lower() not in stop_words and keyword != "":
            cleaned_keywords.append(keyword.lower())
    return cleaned_keywords


# Convert Keywords column to list, handling NaN values and cleaning keywords
data['Keywords'] = data['Keywords'].apply(lambda x: clean_keywords(x.split(', ')) if isinstance(x, str) else [])

# Flatten the list of keywords and calculate frequency
all_keywords = [keyword for sublist in data['Keywords'] for keyword in sublist]
keyword_counts = Counter(all_keywords)

# Define the number of most common keywords to save
num_keywords_to_save = 25  # Set this variable to change the number of keywords saved
common_keywords = keyword_counts.most_common(num_keywords_to_save)

# Save the most common keywords to a file
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
keywords_filename = f'{num_keywords_to_save}_most_common_keywords_{timestamp}.csv'
pd.DataFrame(common_keywords, columns=['Keyword', 'Frequency']).to_csv(keywords_filename, index=False)

# Create a co-occurrence matrix
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X = vectorizer.fit_transform(data['Keywords'])
co_occurrence_matrix = (X.T * X)

# Remove zero values from the co-occurrence matrix
co_occurrence_matrix.setdiag(0)
co_occurrence_df = pd.DataFrame(co_occurrence_matrix.toarray(), index=vectorizer.get_feature_names_out(),
                                columns=vectorizer.get_feature_names_out())
co_occurrence_df = co_occurrence_df.loc[:, (co_occurrence_df != 0).any(axis=0)]
co_occurrence_df = co_occurrence_df[(co_occurrence_df.T != 0).any()]

# Plot the co-occurrence matrix
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_df, cmap='Blues', annot=True, fmt="d", cbar_kws={'label': 'Co-occurrence Count'})
plt.title('Keyword Co-occurrence Matrix')
heatmap_filename = f'keyword_co_occurrence_matrix_{timestamp}.png'
plt.savefig(heatmap_filename)
plt.close()

# Create a network graph of keyword co-occurrences
G = nx.from_pandas_adjacency(co_occurrence_df)
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.3)  # Adjust layout for better spacing
nx.draw_networkx(G, pos=pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, edge_color='grey')
plt.title('Keyword Co-occurrence Network')
network_graph_filename = f'keyword_co_occurrence_network_{timestamp}.png'
plt.savefig(network_graph_filename)
plt.close()

print(f"Most common keywords saved to {keywords_filename}")
print(f"Co-occurrence matrix plot saved to {heatmap_filename}")
print(f"Network graph plot saved to {network_graph_filename}")
