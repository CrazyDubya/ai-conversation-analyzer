import json
import os
import time
import warnings

import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set the path to the directory containing the JSON files
directory = "/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations-2"

# Initialize empty lists to store conversations, soft errors, and hard errors
conversations = []
soft_errors = []
hard_errors = []

# Walk through the directory and load JSON files sequentially
for filename in tqdm(os.listdir(directory), desc="Loading JSON files"):
    if filename.endswith(".json"):
        try:
            with open(os.path.join(directory, filename), "r") as file:
                data = json.load(file)
                conversation = data.get("document_content", {})

                # Check for missing fields and add placeholders
                conversation["conversation_id"] = conversation.get("conversation_id", "")
                conversation["title"] = conversation.get("title", "")
                conversation["messages"] = conversation.get("messages", [])
                conversation["keywords"] = conversation.get("keywords", [])
                conversation["summary"] = conversation.get("summary", "")

                conversations.append(conversation)

                # Log soft errors for missing fields
                if not conversation["conversation_id"]:
                    soft_errors.append(f"Missing 'conversation_id' in {filename}")
                if not conversation["title"]:
                    soft_errors.append(f"Missing 'title' in {filename}")
                if not conversation["messages"]:
                    soft_errors.append(f"Missing 'messages' in {filename}")
                if not conversation["keywords"]:
                    soft_errors.append(f"Missing 'keywords' in {filename}")
                if not conversation["summary"]:
                    soft_errors.append(f"Missing 'summary' in {filename}")

        except Exception as e:
            hard_errors.append(f"Error loading {filename}: {str(e)}")

# Extract features from conversations
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform([json.dumps(conv) for conv in conversations])

# Calculate similarity scores
similarity_matrix = cosine_similarity(features)

# Normalize the similarity matrix
scaler = StandardScaler()
normalized_similarity_matrix = scaler.fit_transform(similarity_matrix)

# Suppress warnings
warnings.filterwarnings("ignore")

# Perform dimensionality reduction
tsne = TSNE(n_components=3, perplexity=30, early_exaggeration=12, random_state=42)
start_time = time.time()
reduced_features = tsne.fit_transform(normalized_similarity_matrix)
end_time = time.time()
reduction_time = end_time - start_time

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_features[:, 0],
    y=reduced_features[:, 1],
    z=reduced_features[:, 2],
    mode='markers',
    marker=dict(size=5),
    text=[conv["title"] for conv in conversations],
    hoverinfo='text'
)])

# Set plot layout
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Save the plot as an HTML file
timestamp = time.strftime("%Y%m%d_%H%M%S")
plot_filename = f"conversation_plot_{timestamp}.html"
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
print(f"Plot saved as: {plot_filename}")
print(f"Reduced features saved as: {reduced_features_filename}")
if soft_errors:
    print(f"Soft errors saved as: {soft_error_filename}")
if hard_errors:
    print(f"Hard errors saved as: {hard_error_filename}")
