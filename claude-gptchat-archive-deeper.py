import json
import os
import time
from typing import Dict, List, Tuple

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anthropic import Anthropic
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

BASE_DIR = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations'  # Update this path

# Define keywords for categorization
PROGRAMMING_KEYWORDS = {'python', 'code', 'programming', 'function', 'algorithm', 'script', 'variable', 'class',
                        'import', 'def'}
IMAGE_GENERATION_KEYWORDS = {'image', 'dall-e', 'midjourney', 'stable diffusion', 'gan', 'style transfer',
                             'neural style'}


def load_conversations(directory: str) -> Dict[str, List[str]]:
    """Load conversations from JSON files in the specified directory."""
    conversations = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    conversation_id = data.get('conversation_id', filename)
                    messages = [msg['content'] for msg in data.get('messages', []) if 'content' in msg]
                    conversations[conversation_id] = messages
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return conversations


def categorize_conversation(conversation: List[str]) -> str:
    """Categorize a conversation based on its content."""
    full_text = ' '.join(conversation).lower()
    if any(keyword in full_text for keyword in PROGRAMMING_KEYWORDS):
        return 'programming'
    elif any(keyword in full_text for keyword in IMAGE_GENERATION_KEYWORDS):
        return 'image_generation'
    else:
        return 'other'


def preprocess_text(text: str) -> str:
    """Preprocess text by removing special characters and extra whitespace."""
    import re
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and remove extra whitespace
    return ' '.join(text.lower().split())


def extract_topics(text: str, n: int = 25) -> List[str]:
    """Extract top N topics from text using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        top_indices = tfidf_matrix.tocsr().indices[tfidf_matrix.tocsr().data.argsort()[-n:][::-1]]
        return [feature_names[i] for i in top_indices]
    except ValueError:
        return []  # Return empty list if no topics could be extracted


def build_topic_graph(conversations: Dict[str, List[str]]) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
    """Build separate graphs for programming, image generation, and other topics."""
    G_prog = nx.Graph()
    G_img = nx.Graph()
    G_other = nx.Graph()

    for conversation_id, messages in conversations.items():
        category = categorize_conversation(messages)
        full_text = preprocess_text(' '.join(messages))
        topics = extract_topics(full_text)

        G = G_prog if category == 'programming' else G_img if category == 'image_generation' else G_other

        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                topic1, topic2 = topics[i], topics[j]
                if G.has_edge(topic1, topic2):
                    G[topic1][topic2]['weight'] += 1
                else:
                    G.add_edge(topic1, topic2, weight=1)

    return G_prog, G_img, G_other


def analyze_topic_communities(G: nx.Graph) -> Dict[str, int]:
    """Detect communities in the topic graph."""
    return community.best_partition(G, resolution=1.2)  # Adjusted resolution for more communities


def create_enhanced_heatmap(G: nx.Graph, communities: Dict[str, int], output_file: str, top_n: int = 50,
                            threshold: float = 0.1):
    """Create an improved enhanced heatmap of topic co-occurrences."""
    """Create an improved enhanced heatmap of topic co-occurrences."""
    adj_matrix = nx.to_pandas_adjacency(G)

    # Select top N topics by degree
    top_topics = sorted([(n, d) for n, d in G.degree()], key=lambda x: x[1], reverse=True)[:top_n]
    top_topic_names = [t[0] for t in top_topics]

    # Create a submatrix with only the top topics
    sub_matrix = adj_matrix.loc[top_topic_names, top_topic_names]

    # Apply logarithmic scaling
    sub_matrix = np.log1p(sub_matrix)

    # Apply threshold
    sub_matrix[sub_matrix < threshold] = 0

    # Sort topics by their communities
    community_sorted_topics = sorted(top_topic_names, key=lambda x: communities[x])
    sub_matrix = sub_matrix.loc[community_sorted_topics, community_sorted_topics]

    plt.figure(figsize=(24, 20))  # Increased figure size

    # Create the heatmap
    g = sns.heatmap(sub_matrix, cmap="YlGnBu", annot=False, square=True, cbar_kws={'label': 'Log(Co-occurrences + 1)'})

    # Rotate x-axis labels for better readability
    plt.setp(g.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.get_yticklabels(), rotation=0)

    # Increase font size
    plt.tick_params(axis='both', which='major', labelsize=8)

    # Add title
    plt.title(f"Enhanced Topic Co-occurrence Heatmap (Top {top_n} Topics)", fontsize=16)

    # Add note about scaling and threshold
    plt.figtext(0.5, 0.01, f"Note: Values are log-scaled and thresholded at {threshold}.", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_topic_importance(G: nx.Graph) -> pd.DataFrame:
    """Analyze topic importance based on graph metrics."""
    importance = pd.DataFrame({
        'Degree': dict(G.degree()),
        'Betweenness': nx.betweenness_centrality(G),
        'Closeness': nx.closeness_centrality(G)
    })
    importance['Importance Score'] = importance.mean(axis=1)
    return importance.sort_values('Importance Score', ascending=False)


def create_interactive_graph(G: nx.Graph, communities: Dict[str, int], output_file: str):
    """Create an interactive graph visualization."""
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    for node in net.nodes:
        node["group"] = communities[node["id"]]
        node["size"] = G.degree(node["id"]) * 3
        node["title"] = f"Topic: {node['id']}<br>Community: {node['group']}<br>Connections: {G.degree(node['id'])}"
    net.show_buttons(filter_=['physics'])
    net.save_graph(output_file)


def analyze_topic_usage(conversations: Dict[str, List[str]], category: str) -> pd.DataFrame:
    """Analyze topic usage across conversations for a specific category."""
    category_conversations = [' '.join(msgs) for msgs in conversations.values()
                              if categorize_conversation(msgs) == category]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(category_conversations)
    feature_names = vectorizer.get_feature_names_out()

    topic_usage = pd.DataFrame(tfidf_matrix.sum(axis=0).T, index=feature_names, columns=['TF-IDF Score'])
    return topic_usage.sort_values('TF-IDF Score', ascending=False)


def ai_analyze_topics(G: nx.Graph, communities: Dict[str, int], usage_df: pd.DataFrame, importance_df: pd.DataFrame,
                      category: str) -> str:
    """Use AI to analyze topic relationships and provide insights for a specific category."""
    context = f"""
    Analysis for {category.replace('_', ' ').title()} Conversations:

    Graph Summary:
    - Number of nodes (topics): {G.number_of_nodes()}
    - Number of edges (connections): {G.number_of_edges()}
    - Number of communities: {len(set(communities.values()))}

    Top 10 Most Used Topics:
    {usage_df.head(25).to_string()}

    Top 10 Most Important Topics:
    {importance_df.head(25).to_string()}

    Please analyze this data and provide insights on:
    1. The overall structure of the {category.replace('_', ' ')} conversations based on topic relationships.
    2. Potential areas for further exploration or discussion within this category.
    3. Any interesting patterns or clusters of topics specific to {category.replace('_', ' ')}.
    4. Recommendations for guiding future conversations or research in this area.
    5. How the topic relationships might reflect broader trends or interests in {category.replace('_', ' ')}.
    """

    message = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": context}
        ]
    )

    return message.content[0].text


def main():
    start_time = time.time()
    conversations = load_conversations(BASE_DIR)
    print(f"Loaded data from {len(conversations)} conversations")

    G_prog, G_img, G_other = build_topic_graph(conversations)
    print(f"Built graphs:")
    print(f"Programming: {G_prog.number_of_nodes()} nodes and {G_prog.number_of_edges()} edges")
    print(f"Image Generation: {G_img.number_of_nodes()} nodes and {G_img.number_of_edges()} edges")
    print(f"Other: {G_other.number_of_nodes()} nodes and {G_other.number_of_edges()} edges")

    for category, G in [('programming', G_prog), ('image_generation', G_img), ('other', G_other)]:
        if G.number_of_nodes() == 0:
            print(f"No topics could be extracted for {category} conversations.")
            continue

        communities = analyze_topic_communities(G)
        print(f"Detected {len(set(communities.values()))} communities for {category}")

        create_enhanced_heatmap(G, communities, f'{category}_heatmap1.png')
        print(f"Created enhanced heatmap: {category}_heatmap.png")

        create_interactive_graph(G, communities, f'{category}_graph1.html')
        print(f"Created interactive graph: {category}_graph.html")

        usage_df = analyze_topic_usage(conversations, category)
        print(f"\nTop 25 Most Used Topics for {category}:")
        print(usage_df.head(25).to_string())

        importance_df = analyze_topic_importance(G)
        print(f"\nTop 25 Most Important Topics for {category}:")
        print(importance_df.head(25).to_string())

        print(f"\nPerforming AI analysis for {category}...")
        ai_insights = ai_analyze_topics(G, communities, usage_df, importance_df, category)
        print(f"\nAI Insights for {category}:")
        print(ai_insights)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
