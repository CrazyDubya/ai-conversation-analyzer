import hashlib
import json
import os
import string
import textwrap
from collections import Counter
from typing import Dict, List

import nltk
import numpy as np
from anthropic import Anthropic
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

BASE_DIR = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations'  # Update this path
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_results')  # Directory to store analysis results

# Create analysis directory if it doesn't exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_conversations(directory: str) -> Dict[str, str]:
    """Load conversations from JSON files in the specified directory."""
    conversations = {}
    for i, filename in enumerate(os.listdir(directory), 1):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            conversations[str(i)] = file_path
    return conversations


def load_conversation_content(file_path: str) -> Dict:
    """Load the content of a single conversation from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def preprocess_text(text: str) -> List[str]:
    """Preprocess text by tokenizing, lowercasing, and removing punctuation and stopwords."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words and token not in string.punctuation]


def analyze_conversation_locally(conversation: Dict) -> Dict:
    """Perform local NLP analysis on the conversation."""
    messages = conversation.get('messages', [])
    full_content = " ".join([msg.get('content', '') for msg in messages])

    words = preprocess_text(full_content)
    sentences = sent_tokenize(full_content)

    word_freq = FreqDist(words)
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(sentence) for sentence in sentences]
    overall_sentiment = sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores)

    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([full_content])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    top_topics = [feature_names[i] for i in tfidf_scores.argsort()[::-1][:5]]

    lexical_diversity = len(set(words)) / len(words)
    avg_sentence_length = np.mean([len(word_tokenize(sentence)) for sentence in sentences])

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "top_words": word_freq.most_common(10),
        "pos_distribution": dict(pos_counts),
        "overall_sentiment": overall_sentiment,
        "sentiment_progression": [score['compound'] for score in sentiment_scores],
        "top_topics": top_topics,
        "lexical_diversity": lexical_diversity,
        "avg_sentence_length": avg_sentence_length
    }


def get_file_hash(file_path: str) -> str:
    """Generate a hash for the file content."""
    with open(file_path, 'rb') as file:
        return hashlib.md5(file.read()).hexdigest()


def save_analysis_results(file_path: str, local_analysis: Dict, claude_conversation: List[Dict]):
    """Save both local and Claude's analysis results."""
    file_hash = get_file_hash(file_path)
    analysis_file = os.path.join(ANALYSIS_DIR, f"{file_hash}_analysis.json")

    results = {
        "original_file": os.path.basename(file_path),
        "local_analysis": local_analysis,
        "claude_conversation": claude_conversation
    }

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def load_analysis_results(file_path: str) -> Dict:
    """Load existing analysis results if available."""
    file_hash = get_file_hash(file_path)
    analysis_file = os.path.join(ANALYSIS_DIR, f"{file_hash}_analysis.json")

    if os.path.exists(analysis_file):
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def analyze_conversation(file_path: str, conversation: Dict) -> None:
    """Perform deep analysis of the conversation using local NLP and Claude."""
    existing_analysis = load_analysis_results(file_path)
    if existing_analysis:
        print("Loading existing analysis results...")
        local_analysis = existing_analysis['local_analysis']
        claude_conversation = existing_analysis['claude_conversation']
    else:
        print("Performing new analysis...")
        local_analysis = analyze_conversation_locally(conversation)
        claude_conversation = []

    if not claude_conversation:
        analysis_summary = f"""
        Word Count: {local_analysis['word_count']}
        Sentence Count: {local_analysis['sentence_count']}
        Top Words: {', '.join([f"{word}({count})" for word, count in local_analysis['top_words']])}
        Parts of Speech Distribution: {local_analysis['pos_distribution']}
        Overall Sentiment: {local_analysis['overall_sentiment']:.2f}
        Top Topics: {', '.join(local_analysis['top_topics'])}
        Lexical Diversity: {local_analysis['lexical_diversity']:.2f}
        Average Sentence Length: {local_analysis['avg_sentence_length']:.2f}
        """

        initial_prompt = f"""
        Based on the following local NLP analysis of a conversation, please provide deeper insights and interpretations:

        {analysis_summary}

        Please consider:
        1. What does the word frequency and top topics suggest about the main themes of the conversation?
        2. How does the sentiment progression throughout the conversation reflect the dynamics of the discussion?
        3. What can we infer from the parts of speech distribution and lexical diversity about the complexity and style of language used?
        4. Are there any notable patterns or anomalies in the data that warrant further investigation?
        5. Based on this analysis, what hypotheses can we form about the nature and content of this conversation?

        Provide a detailed interpretation of these results, highlighting key insights and potential areas for further analysis.
        """
        claude_conversation.append({"role": "user", "content": initial_prompt})

    while True:
        if claude_conversation:
            message = anthropic.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                messages=claude_conversation
            )
            response = message.content[0].text
            claude_conversation.append({"role": "assistant", "content": response})
            print("\nClaude's Analysis:")
            print(textwrap.fill(response, width=100))

        user_input = input("\nEnter your next question (or '/bye' to end the conversation): ")
        if user_input.lower() == '/bye':
            break

        claude_conversation.append({"role": "user", "content": user_input})
        save_analysis_results(file_path, local_analysis, claude_conversation)


def list_saved_analyses():
    """List all saved analyses."""
    analyses = []
    for filename in os.listdir(ANALYSIS_DIR):
        if filename.endswith('_analysis.json'):
            with open(os.path.join(ANALYSIS_DIR, filename), 'r') as f:
                data = json.load(f)
                analyses.append(data['original_file'])
    return analyses


def main():
    while True:
        print("\n1. Analyze a new conversation")
        print("2. Load a previous analysis")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            conversations = load_conversations(BASE_DIR)
            print("\nAvailable conversations:")
            for num, file_path in conversations.items():
                print(f"{num}: {os.path.basename(file_path)}")

            conv_choice = input("\nEnter the number of the conversation you want to analyze: ")
            if conv_choice in conversations:
                file_path = conversations[conv_choice]
                conversation = load_conversation_content(file_path)
                analyze_conversation(file_path, conversation)
            else:
                print("Invalid choice. Please try again.")

        elif choice == '2':
            saved_analyses = list_saved_analyses()
            if not saved_analyses:
                print("No saved analyses found.")
                continue

            print("\nSaved analyses:")
            for i, filename in enumerate(saved_analyses, 1):
                print(f"{i}: {filename}")

            analysis_choice = input("\nEnter the number of the analysis you want to load: ")
            try:
                file_to_load = saved_analyses[int(analysis_choice) - 1]
                file_path = os.path.join(BASE_DIR, file_to_load)
                conversation = load_conversation_content(file_path)
                analyze_conversation(file_path, conversation)
            except (ValueError, IndexError):
                print("Invalid choice. Please try again.")

        elif choice == '3':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
