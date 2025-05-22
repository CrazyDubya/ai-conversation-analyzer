import csv
import json
import os
import re
from collections import Counter
from typing import List, Dict, Tuple

BASE_DIR = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations'


def load_conversation(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_code_blocks(message: str) -> List[str]:
    code_block_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    return code_block_pattern.findall(message)


def identify_language(code_block: str) -> str:
    first_line = code_block.strip().split('\n')[0].lower()
    languages = {
        'python': ['python', 'py'],
        'bash': ['bash', 'sh'],
        'javascript': ['javascript', 'js'],
        'markdown': ['markdown', 'md'],
        'r': ['r']
    }
    for lang, keywords in languages.items():
        if any(keyword in first_line for keyword in keywords):
            return lang
    return 'other'


def is_likely_r_code(content: str) -> bool:
    # List of common R-specific functions and keywords
    r_indicators = [
        'library(', 'install.packages(', 'data.frame(', 'ggplot(',
        'tidyverse', 'dplyr', 'tidyr', 'readr', 'purrr', 'tibble',
        '<-', '%>%', 'function(', 'ifelse(', 'sapply(', 'lapply('
    ]
    return any(indicator in content for indicator in r_indicators)


def analyze_conversation(conversation: Dict) -> Tuple[Counter, Counter, Counter]:
    prog_lang_mentions = Counter()
    code_block_counts = Counter()
    function_counts = Counter()

    languages = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'go', 'kotlin',
                 'rust', 'scala', 'haskell', 'dart', 'lua', 'perl', 'julia', 'elixir', 'clojure',
                 'erlang', 'f#', 'groovy', 'lisp', 'matlab', 'objective-c', 'powershell', 'shell',
                 'sql', 'typescript', 'vb.net', 'html', 'css', 'xml', 'json', 'yaml', 'toml', 'ini',
                 'markdown', 'latex', 'restructuredtext', 'django', 'flask', 'fastapi', 'express',
                 'react', 'angular', 'vue', 'spring', 'rails', 'laravel', '.net']

    conversation_id = conversation.get('conversation_id', 'unknown')

    for message in conversation.get('messages', []):
        content = message.get('content', '')

        # Check for R specifically
        if is_likely_r_code(content):
            prog_lang_mentions['r'] += 1
        else:
            # Check for other languages
            for lang in languages:
                # Use word boundary regex to avoid partial matches
                if re.search(r'\b' + re.escape(lang) + r'\b', content.lower()):
                    prog_lang_mentions[lang] += 1

        code_blocks = extract_code_blocks(content)
        for code_block in code_blocks:
            language = identify_language(code_block)
            code_block_counts[language] += 1

            if language == 'python':
                functions = extract_functions(code_block)
                for function in functions:
                    function_counts[(function, conversation_id)] += 1

    return prog_lang_mentions, code_block_counts, function_counts


def extract_functions(code_block: str) -> List[str]:
    import ast
    try:
        tree = ast.parse(code_block)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except SyntaxError:
        return []


def save_to_csv(data: Counter, filename: str, headers: List[str]):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data.most_common())


def main():
    total_prog_lang_mentions = Counter()
    total_code_block_counts = Counter()
    total_function_counts = Counter()

    for filename in os.listdir(BASE_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(BASE_DIR, filename)

            try:
                conversation = load_conversation(file_path)
                prog_lang_mentions, code_block_counts, function_counts = analyze_conversation(conversation)

                total_prog_lang_mentions += prog_lang_mentions
                total_code_block_counts += code_block_counts
                total_function_counts += function_counts

                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    save_to_csv(total_prog_lang_mentions, 'prog_lang_mentions.csv', ['Language', 'Mentions'])
    save_to_csv(total_code_block_counts, 'code_block_counts.csv', ['Language', 'Count'])

    with open('function_counts.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function', 'Conversation', 'Count'])
        for (function, conversation), count in total_function_counts.items():
            writer.writerow([function, conversation, count])

    print("Analysis completed successfully. CSV files have been generated.")


if __name__ == "__main__":
    main()
