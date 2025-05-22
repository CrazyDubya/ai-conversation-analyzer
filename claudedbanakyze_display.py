import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files into DataFrames
prog_lang_mentions_df = pd.read_csv('prog_lang_mentions.csv')
code_block_counts_df = pd.read_csv('code_block_counts.csv')
function_counts_df = pd.read_csv('function_counts.csv')

# Display the top 10 programming languages by mentions
print("Top 10 Programming Languages by Mentions:")
print(prog_lang_mentions_df.nlargest(10, 'Mentions'))

# Plot the top 25 programming languages by mentions
top_langs = prog_lang_mentions_df.nlargest(25, 'Mentions')
plt.figure(figsize=(10, 6))
plt.bar(top_langs['Language'], top_langs['Mentions'])
plt.xlabel('Language')
plt.ylabel('Mentions')
plt.title('Top 10 Programming Languages by Mentions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display the code block counts by language
print("\nCode Block Counts by Language:")
print(code_block_counts_df)

# Plot the code block counts by language
plt.figure(figsize=(10, 6))
plt.pie(code_block_counts_df['Count'], labels=code_block_counts_df['Language'], autopct='%1.1f%%')
plt.title('Code Block Counts by Language')
plt.tight_layout()
plt.show()

# Display the top 10 most frequently used functions
print("\nTop 10 Most Frequently Used Functions:")
top_functions = function_counts_df.groupby('Function').sum().nlargest(10, 'Count').reset_index()
print(top_functions)

# Plot the top 10 most frequently used functions
plt.figure(figsize=(10, 6))
plt.bar(top_functions['Function'], top_functions['Count'])
plt.xlabel('Function')
plt.ylabel('Count')
plt.title('Top 10 Most Frequently Used Functions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display the number of unique functions used in each conversation
print("\nNumber of Unique Functions Used in Each Conversation:")
unique_functions_per_conversation = function_counts_df.groupby('Conversation').agg(
    {'Function': 'nunique'}).reset_index()
print(unique_functions_per_conversation)
