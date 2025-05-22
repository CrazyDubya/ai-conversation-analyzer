import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/top_10_assistant_messages_20240520214620.csv'
df = pd.read_csv(file_path)

# Sort the dataframe by character count in descending order
df_sorted = df.sort_values(by='Character Count', ascending=False).head(500)

# Visualize the top 500 assistant messages by character count in a bar chart
plt.figure(figsize=(12, 18))
plt.barh(df_sorted['Message ID'], df_sorted['Character Count'], color='skyblue')

# Set y-ticks to show every 50th message ID
plt.yticks(df_sorted['Message ID'][::50])

plt.xlabel('Character Count')
plt.ylabel('Message ID')
plt.title('Top 500 Assistant Messages by Character Count')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot as an image
output_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/top_500_assistant_messages.png'
plt.savefig(output_path)
plt.show()

output_path
