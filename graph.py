import json

import graphviz

# Load data
file_path = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/gptlogs/2_Player_C.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Create a flowchart for a single conversation
conversation = data[0]
dot = graphviz.Digraph(comment='Conversation Flow')

for node_id, node in conversation['mapping'].items():
    message = node.get('message')
    if message:
        author = message['author']['role']
        content = ' '.join(message['content'].get('parts', []))
        dot.node(node_id, f"{author}: {content}")

    parent_id = node.get('parent')
    if parent_id:
        dot.edge(parent_id, node_id)

# Save and render the flowchart
dot.render('conversation_flowchart', format='png')

# Display the flowchart
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('conversation_flowchart.png')
plt.imshow(image)
plt.axis('off')
plt.show()
