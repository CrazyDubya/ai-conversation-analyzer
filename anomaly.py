import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# Load data
with open('conversations.json', 'r') as file:
    data = json.load(file)

# Extract features (e.g., length of messages)
features = [len(conversation['mapping']) for conversation in data]

# Fit Isolation Forest
model = IsolationForest(contamination=0.1)
preds = model.fit_predict(np.array(features).reshape(-1, 1))

# Plot results
plt.scatter(range(len(features)), features, c=preds)
plt.title('Anomaly Detection in Conversations')
plt.xlabel('Conversation Index')
plt.ylabel('Number of Messages')
plt.show()

# Save the anomaly detection plot
plt.savefig('anomaly_detection.png')
