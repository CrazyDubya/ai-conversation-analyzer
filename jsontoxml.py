import json
import os
import xml.etree.ElementTree as ET

# Define the directory containing the JSON files
json_directory = '/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations-2'

# Create the root element of the XML
root = ET.Element("Conversations")

# Walk through the directory and process each JSON file
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        filepath = os.path.join(json_directory, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
            conversation_id = data.get("conversation_id")
            keywords = data.get("keywords", [])
            summary = data.get("summary", "No summary found")

            # Ensure keywords and summary are strings
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            if not isinstance(summary, str):
                summary = str(summary)

            # Create an XML element for this conversation
            conversation_element = ET.SubElement(root, "Conversation")
            ET.SubElement(conversation_element, "ConversationID").text = conversation_id
            ET.SubElement(conversation_element, "Keywords").text = keywords
            ET.SubElement(conversation_element, "Summary").text = summary

# Convert the XML tree to a string and save it to a file
tree = ET.ElementTree(root)
with open("/Users/puppuccino/PycharmProjects/inner_mon/archive_key_sum.xml", "wb") as xml_file:
    tree.write(xml_file)

print("XML file created successfully.")
