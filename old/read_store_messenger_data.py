import os
import json
import pprint
import pandas as pd


current_directory_abspath = os.getcwd()
DATASET_DIRNAME = "dataset"
dataset_abspath = os.path.join(current_directory_abspath, DATASET_DIRNAME)

messenger_abspaths = [
    os.path.join(dataset_abspath, dirname, "messages") 
    for dirname in os.listdir(dataset_abspath) 
    if dirname.startswith("your_facebook_activity")
]

messenger_data_list = []
for messenger_abspath in messenger_abspaths:
    
    message_type_abspaths = os.listdir(messenger_abspath)
    for message_type_abspath in message_type_abspaths:
        
        message_abspaths = os.path.join(messenger_abspath, message_type_abspath)
        for message_abspath in os.listdir(message_abspaths):
            
            json_abspath = os.path.join(messenger_abspath, message_type_abspath, message_abspath, "message_1.json")
            with open(json_abspath, "r") as file:
                text = file.read()
                file_json = json.loads(text)
                messenger_data_list.append(file_json)


# temporary
question_keywords = [
    "?", "what", "when", "where", "who", "why", "how", "ask", "ano", "kailan", "saan", "sino", "bakit", "paano", "tanong", "pwede"
]


# --------------
# messages: List[Dict]
#   sender_name: str
#   content: str or None
questions = []
for messenger_data in messenger_data_list:
    
    messages = messenger_data["messages"]
    for message in messages:
        
        if "content" in message:
            
            content = message["content"]
            for token in content.split():
                
                if token in question_keywords:
                    questions.append(content)
                    break


df = pd.DataFrame({
    "Question": questions
})

filename_w_extension = "questions.xlsx"
df.to_excel(filename_w_extension)
