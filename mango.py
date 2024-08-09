# Install the required libraries
# !pip install openai pymongo
import os
import openai
import pymongo
import uuid
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI API (replace 'your_api_key' with your actual API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to MongoDB (replace 'your_mongo_uri' with your actual MongoDB connection string)
client = pymongo.MongoClient('mongodb+srv://mallesh:root@cluster0.nq4j5hl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['chatbot_db']
collection = db['conversations']

# Function to interact with the chatbot and store responses
def interact_with_chatbot(messages, session_id, summary_buffer):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    
    answer = response.choices[0].message['content'].strip()
    
    # Add the response to the summary buffer
    summary_buffer.append({'prompt': messages[-1]['content'], 'response': answer, 'timestamp': datetime.now()})
    
    # Add the assistant's response to the messages
    messages.append({"role": "assistant", "content": answer})
    
    return answer, summary_buffer, messages

# Function to save memory to MongoDB
def save_memory_to_db(session_id, summary_buffer):
    session_data = {
        'session_id': session_id,
        'summary_buffer': summary_buffer,
        'last_updated': datetime.now()
    }
    collection.insert_one(session_data)

# Function to load a session from MongoDB
def load_session_from_db(session_id):
    session_data = collection.find_one({'session_id': session_id})
    if session_data:
        return session_data['summary_buffer'], session_data['summary_buffer']
    else:
        return [], []

# Function to generate a new session ID
def generate_new_session_id():
    return str(uuid.uuid4())

# Main function to start or continue a conversation
def start_conversation(session_id=None):
    if session_id:
        print(f"Loading session {session_id}...")
        summary_buffer, messages = load_session_from_db(session_id)
        if not messages:
            # Initialize the conversation messages if empty
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
    else:
        # Generate a new session ID and initialize conversation
        session_id = generate_new_session_id()
        summary_buffer = []
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print(f"Starting new session with ID {session_id}...")

    print("Chatbot is ready! Type 'exit' to end the conversation.")
    
    while True:
        user_prompt = input("You: ")
        
        if user_prompt.lower() == 'exit':
            break
        
        messages.append({"role": "user", "content": user_prompt})
        response, summary_buffer, messages = interact_with_chatbot(messages, session_id, summary_buffer)
        print(f"Chatbot: {response}")

    # Save the memory to MongoDB
    save_memory_to_db(session_id, summary_buffer)
    print(f"Session ID: {session_id}")

# Example usage
# Start a new session or provide an existing session ID
session_id = input("Enter session ID to load or press Enter to start a new session: ").strip()
start_conversation(session_id if session_id else None)