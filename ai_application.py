import openai
import gradio as gr
import requests
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

# Define the function to fetch weather
def fetch_weather(location):
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    base_url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': api_key,
        'units': 'metric'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Define the custom handle methods
def handle_weather_data(weather_data):
    if not weather_data:
        return "No data to analyze"
    temp = weather_data.get('main', {}).get('temp', 'N/A')
    weather = weather_data.get('weather', [{}])[0].get('description', 'N/A')
    return {
        'temperature': temp,
        'description': weather
    }

def present_weather_report(analyzed_data):
    if not analyzed_data:
        return "No data to present"
    temp = analyzed_data.get('temperature', 'N/A')
    weather = analyzed_data.get('description', 'N/A')
    return f"The current temperature is {temp}°C with {weather}."

# Define the Gradio-based task executor
prompt = PromptTemplate.from_template(
    "You are an AI assistant. The user wants to perform the following task: {task}. "
    "Gather the required information and execute the task."
)

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

conversation_chain = LLMChain(
    prompt=prompt,
    llm=llm,
)

def handle_input(user_input):
    task = user_input.lower()
    
    if "weather" in task:
        required_fields = ["location"]
        task_parameters = {}
        for field in required_fields:
            task_parameters[field] = user_input
        
        location = task_parameters.get("location", "Unknown Location")
        weather_data = fetch_weather(location)
        if weather_data:
            analyzed_data = handle_weather_data(weather_data)
            result = present_weather_report(analyzed_data)
        else:
            result = "Error: Unable to fetch weather data. Please check the location and API key."
    
    else:
        result = "Sorry, I can only assist with weather-related tasks at the moment."

    return result

# Setup Gradio Interface
iface = gr.Interface(
    fn=handle_input,
    inputs="text",
    outputs="text",
    title="Smart AI Task Executor",
    description="An AI system that helps you perform tasks by understanding your needs and gathering the necessary details."
)

# Launch the interface
iface.launch()
