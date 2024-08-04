import streamlit as st
from crewai import Agent, Task, Crew, Process
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Define the function to fetch weather
def fetch_weather(location):
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
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
    # Perform some analysis on the weather data
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

# Researcher agent to fetch weather data
researcher_agent = Agent(
    role='Weather Data Fetcher',
    goal='Fetch the current weather data for a given location',
    backstory='An expert in accessing and retrieving weather data from APIs.'
)

# Analyst agent to analyze weather data
analyst_agent = Agent(
    role='Weather Data Analyst',
    goal='Analyze the fetched weather data',
    backstory='A specialist in analyzing weather data and extracting key information.'
)

# Writer agent to present weather data
writer_agent = Agent(
    role='Weather Report Writer',
    goal='Present the weather data in a user-friendly format',
    backstory='An expert in creating concise and informative weather reports.'
)

# Task for the researcher agent to fetch weather data
fetch_weather_task = Task(
    description='Fetch the current weather data for a given location',
    expected_output='Raw weather data from the API',
    agent=researcher_agent,
    tools=[fetch_weather]
)

# Task for the analyst agent to analyze weather data
analyze_weather_task = Task(
    description='Analyze the fetched weather data',
    expected_output='Analyzed weather data with key information',
    agent=analyst_agent
)

# Task for the writer agent to present the weather data
write_weather_report_task = Task(
    description='Present the weather data in a user-friendly format',
    expected_output='A user-friendly weather report',
    agent=writer_agent,
    context=[fetch_weather_task, analyze_weather_task]
)

# Define the crew with the agents and tasks
weather_crew = Crew(
    agents=[researcher_agent, analyst_agent, writer_agent],
    tasks=[fetch_weather_task, analyze_weather_task, write_weather_report_task],
    process=Process.sequential
)

def get_weather_report(location):
    # Fetch weather data
    weather_data = fetch_weather(location)
    if weather_data:
        # Analyze weather data
        analyzed_data = handle_weather_data(weather_data)
        # Write weather report
        weather_report = present_weather_report(analyzed_data)
        return weather_report
    else:
        return "Error: Unable to fetch weather data. Please check the location and API key."


# Streamlit interface
st.title("Weather Report")

location = st.text_input("Enter the location:")

if st.button("Get Weather Report"):
    if location:
        weather_report = get_weather_report(location)
        st.write(weather_report)
    else:
        st.write("Please enter a location.")