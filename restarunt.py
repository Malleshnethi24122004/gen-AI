from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import langgraph as lg
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Define the OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("OpenAI API key is missing. Please set it in the .env file.")
    st.stop()

# Initialize the OpenAI LLM
llm = OpenAI(api_key=api_key)

# Define a prompt template
prompt_template = """
You are a restaurant recommendation assistant. Based on the given cuisine type, suggest a restaurant name and a sample menu.

Cuisine: {cuisine}

Response:
Restaurant Name: {{restaurant_name}}
Menu:
- Dish 1: {{dish1}}
- Dish 2: {{dish2}}
- Dish 3: {{dish3}}
"""

# Create a LangChain prompt template
prompt = PromptTemplate(template=prompt_template)

# Create an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit app layout
st.title("Restaurant Recommendation")

# Cuisine selection
cuisine = st.selectbox("Choose a cuisine type:", ["Italian", "Chinese", "Mexican", "Indian", "French", "Japanese"])

# Button to generate recommendation
if st.button("Suggest Restaurant and Menu"):
    try:
        # Generate recommendation using LLMChain
        response = llm_chain.run({"cuisine": cuisine})

        # Debug response output
        st.write("Raw Response:", response)

        # Clean and split the response
        cleaned_response = response.strip().split("\n\n\n")
        restaurant_name = None
        menu_items = []

        # Extract information
        for line in cleaned_response:
            line = line.strip()
            if line.startswith("Restaurant Name:"):
                restaurant_name = line.split(":")[1].strip()
            elif line.startswith("Dish"):
                dish = line.split(":")[1].strip()
                if dish and dish not in menu_items:
                    menu_items.append(dish)

        # Validate the parsed data
        if not restaurant_name or len(menu_items) < 3:
            raise ValueError("The response format is incorrect or incomplete.")

        # Add nodes and edges to the LangGraph
        graph = lg.Graph()
        graph.add_node(restaurant_name, label='Restaurant')
        for dish in menu_items:
            graph.add_node(dish, label='Dish')
            graph.add_edge(restaurant_name, dish)

        # Display the recommendation
        st.subheader("Restaurant Recommendation")
        st.write(f"Restaurant Name: {restaurant_name}")
        st.write("Menu:")
        for dish in menu_items:
            st.write(f"- {dish}")

        # Visualize the LangGraph
        st.subheader("Recommendation Graph")
        graph_viz = lg.visualize(graph)
        st.pyplot(graph_viz)
    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
