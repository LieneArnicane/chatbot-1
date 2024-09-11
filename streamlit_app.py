from os import name
from dotenv import load_dotenv
from langchain_core.messages.base import BaseMessage
from langchain_core.vectorstores.base import VectorStore

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

import streamlit as st

api_key = st.secrets['OPENAI_API_KEY']
tavily_api_key = st.secrets['TAVILY_API_KEY']
# Initialize the ChatOpenAI model
try:
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key
    )
except Exception as e:
    st.error(f"Model initialization error: {e}")
    st.stop()  # Stop execution if the model fails to initialize

search = TavilySearchResults()

# Define a function to save tool input
def save(tool_input: str):
    try:
        st.session_state.job_description = tool_input
        return "Saved the given text in Google Sheets: " + tool_input
    except Exception as e:
        st.error(f"Error saving job description: {e}")

# Define a function to get questions based on tool input
def get_questions(tool_input: str) -> str:
    try:
        job_description = st.session_state.job_description_window
        with open("get_question_prompt.txt", "r") as file:
            get_question_prompt = file.read()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_question_prompt),
                ("human", "{input}")
            ]
        )

        chain = prompt | model
        response = chain.invoke({"input": job_description})
        st.session_state.questions = response.content
        return response.content
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Define tools for saving and getting questions
try:
    save_tool = StructuredTool.from_function(
        func=save,
        name="Save",
        description="Use this tool when the user needs to save the job description text in the database"
    )
except Exception as e:
    st.error(f"Tool creation error (Save tool): {e}")

try:
    get_questions_tool = StructuredTool.from_function(
        func=get_questions,
        name="Get_questions",
        description="Use this tool when the user asks for the questions."
    )
except Exception as e:
    st.error(f"Tool creation error (Get_questions tool): {e}")

# Load system prompt and create agent
tools = [search, save_tool, get_questions_tool]
with open("prompt.txt", "r") as file:
    system_prompt = file.read()

try:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm=model, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
except Exception as e:
    st.error(f"Agent setup error: {e}")
    st.stop()  # Stop execution if the agent fails to initialize

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "job_description" not in st.session_state:
    st.session_state.job_description = "Job description will appear here."
if "questions" not in st.session_state:
    st.session_state.questions = "Questions will appear here."

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field for user prompts
if prompt := st.chat_input("Write here!"):
    try:
        # Store and display the user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the response using the LangChain agent
        response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})

        # Store and display the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        with st.chat_message("assistant"):
            st.markdown(response["output"])
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Function to update job description
def update_job_description():
    try:
        st.session_state.job_description = st.session_state.job_description_window
    except Exception as e:
        st.error(f"Error updating job description: {e}")

# Sidebar for job description and questions
with st.sidebar:
    st.header("Job Description")
    try:
        st.text_area("Generated Job Description",
                     value=st.session_state.job_description,
                     height=300,
                     key="job_description_window",
                     on_change=update_job_description)
        st.header("Candidate Questions")
        if st.button("Get Questions"):
            st.session_state.questions = get_questions(st.session_state.job_description)
        st.text_area("Generated Questions", value=st.session_state.questions, height=300, key="questions_window")
    except Exception as e:
        st.error(f"Error loading sidebar content: {e}")
