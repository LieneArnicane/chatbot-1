from os import name
from dotenv import load_dotenv
from langchain_core.messages.base import BaseMessage
from langchain_core.vectorstores.base import VectorStore
from pydantic.v1 import ValidationError

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

import streamlit as st

api_key= st.secrets['OPENAI_API_KEY']

try:
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key
    )
except ValidationError as e:
    st.error(f"Model initialization error: {e}")
    st.stop()  # Stop execution if the model fails to initialize

search = TavilySearchResults()

def save(tool_input: str):
    try:
        with open("job_description.txt", "w") as file:
            file.write(tool_input)

        st.session_state.job_description = tool_input
        return "Saved the given text in Google Sheets: " + tool_input
    except Exception as e:
        st.error(f"Error saving job description: {e}")

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
    except ValidationError as e:
        st.error(f"Error generating questions: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

try:
    save_tool = StructuredTool.from_function(
        func=save,
        name="Save",
        description="Use this tool when the user needs to save the job description text in the database"
    )
except ValidationError as e:
    st.error(f"Tool creation error (Save tool): {e}")

try:
    get_questions_tool = StructuredTool.from_function(
        func=get_questions,
        name="Get_questions",
        description="Use this tool when the user asks for the questions."
    )
except ValidationError as e:
    st.error(f"Tool creation error (Get_questions tool): {e}")

# here starts the GPT code

tools = [search, save_tool, get_questions_tool]

try:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """"You are an experienced recruiter. I am a client looking to hire a new team member. Your task is to gather all the necessary details to create a comprehensive job description. You should ask questions one by one, covering all relevant aspects such as job title, responsibilities, required skills, experience, qualifications, and any other important details, like information about the company. Ask about why the job is desirable - selling points. Ask about benefits. If the answers are not satisfactory then ask persistently in different ways. Only if the user writes to continue, then let it go. After you have asked all the necessary questions, compile the information and create a job description, including sections for the job title, job summary, responsibilities, qualifications, required skills, experience, and any other relevant information. Answer shortly and ask the questions one by one. Continuesly analyse the given answers to understand if the given information is enough. The job description should include: job summary, offer, perks and benefits, 'our pitch', position key duties,  if about a tech job - tech stack, otherwise necessary skills, requirements - your capabilities, track record, extra credit, team, hiring process, selection process, location.

Do NOT include any information that the user has not given. 

After you have given the job description (in the chat give the job description in text format) ask if the client is satisfied. If not - ask why and redo the job description. If yes, then save the job description with the tool 'save'. After ask if the client wants to get questions for candidate search on HyperScan, based on the job description. If yes, then use the tool get_questions and give the questions to the user."""),  # Truncated for readability
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm=model, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
except ValidationError as e:
    st.error(f"Agent setup error: {e}")
    st.stop()  # Stop execution if the agent fails to initialize

# Create a session state variable to store the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "job_description" not in st.session_state:
    st.session_state.job_description = "Job description will appear here."
if "questions" not in st.session_state:
    st.session_state.questions = "Questions will appear here."

# Display the existing chat messages via `st.chat_message`
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field for user prompts
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
    except ValidationError as e:
        st.error(f"Error processing input: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def update_job_description():
    try:
        st.session_state.job_description = st.session_state.job_description_window
    except Exception as e:
        st.error(f"Error updating job description: {e}")

with st.sidebar:
    st.header("Job Description")
    try:
        with open("job_description.txt", "r") as file:
            st.session_state.job_description = file.read()
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
