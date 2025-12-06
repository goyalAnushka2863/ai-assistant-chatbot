import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
import os

load_dotenv()

EXTERNAL_MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title=f"AI Study Assistant ({EXTERNAL_MODEL} + LangChain)", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 AI Study Assistant")
st.markdown(
    f"""
    Ask your **{EXTERNAL_MODEL}** LLM (using the **LangChain** framework) for a concise, educational summary on any topic. 
    The assistant will also suggest follow-up questions to test your knowledge.
    ---
    """
)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hello! I am ready to generate study guides using the {EXTERNAL_MODEL} model. What topic would you like to study today?"}
    ]

# Initialize the LLM and Prompt Template (cached)
@st.cache_resource
def get_llm_chain():
    """Initializes and returns the LLM chain."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=EXTERNAL_MODEL, 
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        system_prompt = (
            "You are an expert, concise study assistant. "
            "Respond using clear Markdown formatting."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{topic}")
        ])
        
        chain = prompt | llm | StrOutputParser() 
        return chain
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.warning(
            f"Please ensure your {EXTERNAL_MODEL} API key is correctly configured "
            "in your environment variables or Streamlit secrets."
        )
        return None

# Get the chain (this will be cached after first run)
llm_chain = get_llm_chain()

# --- Application Logic (Chat Interface) ---
if llm_chain is not None:
    # 1. Display Chat History
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Handle New User Input
    if user_topic := st.chat_input(f"Ask me anything..."):
        
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_topic})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_topic)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Generating study guide for '{user_topic}'..."):
                try:
                    stream = llm_chain.stream({"topic": user_topic})
                    full_response = st.write_stream(stream)
                    
                    # Add assistant response to chat history
                    st.session_state["messages"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_message = f"An error occurred during generation: {e}"
                    st.error(error_message)
                    st.warning("Please check your Gemini API key and network connection.")
                    st.session_state["messages"].append({"role": "assistant", "content": error_message})
else:
    st.error("Failed to initialize the AI assistant. Please check your API key configuration.")
