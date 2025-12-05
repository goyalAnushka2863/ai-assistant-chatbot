import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # if using Gemini
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
st.set_page_config(
    page_title="AI Study Assistant (Ollama + LangChain)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 AI Study Assistant")


# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hello! I am ready to generate study guides using the gemini model. What topic would you like to study today?"}
    ]
if "is_ready" not in st.session_state:
    st.session_state["is_ready"] = False
if "llm_chain" not in st.session_state:
    st.session_state["llm_chain"] = None


# Initialize the Ollama LLM and Prompt Template
@st.cache_resource
def setup_llm_chain():
    """Initializes the Ollama model and LangChain prompt template."""
    st.info(f"Attempting to initialize LangChain with Ollama model...")
    try:
        # 1. Initialize the LLM (Connects to the running Ollama instance)
        
        # 2. Define the Prompt Template
        system_prompt = (
            "You are an expert, concise study assistant. "
            "Your task is to provide an educational summary of the user's topic and suggest 3 related, challenging follow-up questions to test their knowledge. "
            "Format your response clearly with a 'Summary' section and a 'Follow-Up Questions' section. "
            "Respond using clear Markdown formatting."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{topic}")
        ])
        
        # 3. Create the Chain
        # Add a StrOutputParser to ensure clean string output
        chain = prompt | llm 
        st.session_state["llm_chain"] = chain
        st.session_state["is_ready"] = True
        st.success(f"Assistant ready!")
        return True
    except Exception as e:
        st.error(f"Error initializing Ollama or LangChain: {e}")
        # st.warning(
        #     "Please ensure Ollama is running and the model "
        #     f"'{OLLAMA_MODEL}' is pulled (e.g., `ollama pull {OLLAMA_MODEL}`)."
        # )
        st.session_state["is_ready"] = False
        return False

# Setup the chain once
if not st.session_state["is_ready"]:
    setup_llm_chain()

# --- Application Logic (Chat Interface) ---
if st.session_state["is_ready"] and st.session_state["llm_chain"]:
    
    # 1. Display Chat History
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Handle New User Input
    if user_topic := st.chat_input("Ask me anything"):
        
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_topic})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_topic)

        # Generate response
        with st.chat_message("assistant"):
            # Use a spinner while the LLM processes the request
            with st.spinner(f"Asking assistant to generate study guide for '{user_topic}'..."):
                try:
                    # Invoke the LangChain
                    llm_chain = st.session_state["llm_chain"]
                    # Use stream to show response chunk-by-chunk for a better chat feel
                    stream = llm_chain.stream({"topic": user_topic})
                    
                    full_response = st.write_stream(stream)
                    
                    # Add assistant response to chat history
                    st.session_state["messages"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_message = f"An error occurred during generation: {e}"
                    st.error(error_message)
                    st.warning("Ensure the Ollama service is still running and accessible.")
                    st.session_state["messages"].append({"role": "assistant", "content": error_message})

