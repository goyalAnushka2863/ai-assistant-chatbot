import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # if using Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# --- Configuration ---
# You can change the model name if you pulled a different one (e.g., 'mistral', 'gemma:2b')
# OLLAMA_MODEL = "llama3"

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="AI Study Assistant (Ollama + LangChain)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 AI Study Assistant")
st.markdown(
    """
    Enter any topic or question, and your local **Ollama** LLM (using the **LangChain** framework) will generate a concise, educational summary and suggest follow-up questions.
    ---
    """
)

# Initialize the Ollama LLM and Prompt Template
@st.cache_resource
def setup_llm_chain():
    """Initializes the Ollama model and LangChain prompt template."""
    try:
        # 1. Initialize the LLM (Connects to the running Ollama instance)
        # llm = OllamaLLM(model=OLLAMA_MODEL)
        
        # 2. Define the Prompt Template
        system_prompt = (
            "You are an expert, concise study assistant. "
            "Your task is to provide an educational summary of the user's topic and suggest 3 related, challenging follow-up questions to test their knowledge. "
            "Format your response clearly with a 'Summary' section and a 'Follow-Up Questions' section."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{topic}")
        ])
        
        # 3. Create the Chain
        chain = prompt | llm
        return chain, True
    except Exception as e:
        # This typically fails if Ollama is not running or the model is not pulled.
        st.error(f"Error initializing Ollama or LangChain: {e}")
        st.warning(
            "Please ensure Ollama is running and the model "
            f"'{OLLAMA_MODEL}' is pulled (e.g., `ollama pull {OLLAMA_MODEL}`)."
        )
        return None, False

llm_chain, is_ready = setup_llm_chain()

# --- Application Logic ---
if is_ready and llm_chain:
    # User Input
    user_topic = st.text_area(
        "Enter your study topic (e.g., 'The Krebs Cycle', 'The causes of World War I', or 'Quick summary of React Hooks')",
        key="topic_input",
        height=100
    )

    # Generation Button
    if st.button("Generate Study Guide", type="primary"):
        if user_topic:
            # Use a spinner while the LLM processes the request
            with st.spinner(f"Asking {OLLAMA_MODEL} to generate your study guide..."):
                try:
                    # Invoke the LangChain
                    response = llm_chain.invoke({"topic": user_topic})
                    
                    # Display the result
                    st.divider()
                    st.subheader("📝 Generated Study Guide")
                    st.markdown(response)

                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")
                    st.warning("Ensure the Ollama service is still running and accessible.")
        else:
            st.warning("Please enter a topic before generating the guide.")
    
    st.caption(f"Powered by LangChain and {OLLAMA_MODEL} via Ollama.")

# --- How to Run Instructions (for the user) ---
st.sidebar.markdown("### 🏃 How to Run This App")
st.sidebar.code("streamlit run study_assistant.py")
st.sidebar.markdown(
    """
    **Prerequisites:**
    1.  Install Python dependencies: 
        `pip install streamlit langchain langchain-core langchain-community`
    2.  Install Ollama and pull the model:
        `ollama pull llama3`
    3.  Ensure Ollama is running: 
        `ollama serve`
    """
)
