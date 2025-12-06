import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
import os
import time
import asyncio

load_dotenv()

EXTERNAL_MODEL = "gemini-2.0-flash-exp"  # Fastest available model

st.set_page_config(
    page_title=f"AI Study Assistant ({EXTERNAL_MODEL} + LangChain)", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 AI Study Assistant")
st.markdown(
    f"""
    Ask your **{EXTERNAL_MODEL}** LLM for a concise, educational summary on any topic. 
    The assistant will also suggest follow-up questions to test your knowledge.
    ---
    """
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hello! What topic would you like to study today?"}
    ]

# Initialize the LLM chain with aggressive optimization
@st.cache_resource
def get_llm_chain():
    """Initializes and returns the optimized LLM chain."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            st.error("API key not found in secrets or environment!")
            return None
        
        llm = ChatGoogleGenerativeAI(
            model=EXTERNAL_MODEL, 
            temperature=0.5,  # Lower for faster, more focused responses
            max_output_tokens=800,  # Reduced for speed
            timeout=20,  # Shorter timeout
            max_retries=1,  # Fewer retries
            streaming=True,  # Enable streaming explicitly
            convert_system_message_to_human=True  # Better compatibility
        )
        
        # More concise prompt for faster responses
        system_prompt = (
            "You are a concise study assistant. Provide:\n"
            "1. A brief summary (100 words max)\n"
            "2. Three follow-up questions\n"
            "Keep it short and clear."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{topic}")
        ])
        
        chain = prompt | llm | StrOutputParser() 
        return chain
        
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None

# Get the chain
llm_chain = get_llm_chain()

# --- Chat Interface ---
if llm_chain is not None:
    
    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_topic := st.chat_input("Ask me anything..."):
        
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_topic})
        with st.chat_message("user"):
            st.markdown(user_topic)

        # Generate AI response
        with st.chat_message("assistant"):
            start_time = time.time()
            
            try:
                # Use streaming for immediate feedback
                full_response = ""
                response_container = st.empty()
                
                for chunk in llm_chain.stream({"topic": user_topic}):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                
                # Show timing if slow
                elapsed = time.time() - start_time
                if elapsed > 3:
                    st.caption(f"⏱️ {elapsed:.1f}s")
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": full_response
                })

            except Exception as e:
                elapsed = time.time() - start_time
                
                # More detailed error for debugging
                error_details = str(e)
                st.error(f"Error after {elapsed:.1f}s")
                st.code(error_details, language="text")
                
                # Fallback to non-streaming
                with st.spinner("Retrying..."):
                    try:
                        result = llm_chain.invoke({"topic": user_topic})
                        st.markdown(result)
                        st.session_state["messages"].append({
                            "role": "assistant", 
                            "content": result
                        })
                    except Exception as e2:
                        error_msg = f"❌ Failed: {str(e2)[:100]}"
                        st.error(error_msg)
                        st.info("💡 Tip: Try a simpler question or check if the API is experiencing issues.")
else:
    st.error("❌ Failed to initialize. Check your GEMINI_API_KEY in Streamlit secrets.")
