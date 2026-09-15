import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 1. Page Configuration & Responsive UI Layout
st.set_page_config(
    page_title="UoBS Campus Assistant",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-text {
        color: #4B5563;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 25px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎓 University of Baltistan (UoBS) Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Your friendly guide for campus life, academic programs, admissions, and portal inquiries.</p>', unsafe_allow_html=True)

# 2. Secure API Key Setup for Groq
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ Please configure your `GROQ_API_KEY` in Streamlit Secrets (`.streamlit/secrets.toml`) or environment variables.")
    st.stop()

# 3. Initialize Stable Groq Model (`gemma2-9b-it`)
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="gemma2-9b-it",
        temperature=0.3,
        groq_api_key=api_key
    )

llm = get_llm()

# Built-in UoBS Knowledge Base System Prompt
uobs_system_message = SystemMessage(content=(
    "You are the official virtual assistant for the University of Baltistan, Skardu (UoBS). "
    "You are friendly, polite, helpful, and professional. "
    "Here is core background information about UoBS to help you answer questions:\n"
    "- Name: University of Baltistan, Skardu (UoBS), a chartered public university recognized by the HEC of Pakistan.\n"
    "- Location: Main Campus, Hussainabad / Skardu, Gilgit-Baltistan, Pakistan.\n"
    "- Key Faculties: Faculty of Natural Sciences & Technologies, Faculty of Life Sciences, and Faculty of Humanities & Social Sciences.\n"
    "- Popular Programs: BS Computer Science (BSCS), BS Software Engineering, and various undergraduate/graduate programs.\n"
    "- Portal Purpose: Assisting students with academic inquiries, department info, schedules, and general guidance.\n\n"
    "Always maintain a welcoming tone tailored for students and faculty members. If a user asks something outside your knowledge, guide them politely to check the official UoBS portal or administration office."
))

# 4. Chat Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Assalam-o-Alaikum! Welcome to the University of Baltistan portal assistant. How can I help you today?"}
    ]

# Render Quick Suggestion Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💻 Computing Programs"):
        st.session_state.messages.append({"role": "user", "content": "Tell me about the BS Computer Science and software programs at UoBS."})
with col2:
    if st.button("📍 Campus Location"):
        st.session_state.messages.append({"role": "user", "content": "Where is the main campus located?"})
with col3:
    if st.button("🏛️ Faculties Overview"):
        st.session_state.messages.append({"role": "user", "content": "What faculties are available at UoBS?"})

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input Field & Message Handling
if user_query := st.chat_input("Ask a question about UoBS (e.g., admissions, courses, campus)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            try:
                # Build message history for conversational context
                chat_history = [uobs_system_message]
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
                
                response = llm.invoke(chat_history)
                answer = response.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
