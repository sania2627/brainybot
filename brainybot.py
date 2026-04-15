from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image

# ------------------ Setup ------------------
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(
    page_title="BrainyBot",
    page_icon="icon.png",
    layout="centered"
)

# ------------------ Header ------------------
logo = Image.open("logo.png")
col1, col2 = st.columns([1, 8])

with col1:
    st.image(logo, width=90)

with col2:
    st.markdown(
        """
        <h1 style="
            font-family: 'Space Mono';
            font-size: 44px;
            margin: 0;
            color: #2c3e50;
        ">BrainyBot</h1>
        """,
        unsafe_allow_html=True
    )

st.caption("Powered by LangChain + Ollama")
st.divider()

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "first_input_received" not in st.session_state:
    st.session_state.first_input_received = False  # Track first user input

# ------------------ LangChain ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, friendly AI assistant."),
        ("user", "{question}")
    ]
)

llm = Ollama(model="gemma")
chain = prompt | llm | StrOutputParser()

# ------------------ Chat Bubble (BrainyBot Style) ------------------


def chat_bubble(role, text):
    if role == "assistant":
        align = "left"
        bg = "#F3F4F6"        # Light gray assistant bubble
        color = "#111827"
    else:
        align = "right"
        bg = "#6063FE"        # User bubble (blue-violet)
        color = "#FFFFFF"

    st.markdown(
        f"""
        <div style="
            max-width: 75%;
            background: {bg};
            color: {color};
            padding: 10px 16px;
            border-radius: 50px;
            margin-bottom: 8px;
            float: {align};
            clear: both;
            font-size: 15px;
            line-height: 1.5;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        ">
            {text.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )



# ------------------ Display Chat History ------------------
for msg in st.session_state.messages:
    chat_bubble(msg["role"], msg["content"])


# ------------------ Input ------------------
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Mark first input received
    st.session_state.first_input_received = True

    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    chat_bubble("user", user_input)

    # Assistant response with spinner
    with st.spinner("BrainyBot is thinking..."):
        response = chain.invoke({"question": user_input})

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    chat_bubble("assistant", response)

# ------------------ Styled Clear Button ------------------
if st.session_state.first_input_received:
    col_clear, col_spacer = st.columns([1, 6])  # Left aligned
    with col_clear:
        st.markdown(
            """
            <style>
            .clear-button button {
        

                padding: 5px 10px;
                font-size: 14px;
                border-radius: 50px;
                background-color: #F0FFDF;
                color: #A8DF8E;
                border: 1px solid #F0FFDF;
                cursor: pointer;
            }
            .clear-button button:hover {
                background-color: #e2e4e8;
            }
            </style>
            <div class="clear-button">
                <form action="" method="get">
                    <button type="submit">Clear Chat</button>
                </form>
            </div>
            """,
            unsafe_allow_html=True
        )