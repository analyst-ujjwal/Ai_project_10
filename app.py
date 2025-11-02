import streamlit as st
from agent import Agent

# Initialize the proactive agent
agent = Agent()

# Streamlit page setup
st.set_page_config(page_title="Proactive AI Agent", page_icon="🤖", layout="wide")

# Title
st.title("🤖 Project - AI Agent (Groq + LLaMA)")

# Display available tools
st.markdown("""
### 🧰 **Available Tools**
Here are all the tools this AI Agent can use:
- 🧮 **calculator** — Perform mathematical calculations  
- ✂️ **summarizer** — Summarize long text into concise form  
- 🌍 **translate** — Translate text between English and Hindi  
- 💭 **sentiment** — Analyze sentiment of text (positive/negative/neutral)  
- 📚 **research_plan** — Generate a step-by-step research plan  
- 🔮 **proactivity** — Make proactive suggestions based on context  
- 📝 **logger** — Log and recall previous chat sessions  
- 🧠 **knowledge** — Retrieve stored or factual knowledge  
- 📊 **data_analyzer** — Analyze tabular or numerical data  
- 📆 **reminder** — Set or simulate reminders  
- 📈 **trend_detector** — Detect emerging trends from text input  
- 🧾 **fact_checker** — Check factual accuracy of a claim  
- 🧰 **code_helper** — Explain or debug Python code  
- 🎨 **idea_generator** — Generate creative ideas for projects or problems  
- 🗣️ **conversation_memory** — Maintain memory of past chat  
- 💡 **insight_extractor** — Extract key insights from any text  
- ⚙️ **system_info** — Provide runtime system information
""")

# Chat section
st.markdown("---")
st.subheader("💬 Chat with the Agent")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**🧑‍💻 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Agent:** {msg['content']}")

# Input box
user_input = st.text_input("Enter your message:", key="user_input")

# Send button
if st.button("Send"):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            response = agent.handle(user_input)

            # If the response is a dictionary, extract the best readable part
            if isinstance(response, dict):
                response = (
                    response.get("plan")
                    or response.get("summary")
                    or response.get("note")
                    or response.get("decision")
                    or str(response)
                )

        except Exception as e:
            response = f"⚠️ Error: {e}"

        st.session_state.chat_history.append({"role": "agent", "content": response})
        st.rerun()

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.markdown("---")
st.caption("Built using Streamlit, Groq API, and LLaMA — Project by Ujjwal 🚀")
