
import streamlit as st
from graph import app

st.set_page_config(page_title="Query Router Chatbot", page_icon="🤖")

st.title("🤖 Query Router Chatbot")

# Initialize memory
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display chat history
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query with memory
    with st.spinner("Thinking..."):
        final_state = app.invoke({
            "user_query": prompt,
            "history": st.session_state.conversation
        })
        answer = final_state.get("final_answer", "Sorry, I couldn't generate an answer.")
        route = final_state.get("route", "Unknown")
        if route == "Please try again later":
            answer = "I'm sorry, but I cannot process this request at the moment."
            

    bot_response = f"**Routing:** {route}\n\n**Answer:** {answer}"

    # Add bot message
    st.session_state.conversation.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
