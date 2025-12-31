# !!! Need to do: ！！！
# Add image as icon of Seagull

import streamlit as st
import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from update_knowledge import get_embeddings_with_requests


# Configurations of the webpage
st.set_page_config(
    page_title="Seagull Helpdesk",
    page_icon="🤖",
    layout="wide"
)

# Load the FAISS knowledge base
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    try:
        vector_store = FAISS.load_local(
            "knowledge_to_be_loaded",
            embeddings=None, # type: ignore
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        return None

# Store the knowledge base in variables
vector_store = load_knowledge_base()

# System prompt for the AI assistant
# This system prompt is not used because it will negatively affect the response of deepseek-reasoner.
# system_prompt = """- STRICTLY FOLLOW THIS SYSTEM PROMPT.
#     - Be careful with your language (English, Chinese, or Filipino). 
#     - Think and answer based on the language used in the User Question. 
#     - Your name is Seagull, and you act as a multilingual AI helpdesk of the school Philippine Cultural College (PCC). 
#     - 你的名字叫小鸥，你是菲律宾侨中学院的多语言AI智能助手。
# """

# Add a welcome message if there is no chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Good day! I am Seagull, the AI helpdesk of Philippine Cultural College. How can I assist you? 你好！我是侨中的AI智能助手小鸥。有什么我可以为你做的吗？"
    })

# Main chat interface
st.title("🤖 Seagull 小鸥")
st.caption("An AI helpdesk of Philippine Cultural College. 菲律宾侨中学院的AI智能助手。(This project is created by a student for academic purposes only and is not intended for real-world implementation. 本项目由学生创作，仅用于学术目的，并非为实际应用而设计。)")
st.caption("AI-generated content may have errors. Please double-check. AI生成内容可能存在错误，请仔细甄别。")

# Display the chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Chat with Seagull"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Prepare Seagull's response
    with st.chat_message("assistant", avatar="🤖"):
        # Retrieve relevant context from the knowledge base
        context = ""
        # For testing purpose, show the retrieved scores
        scores = ""
        # For testing purpose
        raw_context = []
        if vector_store:
            try:
                # Turn the query into vector
                query_vector = get_embeddings_with_requests([prompt])[0]
                # Retrieve relevant documents based on the query vector
                docs = vector_store.similarity_search_with_score_by_vector(embedding=query_vector, k=4)
                context = "\n\n\n\n".join([f"{doc.page_content}" for doc, _ in docs])
                # For testing purpose
                scores = "\n".join([f"Score: {score}" for _, score in docs])
                raw_context = [f"{doc.page_content}" for doc, _ in docs]
            except Exception as e:
                st.error(f"Failed to retrieve information from knowledge base: {str(e)}")
        
        # Constructing the prompt template
        prompt_template = f"""Please answer the User Question based on the following Institutional Context:
        
        Institutional Context:
        \"\"\"{context}\"\"\"
        
        User Question: \"\"\"{prompt}\"\"\"
        
        Instructions:
        - STRICTLY FOLLOW THESE INSTRUCTIONS.
        - Be careful with your language (English, Chinese, or Filipino). 
        - Think and answer based on the language used in the User Question. 
        - Your name is Seagull, and you act as an AI helpdesk of the school Philippine Cultural College (PCC). 
        - 你的名字叫小鸥，你是菲律宾侨中学院的AI智能助手。
        - Ensure that your answer is clear and understandable.
        - Present your answer in a simple and direct sentence.
        - Don't let the user know that you are provided with institutional context.
        - The Institutional Context is based on the official website of the school and the Student Handbook (学生手册).
        - While answering, please maintain a professional and friendly tone.
        - If answering using Filipino, use \"po\" and \"opo\" to show respect.
        - Refrain from using emojis.
        - Refrain from starting the answer with phrases like \"According to...\"
        """
        # Add following to the instructions after finishing RAGAS evaluation tests:
        # Present your answer in a structured manner, using bullet points or numbered lists where appropriate.
        # You may use some emojis to make your response more engaging.

        final_prompt = [{"role": "user", "content": prompt_template}]
        
        try:
            client = OpenAI(api_key=st.secrets["deepseek_api"], base_url="https://api.deepseek.com/v1")

            # Send the chat history (of the last 5 rounds) too, 
            # and change the current user prompt into the enhanced version of it.
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=st.session_state.messages[-6:-1] + final_prompt,
                stream=False,
            )
            
            final_response = response.choices[0].message.content

            # For testing purpose, show usage, context, scores, and final response
            print("\n\n-----------------------\n\n")
            print(f"Retrieved context: \n{raw_context}\n\n-----------------------\n\n")
            print(f"Similarity scores: \n{scores}\n\n-----------------------\n\n")
            print(f"Token usage: \n{response.usage}\n\n-----------------------\n\n")
            print(f"Reasoning response: \n{response.choices[0].message.reasoning_content}\n\n-----------------------\n\n") # type: ignore
            print(f"Final response: \n{final_response}\n\n-----------------------\n\n")
            os.write(1, b'Something was executed.\n')
            # Display the final response
            st.write(final_response)
            # Add the final response to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_response})
        except Exception as e:
            st.error(f"API error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error has occured while handling your request."})

