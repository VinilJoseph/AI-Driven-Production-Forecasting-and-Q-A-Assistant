import faiss
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
import streamlit as st

def show_chatbot_page():
    # Load the embeddings
    with open("embedding.pkl", "rb") as f:
        embeddings = pickle.load(f)

    # Load the FAISS index
    index = faiss.read_index("vector_store.index")

    # Load the docstore and index_to_docstore_id
    with open("docstore.pkl", "rb") as f:
        docstore = pickle.load(f)

    with open("index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)

    # Create the FAISS vector store
    vector_store = FAISS(index=index, embedding_function=embeddings, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    # Initialize the LLM
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

    # Initialize the prompt with a hint to the model to limit the response length
    prompt_template = """
    You are a sophisticated petroleum engineering assistant
    with expertise in oil & gas operations, drilling, production, 
    and reservoir engineering. Your responses should be accurate, 
    clear, and based solely on the provided context.

    Instructions for response:
    1. Analyze the context carefully before responding
    2. Focus only on information present in the context
    3. Structure your response with:
    - Main explanation of the concept/answer
    - Supporting technical reasoning
    - If applicable, practical implications
    4. Keep responses concise and technically accurate

    {context}

    Question: {input}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create document and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Initialize memory
    memory = ConversationBufferMemory(return_messages=True)

    # Streamlit UI
    st.title("PDF-based Q&A Chatbot 🗒️ ")

    # Notification Section
    st.markdown("""
    <div style='background-color: rgba(35, 45, 55, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>About This Assistant</h3>
        <p>Welcome to your Petroleum Engineering Knowledge Assistant! This chatbot uses RAG (Retrieval Augmented Generation) 
        technology to answer your questions based on petroleum engineering textbooks and documentation.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 Usage Guidelines"):
        st.markdown("""
        ### You can ask questions about:
        - Petroleum engineering concepts and principles
        - Oil and gas equipment and machinery
        - Drilling operations and techniques
        - Well completion and production
        - Reservoir characteristics
        - Field development and operations
        - Safety procedures and protocols

        ### Example Questions:
        1. "What are the main components of a drilling rig?"
        2. "Explain the function of Christmas tree in well completion?"
        3. "What factors affect reservoir pressure?"
        4. "How does a mud pump work?"
        """)

    query = st.text_input("Ask a question about the PDF")

    def ensure_complete_sentence(response):
        # Ensure the response ends with proper punctuation
        response = response.strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        return response

    if query:
        response = retrieval_chain.invoke({"input": query})
        answer = response["answer"]
        answer_marker = "Answer:"
        start_index = answer.find(answer_marker)

        if start_index != -1:
            generated_output = answer[start_index + len(answer_marker):].strip()
            formatted_output = "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())

            # Ensure the response is a complete sentence
            complete_output = ensure_complete_sentence(formatted_output)
            st.write(complete_output)
        else:
            formatted_output = answer.strip()
            complete_output = ensure_complete_sentence(formatted_output)
            st.write("Answer marker not found. Here is the raw response:")
            st.write(complete_output)

        memory.save_context({"input": query}, {"output": complete_output})


# ================================================3

# import faiss
# import pickle
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.llms import HuggingFaceHub
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.memory import ConversationBufferMemory
# import streamlit as st

# def show_chatbot_page():
#     # Load the embeddings
#     with open("embedding.pkl", "rb") as f:
#         embeddings = pickle.load(f)

#     # Load the FAISS index
#     index = faiss.read_index("vector_store.index")

#     # Load the docstore and index_to_docstore_id
#     with open("docstore.pkl", "rb") as f:
#         docstore = pickle.load(f)

#     with open("index_to_docstore_id.pkl", "rb") as f:
#         index_to_docstore_id = pickle.load(f)

#     # Create the FAISS vector store
#     vector_store = FAISS(index=index, embedding_function=embeddings, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

#     # Initialize the LLM
#     llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

#     # Initialize the prompt with a hint to the model to limit the response length
#     prompt_template = """
#     You are a sophisticated petroleum engineering assistant
#     with expertise in oil & gas operations, drilling, production, 
#     and reservoir engineering. Your responses should be accurate, 
#     clear, and based solely on the provided context.

#     Instructions for response:
#     1. Analyze the context carefully before responding
#     2. Focus only on information present in the context
#     3. Structure your response with:
#     - Main explanation of the concept/answer
#     - Supporting technical reasoning
#     - If applicable, practical implications
#     4. Keep responses concise and technically accurate

#     {context}

#     Question: {input}
#     Answer:
#     """
#     prompt = ChatPromptTemplate.from_template(prompt_template)

#     # Create document and retrieval chain
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = vector_store.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     # Initialize memory
#     memory = ConversationBufferMemory(return_messages=True)

#     # Streamlit UI
#     st.title("PDF-based Q&A Chatbot 🗒️ ")

#     query = st.text_input("Ask a question about the PDF")

#     def ensure_complete_sentence(response):
#         # Ensure the response ends with proper punctuation
#         response = response.strip()
#         if not response.endswith(('.', '!', '?')):
#             response += '.'
#         return response

#     if query:
#         response = retrieval_chain.invoke({"input": query})
#         answer = response["answer"]
#         answer_marker = "Answer:"
#         start_index = answer.find(answer_marker)

#         if start_index != -1:
#             generated_output = answer[start_index + len(answer_marker):].strip()
#             formatted_output = "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())

#             # Ensure the response is a complete sentence
#             complete_output = ensure_complete_sentence(formatted_output)
#             st.write(complete_output)
#         else:
#             formatted_output = answer.strip()
#             complete_output = ensure_complete_sentence(formatted_output)
#             st.write("Answer marker not found. Here is the raw response:")
#             st.write(complete_output)

#         memory.save_context({"input": query}, {"output": complete_output})

# ==========================================================================================

# import faiss
# import pickle
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.llms import HuggingFaceHub

# # from langchain_community.llms import HuggingFaceHub

# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.memory import ConversationBufferMemory
# import streamlit as st



# # import faiss
# # import pickle
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain.llms import HuggingFaceHub
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain.memory import ConversationBufferMemory
# # import streamlit as st



# def show_chatbot_page():
#     # Load the embeddings and other components
#     with open("embedding.pkl", "rb") as f:
#         embeddings = pickle.load(f)

#     index = faiss.read_index("vector_store.index")

#     with open("docstore.pkl", "rb") as f:
#         docstore = pickle.load(f)

#     with open("index_to_docstore_id.pkl", "rb") as f:
#         index_to_docstore_id = pickle.load(f)

#     vector_store = FAISS(index=index, embedding_function=embeddings, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

#     llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

#     # Welcome section
#     st.title("🛢️ Petroleum Engineering Knowledge Assistant")
    
#     # Information section
#     with st.container():
#         st.markdown("""
#         <div style='background-color: rgba(35, 45, 55, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
#         <h3>About This Assistant</h3>
#         <p>Welcome to your Petroleum Engineering Knowledge Assistant! This chatbot uses RAG (Retrieval Augmented Generation) 
#         technology to answer your questions based on petroleum engineering textbooks and documentation.</p>
#         </div>
#         """, unsafe_allow_html=True)

#     # Guidelines section
#     with st.expander("📚 Usage Guidelines"):
#         st.markdown("""
#         ### You can ask questions about:
#         - Petroleum engineering concepts and principles
#         - Oil and gas equipment and machinery
#         - Drilling operations and techniques
#         - Well completion and production
#         - Reservoir characteristics
#         - Field development and operations
#         - Safety procedures and protocols

#         ### Example Questions:
#         1. "What are the main components of a drilling rig?"
#         2. "Explain the function of Christmas tree in well completion?"
#         3. "What factors affect reservoir pressure?"
#         4. "How does a mud pump work?"
#         """)

#     # Prompt template
#     prompt_template = """
#         You are a sophisticated petroleum engineering assistant
#         with expertise in oil & gas operations, drilling, production, 
#         and reservoir engineering. Your responses should be accurate, 
#         clear, and based solely on the provided context.

#         Instructions for response:
#         1. Analyze the context carefully before responding
#         2. Focus only on information present in the context
#         3. Structure your response with:
#         - Main explanation of the concept/answer
#         - Supporting technical reasoning
#         - If applicable, practical implications
#         4. Keep responses concise and technically accurate

#         Context: {context}

#         Question: {input}

#         Answer: Let me explain this clearly based on the provided information.
#         """
#     prompt = ChatPromptTemplate.from_template(prompt_template)

#     # Create document and retrieval chain
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = vector_store.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     # Initialize memory
#     memory = ConversationBufferMemory(return_messages=True)

#     # Chat interface
#     st.markdown("### Ask Your Question 💭")
#     # query = st.text_input(
#     #     "",
#     #     placeholder="Enter your petroleum engineering related question here...",
#     #     key="query_input"
#     # )
#     query = st.text_input(
#         "Your Question",
#         placeholder="Enter your petroleum engineering related question here...",
#         key="query_input",
#         label_visibility="collapsed"
#     )


#     def ensure_complete_sentence(response):
#         response = response.strip()
#         if not response.endswith(('.', '!', '?')):
#             response += '.'
#         return response

#     if query:
#         # Add a spinner while processing
#         with st.spinner('Retrieving information...'):
#             response = retrieval_chain.invoke({"input": query})
#             answer = response["answer"]
#             answer_marker = "Answer:"
#             start_index = answer.find(answer_marker)

#             # Display answer in a nice container
#             st.markdown("### Answer:")
#             with st.container():
#                 if start_index != -1:
#                     generated_output = answer[start_index + len(answer_marker):].strip()
#                     formatted_output = "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())
#                     complete_output = ensure_complete_sentence(formatted_output)
#                     st.markdown(f"""
#                     <div style='background-color: rgba(35, 45, 55, 0.8); padding: 20px; border-radius: 10px;'>
#                     {complete_output}
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     formatted_output = answer.strip()
#                     complete_output = ensure_complete_sentence(formatted_output)
#                     st.warning("Answer marker not found. Here is the raw response:")
#                     st.markdown(f"""
#                     <div style='background-color: rgba(35, 45, 55, 0.8); padding: 20px; border-radius: 10px;'>
#                     {complete_output}
#                     </div>
#                     """, unsafe_allow_html=True)

#             memory.save_context({"input": query}, {"output": complete_output})

#     # Footer with disclaimer
#     st.markdown("""
#     <div style='background-color: rgba(35, 45, 55, 0.8); padding: 15px; border-radius: 10px; margin-top: 20px;'>
#     <h4>📝 Note</h4>
#     <p>This assistant uses RAG (Retrieval Augmented Generation) technology to provide accurate information based on petroleum engineering textbooks. 
#     While it strives for accuracy, please verify critical information from official sources.</p>
#     </div>
#     """, unsafe_allow_html=True)


# ===================================================================

# # Call the function to display the chatbot page
# show_chatbot_page()

# ===========================================================================
# The below code is to run the page alone
# ===========================================================================

# import faiss
# import pickle
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.llms import HuggingFaceHub
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.memory import ConversationBufferMemory
# import streamlit as st


# # Load the embeddings
# with open("embedding.pkl", "rb") as f:
#     embeddings = pickle.load(f)

# # Load the FAISS index
# index = faiss.read_index("vector_store.index")

# # Load the docstore and index_to_docstore_id
# with open("docstore.pkl", "rb") as f:
#     docstore = pickle.load(f)

# with open("index_to_docstore_id.pkl", "rb") as f:
#     index_to_docstore_id = pickle.load(f)

# # Create the FAISS vector store
# vector_store = FAISS(index=index, embedding_function=embeddings, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# # Initialize the LLM
# llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')

# # Initialize the prompt with a hint to the model to limit the response length
# prompt_template = """
#         You are a sophisticated petroleum engineering assistant
#         with expertise in oil & gas operations, drilling, production, 
#         and reservoir engineering. Your responses should be accurate, 
#         clear, and based solely on the provided context.

#         Instructions for response:
#         1. Analyze the context carefully before responding
#         2. Focus only on information present in the context
#         3. Structure your response with:
#         - Main explanation of the concept/answer
#         - Supporting technical reasoning
#         - If applicable, practical implications
#         4. Keep responses concise and technically accurate

#         Context: {context}

#         Question: {input}

#         Answer: Let me explain this clearly based on the provided information.
#         """
# prompt = ChatPromptTemplate.from_template(prompt_template)

# # Create document and retrieval chain
# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = vector_store.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Initialize memory
# memory = ConversationBufferMemory(return_messages=True)

# # Streamlit UI
# st.title("PDF-based Q&A Chatbot")

# query = st.text_input("Ask a question about the PDF")

# def is_response_complete(response):
#     # Simple heuristic to check if the response seems complete
#     complete_markers = [".", "!", "?"]
#     return any(response.strip().endswith(marker) for marker in complete_markers)

# if query:
#     response = retrieval_chain.invoke({"input": query})
#     answer = response["answer"]
#     answer_marker = "Answer:"
#     start_index = answer.find(answer_marker)

#     if start_index != -1:
#         generated_output = answer[start_index + len(answer_marker):].strip()
#         formatted_output = "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())

#         # Check if the response is complete
#         if not is_response_complete(formatted_output):
#             st.write("The response might be incomplete. Consider asking a follow-up question.")
#         st.write(formatted_output)
#     else:
#         formatted_output = answer.strip()
#         st.write("Answer marker not found. Here is the raw response:")
#         st.write(formatted_output)

#     memory.save_context({"input": query}, {"output": formatted_output})



