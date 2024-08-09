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
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    Try to keep your answer concise and within 150 words.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
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
# Answer the following question based only on the provided context.
# Think step by step before providing a detailed answer.
# Try to keep your answer concise and within 150 words.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# {context}

# Question: {input}
# Answer:
# """
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



