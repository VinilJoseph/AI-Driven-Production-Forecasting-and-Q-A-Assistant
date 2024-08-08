from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import pickle

# Path to the PDF and the vector store save path
pdf_path = r"C:\Users\DELL\datasets\petroleum_book\Petroleum_Production_Engineering_Boyun_G.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text().replace("\n", " ")
    return text

# Function to chunk text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(text)

# Extract and process text
text = extract_text_from_pdf(pdf_path)
texts = chunk_text(text)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_texts(texts, embeddings)

# Save the vector store index
faiss.write_index(vector_store.index, "vector_store.index")

# Save the docstore and index_to_docstore_id
with open("docstore.pkl", "wb") as f:
    pickle.dump(vector_store.docstore, f)

with open("index_to_docstore_id.pkl", "wb") as f:
    pickle.dump(vector_store.index_to_docstore_id, f)

# Save the embeddings
with open("embedding.pkl", "wb") as f:
    pickle.dump(embeddings, f)
