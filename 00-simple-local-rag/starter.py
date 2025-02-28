import os.path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Set up a free local embedding model
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Set up Ollama as the LLM
llm = Ollama(model="llama3.2:3b-instruct-fp16", request_timeout=120.0)
Settings.llm = llm

# Print current directory and check data directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index
query_engine = index.as_query_engine()

# Print information about the components
print("\n--- System Information ---")
print(f"Embedding model: {embed_model.model_name}")
print(f"LLM: {llm.model}")
print(f"Index type: {type(index).__name__}")
print(f"Number of documents: {len(index.docstore.docs)}")
print(f"Query engine type: {type(query_engine).__name__}")
print(f"Storage location: {os.path.abspath(PERSIST_DIR)}")
print("-------------------------\n")

response = query_engine.query("Summarize the document in 2 sentences")
print(response)