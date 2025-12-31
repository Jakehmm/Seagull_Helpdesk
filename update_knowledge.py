# !!! Need to do: ！！！
# As of now, none


import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import datetime
import requests
from typing import List
import streamlit as st

def get_embeddings_with_requests(texts: List[str], model: str = "Qwen/Qwen3-Embedding-8B") -> List[List[float]]:
    """
    Using requests to get embeddings from SiliconFlow API.
    """
    url = "https://api.siliconflow.cn/v1/embeddings"
    api_key = st.secrets["siliconflow_embedding_api"]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_embeddings = []
    
    # Get embeddings in batches to avoid request size limits
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Creating embeddings for batch {i//batch_size + 1} ({len(batch_texts)} chunks)...")
        
        payload = {
            "model": model,
            "input": batch_texts,
            "dimensions": 1024
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, 60))
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from the response, and add it to the all_embeddings list
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
            print("Zero vectors will be used as placeholders for this batch.")
            # Failed batch uses zero vectors as placeholders
            zero_vector = [0.0] * 1024
            all_embeddings.extend([zero_vector] * len(batch_texts))
    
    return all_embeddings


# Update the knowledge base (the .faiss file), 
# as well as creating a simple .txt to show the version info.
def update_knowledge():

    print("Updating the knowledge base, please wait for a while...")
    
    # Make sure these two directories exist since /knowledge_base is for the source files,
    # while /knowledge_loaded is for storing the knowledge file.
    os.makedirs("knowledge_base", exist_ok=True)
    os.makedirs("knowledge_to_be_loaded", exist_ok=True)
    
    # Variable to store all loaded .txt files
    all_docs = []
    # Get the files in the knowledge_base directory
    for file in os.listdir("knowledge_base"):
        file_path = os.path.join("knowledge_base", file)
        # Load .txt files only; skip other formats
        try:
            if file.endswith(".txt"):
                loader = TextLoader(file_path, autodetect_encoding = True)
                docs = loader.load()
            else:
                print(f"Skipped an unsupported file: {file}")
                print("Only .txt files are supported currently.")
                continue
            
            print(f"File loaded: {file} ({len(docs)} pages)")
            all_docs.extend(docs)
        except Exception as e:
            print(f"An error has occured while handling {file}: {str(e)}")
    
    if not all_docs:
        print("No files were found in the knowledge_base directory.")
        return
    
    # Splitting the text into chunks with 8000 characters each,
    # and an overlap of 1200 characters between two neighboring chunks.
    # Separate at \n\n first; if still too long, separate at \n, Chinese and English punctuation marks, etc.
    separators = [
        "\n\n",          # Double new line
        "\n",            # Single new line
        "。", "！", "？", # Chinese sentence ending punctuation
        ". ", "! ", "? ", # English sentence ending punctuation (with a space after it)
        "；", "; ",      # Semicolon
        "，", ", ",      # Comma
        "、",            # Chinese enumeration comma
        "：", ": ",      # Colon
        "(", ")", "[", "]", "{", "}", # Parentheses
        "\"", "'", "`",  # Quotation marks
        " ",             # Space
        "",              # No separator
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1200,
        length_function=len,
        separators=separators,
        keep_separator=True
    )
    text_chunks = text_splitter.split_documents(all_docs)
    print(f"Successfully split the files into a total of {len(text_chunks)} chunks.")
    
    # Get the texts and metadatas from the text chunks
    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]
    
    # Create a .txt file to store chunked texts for debugging purpose
    with open("chunked_texts/chunked_texts_PCC.txt", "w", encoding="utf-8") as file:
        # Join list items with a newline separator
        file_content = '\n\n----------------------\n\n'.join(texts)
        # Write the combined string to the file
        file.write(file_content)

    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings_list = get_embeddings_with_requests(texts, model="Qwen/Qwen3-Embedding-8B")
    print(f"Successfully created {len(embeddings_list)} embeddings.")

    print("Creating FAISS vector database...")

    if len(texts) != len(embeddings_list):
        print(f"Warning: Text count ({len(texts)}) does not match embedding count ({len(embeddings_list)})!")
        # Truncate to the smaller length so that there won't be errors in FAISS vector database creation
        min_len = min(len(texts), len(embeddings_list))
        texts = texts[:min_len]
        embeddings_list = embeddings_list[:min_len]
        metadatas = metadatas[:min_len]
    
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings_list)),
        embedding=None,  # No need since we provide embeddings directly # type: ignore
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    # For testing purpose, print the distance strategy
    print(f"Current Distance Strategy: {vector_store.distance_strategy}")
    
    # Storing embeddings into a .faiss file
    vector_store.save_local("knowledge_to_be_loaded")
    
    # Creating a version file
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("knowledge_to_be_loaded/version.txt", "w") as f:
        f.write(version)
    
    print(f"Knowledge base is updated. Current version: {version}")
    print(f"Number of chunks: {len(text_chunks)}")
    print(f"Embedding dimension: {len(embeddings_list[0]) if embeddings_list else 0}")
    
    return version

if __name__ == "__main__":
    update_knowledge()
