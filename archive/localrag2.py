import torch
import ollama
import os
from openai import OpenAI
import json
from pathlib import Path
import logging

from dotenv import load_dotenv

# Start logging
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

embedding_model = 'nomic-embed-text'  # or nomic-embed-text
llm_model = 'llama3.1:8b' # or mistral
llm_model = 'mistral'
top_k = 5 # number of chunks pulled

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

CHUNKS_FOLDER = "chunks_and_metadata"

# List the markdown files to use
DOCUMENTS_DIR = Path("documents")
DOCUMENTS = [
    str(p)
    for p in DOCUMENTS_DIR.glob("*")
    if p.is_file() and not p.name.startswith(".")
]

# if not DOCUMENTS:
#     logger.warning(f"No documents found in {DOCUMENTS_DIR.resolve()}")
# else:
#     logger.info(f"Found {len(DOCUMENTS)} documents in {DOCUMENTS_DIR.resolve()}")
#     for doc in DOCUMENTS:
#         logger.info(f" - {doc}")

def load_chunks_and_metadata(document_filename):
    """Load the text chunks and metadata corresponding to a markdown file."""
    base_name = os.path.splitext(os.path.basename(document_filename))[0]
    chunks_path = os.path.join(CHUNKS_FOLDER, f"chunks_markdown_{base_name}.txt")
    metadata_path = os.path.join(CHUNKS_FOLDER, f"metadata_markdown_{base_name}.json")

    if not os.path.exists(chunks_path):
        print(f"⚠️ Chunks file not found: {chunks_path}")
        return [], []

    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file not found: {metadata_path}")
        return [], []

    with open(chunks_path, 'r', encoding='utf-8') as f_txt:
        vault_content = [line.strip() for line in f_txt if line.strip()]

    with open(metadata_path, 'r', encoding='utf-8') as f_json:
        vault_metadata = json.load(f_json)

    return vault_content, vault_metadata

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k, vault_metadata):
    if not rewritten_input.strip():
        print("⚠️ Empty rewritten query — skipping context retrieval.")
        return []

    if vault_embeddings.nelement() == 0:
        return []

    try:
        response = ollama.embeddings(model=embedding_model, prompt=rewritten_input)
        input_embedding = response.get("embedding", [])
    except Exception as e:
        print(f"⚠️ Embedding generation failed: {e}")
        return []

    if not input_embedding:
        print("⚠️ Ollama returned an empty embedding — skipping similarity search.")
        return []

    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    relevant_context = []
    print("\n" + CYAN + "🔍 Top Retrieved Chunks:" + RESET_COLOR)
    for rank, idx in enumerate(top_indices, start=1):
        chunk_text = vault_content[idx].strip()
        meta = vault_metadata[idx]
        title = meta.get("title", "N/A")
        chunk_index = meta.get("chunk_index", "N/A")

        print(f"\n{YELLOW}Chunk #{rank}{RESET_COLOR}")
        print(f"  Title: {NEON_GREEN}{title}{RESET_COLOR}")
        print(f"  Chunk Index: {NEON_GREEN}{chunk_index}{RESET_COLOR}")
        print(f"  Content:\n{CYAN}{chunk_text}{RESET_COLOR}")

        # relevant_context.append(chunk_text)
        relevant_context.append(merge_text_with_metadata(vault_content[idx], vault_metadata[idx]))


    return relevant_context

def merge_text_with_metadata(chunk_text, meta):
    """
    Combine the chunk text with its metadata into a single string for embedding.
    """
    meta_str = (
        f"Title: {meta.get('title', '')}\n"
        f"Section: {meta.get('section_title', '')}\n"
        f"Source: {meta.get('source', '')}\n"
        f"Chunk #{meta.get('chunk_index', '')} of {meta.get('total_chunks', '')}\n"
        f"Token Count: {meta.get('token_count', '')}, Word Count: {meta.get('word_count', '')}\n"
    )
    # Combine metadata + chunk text
    combined_text = meta_str + "\nContent:\n" + chunk_text
    return combined_text

def rewrite_query(user_input_json, conversation_history, llm_model, retrieved_context=""):
    """
    Rewrites the user query using previous conversation history and retrieved context.
    """
    user_input = json.loads(user_input_json)["Query"]
    
    # Exclude the current user message from context
    previous_history = conversation_history[:-1] if len(conversation_history) > 1 else []
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in previous_history])
    prompt = f"""
Rewrite the latest user query to make it more specific and informative for document retrieval.
You MUST always return a rewritten query, even if no conversation history or previous context exists.
Do NOT answer the query—only rewrite it.

Conversation History:
{context}

Previously Retrieved Context:
{retrieved_context}

Latest user query: [{user_input}]

Rewritten latest user query: 
"""
    # response = client.chat.completions.create(
    #     model=llm_model,
    #     messages=[{"role": "system", "content": prompt}],
    #     max_tokens=2000,
    #     n=1,
    #     temperature=0.1,
    # )

    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": prompt},
                #   {"role": "user", "content": user_input}
                *conversation_history
                  ],
        max_tokens=2000,
        n=1,
        temperature=0.1,
    )

    rewritten_query = response.choices[0].message.content.strip()
    # print(prompt)
    return json.dumps({"Rewritten Query": rewritten_query})


def ollama_chat(user_input, system_message, vault_embeddings, vault_content, llm_model, conversation_history, retrieved_context_history):
    """
    Handles a single chat turn:
    - Appends user input to conversation history
    - Rewrites query using previous conversation and retrieved context
    - Retrieves relevant context from vault
    - Generates assistant response
    """
    # Add raw user input first
    conversation_history.append({"role": "user", "content": user_input})

    # Combine all previously retrieved context for rewriting
    combined_context = "\n".join([ctx for ctx in retrieved_context_history]) if retrieved_context_history else ""

    # Rewrite the query (for 2nd+ user messages)
    if len(conversation_history) > 1:
        query_json = {"Query": user_input}
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, llm_model, combined_context)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        # print(combined_context)
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input

    # Retrieve relevant context from vault
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content, top_k, vault_metadata)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        # print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        retrieved_context_history.append(context_str)  # store for future rewrites
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    # Append retrieved context to the user input for the LLM
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context += "\n\nRelevant Context:\n" + context_str

    # Update the last user message with context
    conversation_history[-1]["content"] = user_input_with_context

    # Prepare messages for assistant response
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        max_tokens=2000,
    )

    assistant_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply

# -----------------------
# MAIN SCRIPT
# -----------------------

print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(base_url='http://localhost:11434/v1', api_key='')

# Load chunks and metadata for all markdown files specified in the script
vault_content = []
vault_metadata = []
for docs in DOCUMENTS:
    print(NEON_GREEN + f"Loading chunks and metadata for {docs}..." + RESET_COLOR)
    content, metadata = load_chunks_and_metadata(docs)
    vault_content.extend(content)
    vault_metadata.extend(metadata)

# Generate embeddings for the vault content
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
# vault_embeddings = []
# for content in vault_content:
#     response = ollama.embeddings(model=embedding_model, prompt=content)
#     vault_embeddings.append(response["embedding"])

vault_embeddings = []
for chunk_text, meta in zip(vault_content, vault_metadata):
    combined_text = merge_text_with_metadata(chunk_text, meta)
    response = ollama.embeddings(model=embedding_model, prompt=combined_text)
    vault_embeddings.append(response["embedding"])

vault_embeddings_tensor = torch.tensor(vault_embeddings)

print("Embeddings tensor created for the vault chunks.")

# Conversation loop
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

print("Starting conversation loop...")
retrieved_context_history = [] # initialise retrieved_context_history
while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break

    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, llm_model, conversation_history, retrieved_context_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
