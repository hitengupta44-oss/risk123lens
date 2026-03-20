import json
import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_BASE_FILE = "data/knowledge_base.json"

def load_knowledge_base(file_path: str) -> list:
    """Loads the knowledge base from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_chroma_collection(client, name: str, embedding_function):
    """Gets or creates a ChromaDB collection."""
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
    )


def add_knowledge_base_to_collection(collection, knowledge_base: list):
    """Adds the knowledge base to a ChromaDB collection."""
    collection.add(
        ids=[item["id"] for item in knowledge_base],
        documents=[item["content"] for item in knowledge_base],
        metadatas=[
            {k: v for k, v in item.items() if k not in ["id", "content"]}
            for item in knowledge_base
        ],
    )


def query_collection(collection, query: str, n_results: int = 5) -> list:
    """Queries the collection and returns the most relevant documents."""
    return collection.query(query_texts=[query], n_results=n_results)


def format_prompt(query: str, documents: list) -> str:
    """Formats the prompt for the language model."""
    context = "\n".join(documents)
    return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""


def main():
    """Main function to run the chatbot."""
    # Set up ChromaDB
    client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = get_chroma_collection(client, "knowledge_base", embedding_function)

    # Load and add knowledge base to the collection
    knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_FILE)
    add_knowledge_base_to_collection(collection, knowledge_base)

    print("Chatbot is ready! Ask me anything about the knowledge base.")

    while True:
        query = input("> ")
        if query.lower() == "exit":
            break

        # Query the collection
        results = query_collection(collection, query)
        documents = results["documents"][0]

        # Format the prompt
        prompt = format_prompt(query, documents)

        # Get the LLM's response (mocked for now)
        # In a real application, you would send the prompt to your LLM API
        print("--- PROMPT ---")
        print(prompt)
        print("--- END PROMPT ---")
        print("LLM Response: (mocked) I would answer the question based on the context.")

if __name__ == "__main__":
    main()
