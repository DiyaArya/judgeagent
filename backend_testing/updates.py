import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("chatfaq")

new_documents = [
    "take the myvi plan for 4 people which includes mobile devices and data devices."
]

new_metadatas = [
    {"question": "What is the perfect plan for a family of 4"}
]

new_ids = ["test12"]

collection.add(
    documents=new_documents,
    metadatas=new_metadatas,
    ids=new_ids
)