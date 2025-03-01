from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

app = FastAPI()

# Initialisation des clients
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "rag-chatbot"
    if index_name not in pc.list_indexes().names():
        # Création de l'index s'il n'existe pas
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Région AWS valide
            )
        )
        print(f"Index '{index_name}' créé avec succès!")

    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

except Exception as e:
    print(f"Erreur d'initialisation : {str(e)}")
    raise

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(question: Question):
    try:
        results = search_docs(question.query)
        context = " ".join([doc["metadata"]["text"] for doc in results])
        response = llm(f"Utilise ces infos : {context}. Réponds à : {question.query}")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def search_docs(query: str):
    vector = embeddings.embed_query(query)
    results = index.query(vector, top_k=3, include_metadata=True)
    return results["matches"]

@app.get("/add-test-document")
async def add_test_document():
    try:
        # Créer l'embedding directement
        doc_text = "Ceci est un test"
        doc_vector = embeddings.embed_query(doc_text)

        # Utiliser l'API Pinecone directement sans passer par LangChain
        index.upsert(
            vectors=[
                {
                    "id": "test-doc-1",
                    "values": doc_vector,
                    "metadata": {"text": doc_text}
                }
            ],
            namespace="test"
        )

        return {"message": "Document test ajouté avec succès !"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
