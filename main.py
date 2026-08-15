from dotenv import load_dotenv  
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

llm = ChatOpenAI(
        model = AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        base_url= f'{AZURE_OPENAI_ENDPOINT}/openai/v1',
        api_key = AZURE_OPENAI_API_KEY
    )

embeddings = OpenAIEmbeddings(
        model = AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        base_url= f'{AZURE_OPENAI_ENDPOINT}/openai/v1',
        api_key = AZURE_OPENAI_API_KEY
    )

def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)

def get_pinecone_index(index_name):
    pc = get_pinecone_client()
    return pc.Index(index_name)

def upsert_to_pinecone(index_name, vectors):
    print ("pinecone upsert started")

    pc = get_pinecone_client()
    index = pc.Index(index_name)
    index.upsert(vectors=vectors)

    print ("pinecone upsert completed")

def get_document_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        documents.append(
            {
                'context' : text,
                'metadata':
                    {
                        'source': 'sample.pdf',
                        'page_number': page_number + 1
                    }
            }
        )

    print(f"Pages: {len(documents)}")
    return documents;

def get_chunks_from_documents(documents):
    splitter_option = RecursiveCharacterTextSplitter(
        chunk_size= 800,
        chunk_overlap = 200
    )

    chunks = []

    for document in documents:
        page_chunk = splitter_option.split_text(document['context'])

        for chunk in page_chunk:
            chunks.append(
                {
                    'context': chunk,
                    'metadata': document['metadata']
                }
            )

    print(f"Chunks: {len(chunks)}")
    return chunks

def get_vectors_from_chunks(chunks):
    texts = [chunk['context'] for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    print (f"Vectors: {len(vectors)}")
    return vectors

def make_vectors_for_pinecone(chunks, vectors):
    pinecone_vectors = []
    for i in range(len(chunks)):
        pinecone_vectors.append(
            {
                'id': f"{chunks[i]['metadata']['source']}_{chunks[i]['metadata']['page_number']}_{i}",
                'values': vectors[i],
                'metadata': {
                    'text': chunks[i]['context'],
                    **chunks[i]['metadata']
                }
            }
        )
    return pinecone_vectors

def search_by_vector(query:str):
    index_name = "lively-apple"
    index = get_pinecone_index(index_name)
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    return results

# not working
def search_by_text(query:str):
    index_name = "lively-apple"
    index = get_pinecone_index(index_name)
    results = index.search(
        namespace = '__default__',
        query =
            {
                'inputs': {'text': query},
                'top_k': 3
            }
        )
    return results

def get_context(results):
    contexts = []
    sources = set()

    for match in results['matches']:
        metadata = match.get('metadata',{})

        sources.add(metadata.get('source'))

        contexts.append(f"""
            source: {metadata.get('source')}
            Page: {metadata.get('page_number')}

            {metadata.get('text')}
        """)

    return {
        'context': "\n\n---\n\n".join(contexts),
        'sources': sources
    }

def invoke_llm(question, context):
    prompt = f"""
        You are a helpful assistant.

        Answer the user's question using ONLY the provided context.

        If the answer cannot be found in the context, say:
        "I don't have enough information in the provided documents."

        Context:
        ----------------
        {context}
        ----------------

        Question:
        {question}
    """

    response = llm.invoke(prompt)

    return response

def start():
    pdf_path = "handbook.pdf"
    index_name = "lively-apple"
    documents = get_document_from_pdf(pdf_path)
    chunks = get_chunks_from_documents(documents)
    vectors = get_vectors_from_chunks(chunks)
    pinecone_vectors = make_vectors_for_pinecone(chunks, vectors)
    upsert_to_pinecone(index_name, pinecone_vectors)

def ask_question(question):
    matches = search_by_vector(question)
    contextResult = get_context(matches)
    response = invoke_llm(question,contextResult['context'])
    print (f"{question}\n\n{response.content} \n\nsource:{" , ".join(contextResult['sources'])}")

#start()
#print(search_by_vector("What is the purpose of this handbook?"))
#print('next question')
#print(search_by_text("What is the HR policy in this handbook?"))

ask_question("what is the purpose of this handbook?")