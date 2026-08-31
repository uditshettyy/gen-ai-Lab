from langchain_openai import OpenAI, OpenAIEmbeddings
from dotenv import  load_dotenv
load_dotenv()
embeddings=OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=64
)
texts=[
    "Hello this is Akarsh Vyas",
    "Hello youur name is Youtube",
    "And you all are very beautiful"
]
vector=embeddings.embed_documents(texts)
print(vector)