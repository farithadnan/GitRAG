import os
import locale
from pyexpat import model
import re
from sre_parse import State
from tempfile import tempdir

from dotenv import load_dotenv

load_dotenv()
# Set the locale to UTF-8
locale.getpreferredencoding = lambda: "UTF-8"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

if GITHUB_TOKEN is None:
    raise ValueError("GITHUB_TOKEN environment variable is not set.")


# Generate a GitHub access token
from getpass import getpass

ACCESS_TOKEN = getpass(GITHUB_TOKEN)

# Prepare the data
from langchain.document_loaders import GitHubRepositoryLoader

loader = GitHubRepositoryLoader(repo=GITHUB_REPO, access_token=ACCESS_TOKEN, include_prs=False, state="all")
docs = loader.load()

# Chunking or splitting the documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chuck_overlap=30)
chucked_docs = splitter.split_documents(docs)


# Create embeddings + retriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Create the vector store
db = FAISS.from_documents(chucked_docs, HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL))

# Set up the retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Load quantized model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = LLM_MODEL
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
tokenizer=AutoTokenizer.from_pretrained(model_name)

# Setup LLM Chains
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

# Combine llm chain with retriever
from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# Compare the results
question = "How do you combine multiple adapters?"

llm_chain.invoke({"context": "", "question": question})
rag_chain.invoke(question)