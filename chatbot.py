from langchain import document_loaders as dl
from langchain import embeddings
from langchain import text_splitter as ts
from langchain import vectorstores as vs
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from operator import itemgetter
import torch
from transformers import pipeline,AutoConfig
from transformers import AutoTokenizer,AutoModelForCausalLM
import re
import time
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import streamlit as st
import asyncio
import shortuuid
from transformers import T5Tokenizer,T5Model
EMBED_MODEL="all-mpnet-base-v2"
INFERENCE_MODEL="t5-small"
import torch.backends.mps


class RAG:
    def __init__(self):
        self.embed_model = EMBED_MODEL
        self.inference_model = INFERENCE_MODEL
        self.chunk_overlap=20
        self.chunk_size = 100
        self.db = chromadb.HttpClient(host="localhost",port=8000,settings=Settings(allow_reset=True,anonymized_telemetry=False))
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = self.embed_model
        )    


    def preprocessing(self,document_path,file_name):
        documents = self.split_doc(document_path,file_name)
        result = self.create_db(documents)
        if result:
            return True
        else:
            return False            

    def split_doc(self,document_path,file_name):
        try:
            self.document_path = document_path
            self.collection_name = file_name[:10]+'_'+shortuuid.uuid()
            loader = dl.PyPDFLoader(self.document_path)
            document = loader.load()
            text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
            document_splitted = text_splitter.split_documents(documents=document)
            return document_splitted
        except Exception as ex:
            st.write("Error occured while splitting document",ex)

    def create_db(self,documents):
        try:
            self.collection = self.db.create_collection(
                 name=self.collection_name,
                 embedding_function=self.embedding_func,
                 metadata={'hnsw:space':'cosine'}    
                )
            self.collection.add(
                documents=[document.page_content for document in documents],
                ids=[f"id{i}" for i in range(len(documents))],
            )
            return True
        except Exception as ex:
            st.write("Error while write to db",ex)
            return False
    
    def query_db(self,query):
        query_results = self.collection.query(query_texts=[query],n_results=1)
        return query_results['documents'][0][0]
   
    def generate(self,question,context):
        #device = "mps" if torch.backends.mps.is_available() else "cpu"
        device="cpu"
        template = f"""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Helpful Answer:"""

        model = AutoModelForCausalLM.from_pretrained(
            "Phi-3-mini-128k-instruct",
            device_map="auto",  # Switches automatically to MPS if available
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("Phi-3-mini-128k-instruct")
        if model.config.eos_token_id is None:
            model.config.eos_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        inputs = tokenizer(template, return_tensors="pt").to(device)

        try:
            # Generate the sequence directly
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                temperature=0.0,
                do_sample=False,
            )

            # Decode the output sequence into human-readable text
            output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            return output_text
        except Exception as ex:
            st.write("Error during inference:", ex)
            return None



        return output[0]['generated_text']
def main():
    rag = RAG()
    st.title("File Q&A with model")
    uploaded_file = st.file_uploader("upload the file",type="pdf")
    save_path=None
    if uploaded_file is not None:
        save_path = f"./{uploaded_file.name}"
        with open(save_path,"wb") as file:
            file.write(uploaded_file.getbuffer())
        result = rag.preprocessing(save_path,uploaded_file.name)
        if result:
            question = st.text_input(
                "Ask something about the article",
                disabled=not uploaded_file)

            if uploaded_file and question:
                st.text("File and question uploaded")
                doc = rag.query_db(question)
                answer = rag.generate(question,doc)
                st.write(answer)

if __name__ == '__main__':
    main()
