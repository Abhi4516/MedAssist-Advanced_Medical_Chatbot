# langchain_integration/rag_pipeline.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import yaml
from langchain.schema import BaseRetriever, Document
from custom_logging.logger import get_logger

logger = get_logger("rag_integration_log")

class DummyRetriever(BaseRetriever):
    """
    Fallback retriever if no FAISS index is found.
    Implements the updated `_get_relevant_documents` methods to avoid deprecation warnings.
    """
    def _get_relevant_documents(self, query: str) -> list[Document]:
        return []
    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return []

class RAGPipelineManager:
    """
    A 'strict retrieval only' approach.
    1) We manually retrieve top-k passages from FAISS (or dummy if not found).
    2) We unify those passages + a strong system instruction + the user query into one prompt.
    3) We pass that single prompt to the LLM pipeline (no chain memory).
    """
    def __init__(self, model_path, index_path=None, device="cpu"):
        logger.info(f"Initializing RAGPipelineManager with model={model_path}, index={index_path}")

       
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
      
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
    
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        logger.info("HuggingFace pipeline initialized (strict retrieval approach).")

     
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if index_path and os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.vector_store = None
        else:
            self.vector_store = None
            logger.warning("No FAISS index found; using dummy retriever fallback.")
        
   
        if not self.vector_store:
            self.retriever = DummyRetriever()
        else:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})  

        # This system prompt strongly instructs the LLM to trust the retrieved text over internal memory
        self.system_prompt = (
            "You are a professional medical assistant with access to the following authoritative passages.\n"
            "If there's any conflict between these passages and your memory, always trust these passages.\n"
            "If uncertain, encourage consulting a real doctor.\n\n"
        )

    def ask_question_strict(self, user_query: str) -> str:
        """
        Strict retrieval:
        1) Retrieve docs from FAISS or dummy
        2) Combine them + system prompt + user query into one string
        3) Call pipeline.run(...) with that single string (no chain memory).
        """
        logger.info(f"ask_question_strict called with user query: {user_query}")

        # Step 1: Retrieve top docs
        docs = self.retriever.get_relevant_documents(user_query)  
        # Combine them into a single text block
        retrieved_text = "\n\n".join([doc.page_content for doc in docs])

        # Step 2: Construct a single input prompt
        combined_prompt = (
            f"{self.system_prompt}"
            f"AUTHORITATIVE PASSAGES:\n{retrieved_text}\n\n"
            f"USER QUERY:\n{user_query}\n\n"
            "FINAL ANSWER (ONLY use the passages above if there's a conflict):\n"
        )

       
        try:
            result = self.llm(combined_prompt)
            logger.info(f"Strict retrieval answer: {result}")
            return result
        except Exception as e:
            logger.error(f"Error generating answer in ask_question_strict: {e}")
            raise e
