

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import yaml
from custom_logging.logger import get_logger
from langchain.schema import BaseRetriever, Document

logger = get_logger("rag_integration_log")


class DummyRetriever(BaseRetriever):
    """
    Fallback retriever if no FAISS index is found.
    Implements the new _get_relevant_documents methods to avoid deprecation warnings.
    """
    def _get_relevant_documents(self, query: str) -> list[Document]:
        return []
    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return []

class RAGPipelineManager:

    def __init__(self, model_path, index_path=None, device="cpu"):
        logger.info(f"Initializing RAGPipelineManager with model={model_path}, index={index_path}")
        
        # Load your local fine-tuned model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Create pipeline with specified token generation parameters
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            do_sample= True,
            min_new_tokens=1,  
            max_new_tokens=4000,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=2.5 
            )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        logger.info("HuggingFace pipeline for text2text-generation initialized (stateless).")

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if index_path and os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                logger.info("FAISS index loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.vector_store = None
        else:
            self.vector_store = None
            logger.warning("No FAISS index found; using DummyRetriever fallback.")

        if self.vector_store:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        else:
            retriever = DummyRetriever()

        # Build a single-input RetrievalQA chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        logger.info("RetrievalQA chain ready (single-input, stateless).")

    def ask_question(self, combined_text: str) -> str:
        """
        combined_text: A single string containing your prompt.
        
        In this stateless version, the prompt should include only the system instructions
        and the current user query. Previous conversation context is not added.
        
        # If you want to include previous context, you could do something like:
        #    additional_context = "<previous context text>"
        #    combined_text = additional_context + "\n\n" + combined_text
        # (Then the model would see the previous conversation as well.)
        """
        logger.info(f"Asking question with combined_text length: {len(combined_text)}")
        try:
            answer = self.chain.run(combined_text)
            logger.info(f"RAG answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in chain.run(...): {e}")
            raise e

# For testing 
if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_dir = config['training']['output_dir']
    index_path = "faiss_index"  
    rag_manager = RAGPipelineManager(model_path=model_dir, index_path=index_path, device="cpu")
    # Example prompt (stateless): only system prompt + current query
    prompt = "You are a professional medical assistant. Please provide accurate advice.\nUser: I have a headache, what should I do?"
    print("Answer:", rag_manager.ask_question(prompt))
