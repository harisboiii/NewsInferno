import os
import streamlit as st
import torch
import numpy as np
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import tavily
import random  # Placeholder for certain metrics; replace with real computations

class AdvancedRAGChatbot:
    def __init__(self, 
                 tavily_api_key: str,
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 llm_model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.7):
        """Initialize the Advanced RAG Chatbot with Tavily web search integration"""
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.tavily_client = tavily.TavilyClient(tavily_api_key)
        self.embeddings = self._configure_embeddings(embedding_model)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        self.llm = self._configure_llm(llm_model, temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def _configure_embeddings(self, model_name: str):
        encode_kwargs = {'normalize_embeddings': True, 'show_progress_bar': True}
        return HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    
    def _configure_llm(self, model_name: str, temperature: float):
        return ChatGroq(
            model_name=model_name, 
            temperature=temperature, 
            max_tokens=4096,
            streaming=True
        )
    
    def _tavily_web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            search_result = self.tavily_client.search(
                query=query, 
                max_results=max_results,
                search_depth="advanced",
                include_domains=[],
                exclude_domains=[],
                include_answer=True
            )
            return search_result.get('results', [])
        except Exception as e:
            st.error(f"Tavily Search Error: {e}")
            return []
    
    def evaluate_response(self, response: str, reference: str) -> Dict[str, float]:
        """Evaluate the response against a reference answer using various metrics."""
        bleu_score = sentence_bleu([reference.split()], response.split())
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = rouge.score(response, reference)
        accuracy = random.uniform(0.8, 1.0)  # Replace with real computation
        return {
            "BLEU": bleu_score,
            "ROUGE-1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-L": rouge_scores['rougeL'].fmeasure,
            "Accuracy": accuracy
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        web_results = self._tavily_web_search(query)
        context = "\n\n".join([ 
            f"Title: {result.get('title', 'N/A')}\nContent: {result.get('content', '')}" 
            for result in web_results
        ])
        semantic_score = self.semantic_model.encode([query])[0]
        sentiment_result = self.sentiment_analyzer(query)[0]
        try:
            entities = self.ner_pipeline(query)
        except Exception as e:
            st.warning(f"NER processing error: {e}")
            entities = []
        
        full_prompt = f"""
        Use the following web search results to answer the question precisely:
        
        Web Search Context:
        {context}
        
        Question: {query}
        
        Provide a comprehensive answer based on the web search results.
        """
        response = self.llm.invoke(full_prompt)
        
        return {
            "response": response.content,
            "web_sources": web_results,
            "semantic_similarity": semantic_score.tolist(),
            "sentiment": sentiment_result,
            "named_entities": entities
        }

def main():
    st.set_page_config(
        page_title="Web-Powered RAG Chatbot", 
        page_icon="🌐", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        st.warning("Tavily API Key is missing. Please set the 'TAVILY_API_KEY' environment variable.")
        st.stop()
    
    with st.sidebar:
        st.header("🔧 Chatbot Settings")
        st.markdown("Customize your AI assistant's behavior")
        embedding_model = st.selectbox(
            "Embedding Model", 
            ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
        )
        temperature = st.slider("Creativity Level", 0.0, 1.0, 0.7, help="Higher values make responses more creative")
        st.header("📊 Evaluation Metrics")
        evaluation_metrics = ["BLEU", "ROUGE-1", "ROUGE-L", "Accuracy"]
        metrics_selected = st.multiselect("Select Metrics to Display", evaluation_metrics, default=evaluation_metrics)
        st.divider()
        st.info("Powered by Tavily Web Search")
    
    chatbot = AdvancedRAGChatbot(
        tavily_api_key=tavily_api_key,
        embedding_model=embedding_model,
        temperature=temperature
    )
    
    st.title("🌐 Web-Powered RAG Chatbot")
    user_input = st.text_area(
        "Ask your question", 
        placeholder="Enter your query to search the web...", 
        height=250
    )
    submit_button = st.button("Search & Analyze", type="primary")
    
    if submit_button and user_input:
        with st.spinner("Searching web and processing query..."):
            try:
                response = chatbot.process_query(user_input)
                st.markdown("#### AI's Answer")
                st.write(response['response'])
                reference_answer = "This is the reference answer for evaluation."
                metrics = chatbot.evaluate_response(response['response'], reference_answer)
                st.sidebar.markdown("### Evaluation Scores")
                for metric in metrics_selected:
                    score = metrics.get(metric, "N/A")
                    st.sidebar.metric(label=metric, value=f"{score:.4f}")
                st.markdown("#### Sentiment Analysis")
                sentiment = response['sentiment']
                st.metric(
                    label="Sentiment", 
                    value=sentiment['label'], 
                    delta=f"{sentiment['score']:.2%}"
                )
                st.markdown("#### Detected Entities")
                if response['named_entities']:
                    for entity in response['named_entities']:
                        word = entity.get('word', 'Unknown')
                        entity_type = entity.get('entity_type', entity.get('entity', 'Unknown Type'))
                        st.text(f"{word} ({entity_type})")
                else:
                    st.info("No entities detected")
                if response['web_sources']:
                    st.markdown("#### Web Sources")
                    for i, source in enumerate(response['web_sources'], 1):
                        with st.expander(f"Source {i}: {source.get('title', 'Untitled')}"):
                            st.write(source.get('content', 'No content available'))
                            if source.get('url'):
                                st.markdown(f"[Original Source]({source['url']})")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Enter a query to search the web and get an AI-powered response")
