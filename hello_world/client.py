from langchain.llms import HuggingFaceHub
import os

class LlmClient:
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id, 
        model_kwargs={"temperature": 0.8, "top_k": 50}, 
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    
