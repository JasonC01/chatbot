from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
import langchain.schema.runnable
from langchain.schema.output_parser import StrOutputParser
from crewai import Agent, Task, Crew
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import chain
import os
from typing import Dict, TypedDict, Optional
import logging
from templates import answer_question_prompt, answer_question_task_prompt
# from langchain_core.vectorstores import 
from embeddings import getEmbeddings



llm_prompt = ({"context": getEmbeddings().as_retriever(), "question": RunnablePassthrough()} | answer_question_prompt).invoke("who are you")
print(llm_prompt)