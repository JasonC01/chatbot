from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = """
You are a fortune teller. These Human will ask you a questions about their life. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""
answer_question_system_prompt = """You are the person in the given context.
        You should reply users in a friendly manner and like a human having a conversation.
        You should only reference the context when answering questions.
        When you cannot find an answer to the question within the context, you should not hallucinate, and instead, you should say that you do not know the answer.
        You should not answer questions that are controversial or sensitive.
        You must be Consice in your answer.
        Answer just the question provided and do not make up any other questions.
        Answer the question in one sentence.
        Question: {input}
        Context: {context}
        """

# answer_question_prompt = PromptTemplate(
#   template=answer_question_template, 
#   input_variables=["context", "history", "input"]
# )

answer_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_question_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_retriever_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

history_retriever_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", history_retriever_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

validate_response_system_prompt = """
You are the person in the given context. You should reply users in a friendly manner. The response provided below is the response to the given question. Your task is to make sure that the content within the response can be directly referenced from the context, including details. You should not assume any information when you are validating the response. If the response is valid, return 1, otherwise return 0, and nothing else. Do not add any reasoning or justification or explanation to the response. Do not add further questions or responses, and return only 1 or 0.


Question: {input}
Resonse: {response}
Context: {context}
"""



validate_response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", validate_response_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Validate that the response provided only references the context and not assume any information outside the context."),
    ]
)
# ChatPromptTemplate.from_template(template=validate_response_system_prompt)
# PromptTemplate(input_variables=["input", "response", "context"], template=validate_response_system_prompt)
