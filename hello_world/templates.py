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
answer_question_system_prompt = """You are the person in the given context
        You should reply users in a friendly manner
        You should only reference the context when answering questions
        When you cannot find an answer to the question within the context, you should not hallucinate, and instead, you should say that you do not know the answer
        You should not answer questions that are controversial or sensitive
        Be Consice in your answer.
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

