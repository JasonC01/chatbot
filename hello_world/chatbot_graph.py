from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceEndpoint
from crewai import Agent, Task, Crew
from langgraph.graph import END, StateGraph
import os
from typing import TypedDict, Optional

import logging
from templates import answer_question_prompt, history_retriever_prompt, validate_response_prompt
from embeddings import context
from memory import ExtendedConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
logger = logging.getLogger()
# device_map = infer_auto_device_map(
#     self._model, max_memory=max_memory, no_split_module_classes=["GPTNeoXLayer"], dtype="float16"
# )  
# quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage = True
# ).cpu()
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )


# disk_offload(model=model, offload_dir="offload")



# tokenizer = AutoTokenizer.from_pretrained(model_id)

# pipe = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=256
# )

# llm = HuggingFacePipeline(pipeline=pipe)
# memory = ConversationBufferWindowMemory(k=3)

# tokenizer = AutoTokenizer.from_pretrained(model_id)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# model_kwargs = {
    
#     "eos_token_id":terminators,

# }

llm = HuggingFaceEndpoint(
    repo_id=model_id, 
    # temperature=0.8,
    top_k=50,
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
    do_sample=True,
    temperature=0.6,
    max_new_tokens= 256,
    top_p=0.9,
    # model_kwargs=model_kwargs
)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



# conversational_chain = RunnableWithMessageHistory(
#     retrieval_with_history_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# validator_chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, validate_response_prompt))
# validator_conversational_chain = RunnableWithMessageHistory(
#     validator_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

def get_conversational_chain(prompt_template:ChatPromptTemplate):
    history_aware_retriever = create_history_aware_retriever(
        llm, context(), history_retriever_prompt
    )

    answer_question_chain = create_stuff_documents_chain(llm, prompt_template)

    retrieval_with_history_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)

    conversational_chain = RunnableWithMessageHistory(
        retrieval_with_history_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_chain

class GraphState(TypedDict):
    question: Optional[str] = None
    context: Optional[any] = None
    response: Optional[str] = None
    validate_response: Optional[int] = 1

class ChatbotGraph:
    load_dotenv()
    def answer_question_node(self, state):
        
        question = state.get('question', '')
        print(f"Question: {question}")
        conversational_chain = get_conversational_chain(answer_question_prompt)
        result = conversational_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"]
        # chain = (
        #     {"question": RunnablePassthrough(), "context": context} | answer_question_prompt | llm
        # )
        # result = chain.invoke(question)
        print(result)
        response = result.split(":")
        trimmed_response = response[1].strip()
        print(trimmed_response)
        return {"response": trimmed_response}
    
    # @chain()
    def validate_answer_node(self, state):
        response = state.get('response', '')
        question = state.get('question', '')

        validator_conversational_chain = get_conversational_chain(validate_response_prompt)

        result = validator_conversational_chain.invoke({"input": question, "response": response}, config={
                "configurable": {"session_id": "abc123"}
            },)["answer"]
        
        response = result.split(":")
        trimmed_response = response[1].strip().strip("\n")

        print(trimmed_response)

        return {"validate_response": trimmed_response}
    
    # @chain()
    def remove_irrelevant_response_node(self, state):
        response = state.get('response', '')
        context = state.get('context', '')
        content_parser = Agent(
            role="Content Parser",
            goal="Remove any information in the text: {text}, that is not in the given context: {context}",
            backstory="You are supposed to remove any content within the text: {text} that cannot be identified in the given context: {context}"
                    "If the content of the text is insignificant after the removal, say that you don't know the answer"
                    "Return the output text after removal if and only if it contains significant information, and all information can be identified in the given context: {context}",           
            llm=llm,
            allow_delegation=False,
            verbose=True
        )
        remove = Task(
            description=(
                "Use the given context: {context}, and remove any content within the text: {text}, that cannot be identified in the context"
                "Parse the output in well structured sentences"
            ),
            expected_output="Well formatted answer, without any information not identifiable in the context: {context}",
            agent=content_parser,
        )
        crew = Crew(
                agents=[content_parser],
                tasks=[remove],
                verbose=2
            )
        inputs = {"context": context, "text": response}
        result = crew.kickoff(inputs=inputs)
        return {"reponse": result}

    # @chain()
    def should_remove_content_node(self, state):
        validate_response = state.get("validate_response", '')
        if validate_response == 1:
            return "end"
        else:
            return "continue"
        
    # @chain()
    def retrieve_context_node(self, state):
        question = state.get('question')
        docsearch = context(question=question)
        return {"context": docsearch}
    
    def run(self):
        workflow = StateGraph(GraphState)
        # workflow.add_node("should_remove_content_node", self.should_remove_content_node)
        # workflow.add_node("retrieve_context_node", self.retrieve_context_node)
        # workflow.add_node("remove_irrelevant_response_node", self.remove_irrelevant_response_node)
        workflow.add_node("validate_answer_node", self.validate_answer_node)
        workflow.add_node("answer_question_node", self.answer_question_node)

        workflow.set_entry_point("answer_question_node")

        workflow.add_edge("validate_answer_node", END)
        # workflow.add_edge("retrieve_context_node", "answer_question_node")
        # workflow.add_edge("retrieve_context_node", "answer_question_node")
        workflow.add_edge("answer_question_node", "validate_answer_node")
        # workflow.add_conditional_edges("validate_answer_node", self.should_remove_content_node, {
        #     "continue": "remove_irrelevant_response_node",
        #     "end": END
        # })
        # workflow.add_edge("remove_irrelevant_response_node", END)

        app = workflow.compile()
        return app.invoke({"question": "what is your favourite color?"})

chatbot = ChatbotGraph()
print(chatbot.run())
