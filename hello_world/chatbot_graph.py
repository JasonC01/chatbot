from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceEndpoint
from crewai import Agent, Task, Crew
from langgraph.graph import END, StateGraph
import os
from typing import TypedDict, Optional

import logging
from templates import answer_question_prompt, history_retriever_prompt
from embeddings import context
from memory import ExtendedConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
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


llm = HuggingFaceEndpoint(
    repo_id=model_id, 
    temperature=0.8,
    top_k=50,
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

history_aware_retriever = create_history_aware_retriever(
    llm, context(), history_retriever_prompt
)

answer_question_chain = create_stuff_documents_chain(llm, answer_question_prompt)

retrieval_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)

conversational_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

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
        result = conversational_chain.invoke(
            {"input": "What is Task Decomposition?"},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"]
        # chain = (
        #     {"question": RunnablePassthrough(), "context": context} | answer_question_prompt | llm
        # )
        # result = chain.invoke(question)
        print(result)
        return {"response": result}
    
    # @chain()
    def validate_answer_node(self, state):
        answer = state.get('response', '')
        context = state.get('context', '')
        validator = Agent(
            role="Validator",
            goal="Validate that everything mentioned in the text: {text}, can be found within the given context: {context}",
            backstory="You are a validator, making sure that there are no false informations in the given text: {text}"
                    "You must make sure the text: {text} is generated only with the given context: {context}, and not any other information"
                    "Return 0 if the text: {text} contains false information outside of context: {context}, and 1 if and only if the text: {text} is generated solely from the given context: {context}",
            llm=llm,
            allow_delegation=False,
            verbose=True
        )
        validate = Task(
            description=(
                "Validate that the text: {text} is generated only with the given context: {context}, and not any other information"
            ),
            expected_output="Return 0 if the text: {text} contains false information outside of context: {context}, and 1 if and only if the text: {text} is generated solely from the given context: {context}",
            agent=validator,
        )
        crew = Crew(
                agents=[validator],
                tasks=[validate],
                verbose=2
            )
        inputs = {"text": answer, "context": context}
        result = crew.kickoff(inputs=inputs)
        return {"validate_response": result}
    
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
        # workflow.add_node("validate_answer_node", self.validate_answer_node)
        workflow.add_node("answer_question_node", self.answer_question_node)

        workflow.set_entry_point("answer_question_node")

        workflow.add_edge("answer_question_node", END)
        # workflow.add_edge("retrieve_context_node", "answer_question_node")
        # workflow.add_edge("retrieve_context_node", "answer_question_node")
        # workflow.add_edge("answer_question_node", "validate_answer_node")
        # workflow.add_conditional_edges("validate_answer_node", self.should_remove_content_node, {
        #     "continue": "remove_irrelevant_response_node",
        #     "end": END
        # })
        # workflow.add_edge("remove_irrelevant_response_node", END)

        app = workflow.compile()
        return app.invoke({"question": "who are you"})

chatbot = ChatbotGraph()
print(chatbot.run())
