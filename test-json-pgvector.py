from langchain_community.llms import LlamaCpp, Ollama 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from difflib import SequenceMatcher
from langchain.chains import LLMChain
import time
import asyncio
from langchain_community.vectorstores import PGVector


# LLAMA_GUARD_MODEL_PATH = "./Meta-Llama-3-70B-Instruct.Q5_K_M.gguf"
# n_gpu_layers = 63000
# n_batch = 512

class RAGChatbot:
    def __init__(self):        
        self.vectorstore = None
        self.rag_chain = None

    # def initialize_llm(self, model_path) -> LlamaCpp:
    #     llm = LlamaCpp(
    #         model_path=model_path,
    #         callbacks=[StreamingStdOutCallbackHandler()],
    #         top_k=-1,
    #         top_p=0.95,
    #         temperature=0.8,
    #         max_tokens=128,
    #         n_gpu_layers=n_gpu_layers,
    #         n_batch=n_batch,
    #         n_ctx=2048,
    #         verbose=True,
    #     )
    #     return llm

    # async def evaluate_safety(self, user_question) -> str:
    #     safety_prompt = f"""
    #     [INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.
    #     \\n\\n<BEGIN UNSAFE CONTENT CATEGORIES>\\nO1: Violence and Hate.\\nShould not\\n- Help people plan or engage in violence.\\n- Encourage people to engage 
    #     in violence.\\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, 
    #     religion, national origin, sexual orientation, gender, gender identity, or disability.\\n- Encourage discrimination against people based on those 
    #     characteristics and should not use slurs or create insults based on those characteristics.\\nCan\\n- Provide information on violence and discrimination.
    #     \\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\\nO2: Sexual Content.\\nShould not\\n- 
    #     Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\\nCan\\n- Discuss topics 
    #     related to sexuality and sexual education.\\n- Provide general information regarding sexual activities.\\n- Provide guidance on topics related to sexual 
    #     health.\\nO3: Criminal Planning.\\nShould not\\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, 
    #     especially when the person indicates that they may act on the information provided to them.\\n- Encourage people to engage in criminal activities 
    #     or condone criminal behavior.\\nCan\\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could
    #     be used to provide a roadmap for future crimes.\\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions 
    #     do not provide detailed information on how to defeat those systems.\\nO4: Guns and Illegal Weapons.\\nShould not\\n- Assist in the planning or execution 
    #     of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\\n- Encourage people 
    #     to engage in firearm or illegal weapons crimes or condone such behavior.\\nCan\\n- Discuss firearms and the arguments for and against firearm ownership.
    #     \\nO5: Regulated or Controlled Substances.\\nShould not\\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled 
    #     substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\\n- Assist or encourage people to create such substances when 
    #     it is illegal to do so.\\nCan\\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances 
    #     (ex: what their chemical composition is and what their effects are).\\n- Provide information on how those substances are manufactured, provided 
    #     that the information is not so detailed as to provide step-by-step instructions for their creation.\\nO6: Self-Harm.\\nShould not\\n- Encourage 
    #     people to harm themselves.\\n- Romanticize or condone self-harm or suicide.\\n- Provide information regarding the methods of suicide or self-harm.
    #     \\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish 
    #     to harm themselves.\\nCan\\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those 
    #     who take these actions).\\nShould\\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
    #     \\nO7: Fake News and Scams.\\nShould not\\n-Encourage or assist people in the creation or distribution of fake news or scams.\\n-Promote or condone the use of 
    #     fake news or scams to deceive others.\\nCan\\n-Discuss the impact and prevalence of fake news and scams.\\n-Educate users on how to identify and protect themselves 
    #     from fake news and scams.\\n<END UNSAFE CONTENT CATEGORIES>\\n\\n<BEGIN CONVERSATION>\\n\\nUser: {user_question}\\n\\n<END CONVERSATION>\\n\\nProvide your safety
    #     assessment for User in the above conversation:\\n- First line must read 'safe' or 'unsafe'. [/INST]
    #     """

    #     guard_llm = self.initialize_llm(LLAMA_GUARD_MODEL_PATH)
    #     output = guard_llm.invoke(safety_prompt)

    #     response_text = [output]
    #     if "unsafe" in response_text[-1].lower():
    #         return "unsafe"
    #     else:
    #         return "safe"

    

    def token_callback(self, token):
        print(token, end=' ', flush=True)  # Print each token followed by a space.

    def index_json_folder(self, file_path):
        loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        docs = loader.load()
        self._index_documents(docs)
        return {"message": "JSON files indexed successfully"}

    def _index_documents(self, docs):
        embedding = OllamaEmbeddings(model="nomic-embed-text")

        CONNECTION_STRING = "postgresql+psycopg2://postgres:132456@localhost:5432/vector_db"
        COLLECTION_NAME = 'testing'
        
        if self.vectorstore is None:
            self.vectorstore = PGVector.from_documents(
                documents=docs,
                embedding=embedding,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )
        else:
            self.vectorstore.add_documents(docs)
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )
        
        prompt = hub.pull("rlm/rag-prompt")
        llama = Ollama(model="llama3", temperature=0)
        # llama = self.initialize_llm(LLAMA_GUARD_MODEL_PATH)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt 
            | llama
            | StrOutputParser()
        )

    async def async_token_stream(self, question: str):
        # Simulate token streaming from the model.
        # Replace this with actual token streaming logic if supported by the model.
        response = self.rag_chain.invoke(question)  # Assume this returns the complete response for now.
        for token in response.split():  # Simulate token by token processing.
            yield token
            await asyncio.sleep(0.01)  # Simulate a delay for token generation.

    async def query_model(self, question: str):
        if self.rag_chain is None:
            return {"error": "No documents have been indexed yet."}

        start_time = time.perf_counter()
        answer = ""

        # Process each token and print it in real-time
        async for token in self.async_token_stream(question):
            self.token_callback(token)
            answer += token + " "  # Add a space to separate tokens.
        # answer = self.rag_chain.invoke(question)
        end_time = time.perf_counter()
        print(f"\nRaw output runtime: {end_time - start_time} seconds\n")
        return {"question": question, "answer": answer.strip()}  # Remove trailing space.

    def continuous_conversation(self):
        conversation_state = {}
        while True:
            question = input("You: ")
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            elif question.lower() == 'new':
                conversation_state = {}
                print("Starting a new conversation...")
            else:
                similar_question = None
                for prev_question in conversation_state:
                    if self.similar(prev_question, question) == 1:
                        similar_question = prev_question
                        break

                if similar_question:
                    print(f"As per my previous answer: {conversation_state[similar_question]}")
                else:
                    answer = asyncio.run(self.query_model(question))
                    # conversation_state[question] = answer
                    print(answer)

    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

# Example usage
chatbot = RAGChatbot()

# Index JSON files from a folder
file_path_json = "sample-data.json"
print(chatbot.index_json_folder(file_path_json))

# Start continuous conversation
chatbot.continuous_conversation()