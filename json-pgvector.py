from langchain_community.llms import LlamaCpp, Ollama 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from difflib import SequenceMatcher
from langchain.chains import LLMChain
import time
import asyncio
from langchain_community.vectorstores import PGVector


class RAGChatbot:
    def __init__(self):        
        self.vectorstore = None
        self.rag_chain = None

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