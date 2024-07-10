from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import LlamaCppEmbeddings
from difflib import SequenceMatcher
from langchain.chains import LLMChain
import os
import json
import time
import asyncio

LLAMA_GUARD_MODEL_PATH = "./Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

n_gpu_layers = 4000  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
def initialize_llm(model_path) -> LlamaCpp:
    # Initialize the LlamaCpp instance with local model path
    llm = LlamaCpp(
        model_path=model_path,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.75,
        max_tokens=2000,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        verbose=True,
    )
    return llm

async def evaluate_safety(user_question) -> str:
    safety_prompt = f"""
    [INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.
    \\n\\n<BEGIN UNSAFE CONTENT CATEGORIES>\\nO1: Violence and Hate.\\nShould not\\n- Help people plan or engage in violence.\\n- Encourage people to engage 
    in violence.\\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, 
    religion, national origin, sexual orientation, gender, gender identity, or disability.\\n- Encourage discrimination against people based on those 
    characteristics and should not use slurs or create insults based on those characteristics.\\nCan\\n- Provide information on violence and discrimination.
    \\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\\nO2: Sexual Content.\\nShould not\\n- 
    Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\\nCan\\n- Discuss topics 
    related to sexuality and sexual education.\\n- Provide general information regarding sexual activities.\\n- Provide guidance on topics related to sexual 
    health.\\nO3: Criminal Planning.\\nShould not\\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, 
    especially when the person indicates that they may act on the information provided to them.\\n- Encourage people to engage in criminal activities 
    or condone criminal behavior.\\nCan\\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could
    be used to provide a roadmap for future crimes.\\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions 
    do not provide detailed information on how to defeat those systems.\\nO4: Guns and Illegal Weapons.\\nShould not\\n- Assist in the planning or execution 
    of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\\n- Encourage people 
    to engage in firearm or illegal weapons crimes or condone such behavior.\\nCan\\n- Discuss firearms and the arguments for and against firearm ownership.
    \\nO5: Regulated or Controlled Substances.\\nShould not\\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled 
    substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\\n- Assist or encourage people to create such substances when 
    it is illegal to do so.\\nCan\\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances 
    (ex: what their chemical composition is and what their effects are).\\n- Provide information on how those substances are manufactured, provided 
    that the information is not so detailed as to provide step-by-step instructions for their creation.\\nO6: Self-Harm.\\nShould not\\n- Encourage 
    people to harm themselves.\\n- Romanticize or condone self-harm or suicide.\\n- Provide information regarding the methods of suicide or self-harm.
    \\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish 
    to harm themselves.\\nCan\\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those 
    who take these actions).\\nShould\\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
    \\nO7: Fake News and Scams.\\nShould not\\n-Encourage or assist people in the creation or distribution of fake news or scams.\\n-Promote or condone the use of 
    fake news or scams to deceive others.\\nCan\\n-Discuss the impact and prevalence of fake news and scams.\\n-Educate users on how to identify and protect themselves 
    from fake news and scams.\\n<END UNSAFE CONTENT CATEGORIES>\\n\\n<BEGIN CONVERSATION>\\n\\nUser: {user_question}\\n\\n<END CONVERSATION>\\n\\nProvide your safety
    assessment for User in the above conversation:\\n- First line must read 'safe' or 'unsafe'. [/INST]
    """

    guard_llm = initialize_llm(LLAMA_GUARD_MODEL_PATH)
    output = guard_llm.invoke(safety_prompt)

    # Extract the safety assessment from the output
    response_text = output
    if "unsafe" in response_text.lower():
        return "unsafe"
    return "safe"

def index_pdf(file_path):
    # Initialize the Local Model
    llm =Ollama(
        model="llama3",
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.75,
        verbose=True,
        )

    # Indexing: Load
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Indexing: Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    # Indexing: Store
    # embedding = LlamaCppEmbeddings(model_path="./nomic-embed-text-v1.5.f32.gguf")
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    # Store retriever globally for later use
    global rag_chain
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_chain():
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm 
            | StrOutputParser()
        )

    rag_chain = create_rag_chain()
    
    return {"message": "PDF indexed successfully"}

async def query_model(question: str):
    if 'rag_chain' not in globals():
        return "No documents have been indexed yet."

    if await evaluate_safety(question) == "unsafe":
        answer="Sorry, I cannot answer this question, please try again"
        print(answer)
    else:
        start_time = time.perf_counter()
        answer = rag_chain.invoke(question)
        end_time = time.perf_counter()

        print(f"\nRaw output runtime: {end_time - start_time} seconds\n")
    
    return answer

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Example usage
file_path = "./update-28-covid-19-what-we-know.pdf"
print(index_pdf(file_path)["message"])

# Interactive RAG System
conversation_state = {}
while True:
    question = input("Please enter your question or type 'exit' to end the conversation: ")
    if question.lower() == 'exit':
        conversation_state = {}
        print("Ending the current conversation...")
        break
    elif question.lower() == 'new':
        conversation_state = {}
        print("Starting a new conversation...")
    else:
        # Check if a similar question has been asked before
        similar_question = None
        for prev_question in conversation_state:
            if similar(prev_question, question) > 0.8:
                similar_question = prev_question
                break

        if similar_question:
            print(f"As per my previous answer: {conversation_state[similar_question]}")
        else:
            # Use asyncio.run to run the coroutine and get the result
            answer = asyncio.run(query_model(question))
            conversation_state[question] = answer
            print(answer)

