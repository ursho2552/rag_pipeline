import os
#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

LLM_MODEL = os.getenv('LLM_MODEL')#, 'mistral')

# Function to get the prompt templates for generating alternative questions and answering based on context
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate five diverse alternative phrasings of the question for retrieving documents:
        Original: {question}
        - Alternative 1:
        - Alternative 2:
        - Alternative 3:
        - Alternative 4:
        - Alternative 5:
        """,  # Clearer instructions
    )

    # Updated answer prompt template
    template = """Answer the question using ONLY the context provided:
    ---
    Context: {context}
    ---
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return QUERY_PROMPT, prompt


def query(input, filename=None):
    if input:
        llm = ChatOllama(model=LLM_MODEL)
        db = get_vector_db()

        # Use filename as a filter if provided
        retriever = db.as_retriever(search_kwargs={"filters": {"filename": filename}}) if filename else db.as_retriever()

        QUERY_PROMPT, prompt = get_prompt()

        # Set up the chain for retrieval and answering
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(input)
        return response

    return None
