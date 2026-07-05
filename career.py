from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)




loader = PyPDFLoader("CAREER_200_pages.pdf")
docs = loader.load()

loader = PyPDFLoader("Career_Guide_Multiple_Fields.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="agent_rag"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



@tool
def calculator(expression: str) -> str:
    """Evaluate simple math expressions"""
    return str(eval(expression))

@tool
def search_docs(query: str) -> str:
    """Search PDF knowledge base"""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

tools = [calculator, search_docs]

llm_with_tools = llm.bind_tools(tools)

tool_map = {
    "calculator": calculator,
    "search_docs": search_docs
}


messages = [
    SystemMessage(content="""
You are a Career AI Agent.

You can:
- Use documents 
- Do calculations
- Give career guidance

Always be helpful and clear.
""")
]


print("Career AI Agent Started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    # Step 1: LLM decides tool or answer
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Step 2: If tool is needed
    if response.tool_calls:
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool_result = tool_map[tool_name].invoke(tool_args)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # Step 3: Final response after tool execution
        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)

        print("Agent:", final_response.content)

    else:
        print("Agent:", response.content)