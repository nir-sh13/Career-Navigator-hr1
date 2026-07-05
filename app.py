

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

PDF_PATH = os.getenv("PDF_PATH") or input(
    "Enter path to your career/resume PDF: "
).strip()

if not os.path.isfile(PDF_PATH):
    raise FileNotFoundError(f": {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Loaded {len(documents)} page(s), split into {len(chunks)} chunk(s).")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="career_navigator",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


def retrieve_from_pdf(query: str) -> str:
    """Search the user's uploaded career document (resume / career guide)."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."
    return "\n\n".join(doc.page_content for doc in docs)
pdf_tool = Tool(
    name="career_document_search",
    func=retrieve_from_pdf,
    description=(
        "Use this to answer questions about the user's own resume/CV or "
        "uploaded career document (skills, experience, education listed in it). "
        "Input should be a search query string."
    ),
)

web_search = DuckDuckGoSearchRun()
web_search_tool = Tool(
    name="job_market_search",
    func=web_search.run,
    description=(
        "Use this for up-to-date info the resume can't answer: current job "
        "openings, salary ranges, in-demand skills, company info, or industry "
        "trends. Input should be a search query string."
    ),
)


def resume_tips(topic: str) -> str:
    """Static best-practice tips for resume/career-doc writing."""
    tips = {
        "default": (
            "General resume tips: keep it to 1-2 pages, lead bullet points with "
            "action verbs, quantify impact with numbers, tailor keywords to the "
            "job description, and keep formatting clean and ATS-friendly."
        )
    }
    return tips.get(topic.lower(), tips["default"])


resume_tips_tool = Tool(
    name="resume_writing_tips",
    func=resume_tips,
    description=(
        "Use this when the user asks for general advice on how to write, "
        "structure, or improve a resume/CV/cover letter (not specific to "
        "their uploaded document). Input can be any topic string or 'default'."
    ),
)

tools = [pdf_tool, web_search_tool, resume_tips_tool]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Career Navigator assistant. You have access to "
            "tools: one to search the user's own uploaded career document, one "
            "to search the web for live job-market info, and one for general "
            "resume-writing tips. Pick the right tool(s) for each question. "
            "If none of the tools return enough information to answer "
            "confidently, say 'I don't know' rather than guessing.",
        ),
        ("human", "{query}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print("\nCareer Navigator ready. Type 'exit' to quit.\n")
    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue
        result = agent_executor.invoke({"query": user_query})
        print("\n" + result["output"] + "\n")
