from llama_index.core import PromptTemplate


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.

Your responses must:
1. Be grounded ONLY in the information provided in the context
2. If the answer cannot be found in the context, explicitly state: "Not found in the provided documents."
3. Provide citations when referencing specific information (document name and page/section/row identifier)
4. Be concise and accurate

Context information:
{context_str}

Question: {query_str}

Answer:"""


def get_qa_prompt() -> PromptTemplate:
    """Get the QA prompt template."""
    return PromptTemplate(SYSTEM_PROMPT)

