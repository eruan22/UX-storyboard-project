# imports
import re
import json
from urllib import response

from urllib import response

from langchain_core.prompts import PromptTemplate
from sympy import re
from langchain_core.output_parsers import StrOutputParser

# BUILD CONTEXT RELEVANCE CHAIN
def build_context_relevance_chain(chat_model):
    context_relevance_parser = StrOutputParser()
    # Context relevance prompt
    CONTEXT_RELEVANCE_PROMPT = PromptTemplate.from_template("""Rate how relevant the retrieved context is for answering the question.

    Question: {question}

    Retrieved Context:
    {context}

    Rate from 1-5:
    5 = All context is highly relevant
    4 = Most context is relevant
    3 = Some relevant, some noise
    2 = Mostly irrelevant
    1 = No relevant context

    Score (return ONLY the number):""")
    # build chain
    context_relevance_chain = CONTEXT_RELEVANCE_PROMPT | chat_model | context_relevance_parser
    return context_relevance_chain

# BUILD FAITHFULNESS CHAIN
def build_faithfulness_chain(chat_model):
    faithfulness_parser = StrOutputParser()
    # Faithfulness prompt
    FAITHFULNESS_PROMPT = PromptTemplate.from_template("""Rate if the answer is grounded in (supported by) the context.

    Context:
    {context}

    Answer:
    {answer}

    Rate from 1-5:
    5 = Completely grounded, all claims supported
    4 = Mostly grounded, minor unsupported details
    3 = Partially grounded
    2 = Mostly ungrounded
    1 = Contradicts context or hallucinated

    Score (return ONLY the number):""")
    # build chain
    faithfulness_chain = FAITHFULNESS_PROMPT | chat_model | faithfulness_parser
    return faithfulness_chain

# BUILD ANSWER RELEVANCE CHAIN
def build_answer_relevance_chain(chat_model):
    answer_relevance_parser = StrOutputParser()
    # Answer relevance prompt
    ANSWER_RELEVANCE_PROMPT = PromptTemplate.from_template("""Rate if the answer addresses the question asked.

    Question: {question}

    Answer: {answer}

    Rate from 1-5:
    5 = Directly and completely answers
    4 = Answers with minor omissions
    3 = Partially answers
    2 = Tangentially related
    1 = Does not address the question

    Score (return ONLY the number):""")
    # build chain
    answer_relevance_chain = ANSWER_RELEVANCE_PROMPT | chat_model | answer_relevance_parser
    return answer_relevance_chain

# PARSE RAG TRIAD OUTPUT
def parse_output(output_str):
    match = re.search(r'\b([1-5])\b', output_str)
    return float(match.group(1)) if match else 0.0


# RUN RAG TRIAD EVALUATION
def run_rag_triad(panels, retrieved_docs, answer, chat_model):
    """
    Run all three RAG triad metrics for a single critic agent run.
    
    Args:
        panels: List of Panel objects (the storyboard)
        retrieved_docs: List of strings retrieved from ChromaDB
        critique: String response from the critic agent
        chat_model: the LLM
    
    Returns:
        dict with context_relevance, faithfulness, answer_relevance scores
    """
    # Build chains
    context_relevance_chain = build_context_relevance_chain(chat_model)
    faithfulness_chain = build_faithfulness_chain(chat_model)
    answer_relevance_chain = build_answer_relevance_chain(chat_model)

    # format answer
    answer = json.dumps([c.dict() for c in answer.critiques], indent=2)

    # invoke the chains
    context_relevance_score = context_relevance_chain.invoke({
        "question": "What are the UX pain points in the storyboard panels?",
        "context": "\n\n".join(retrieved_docs)
    })
    faithfulness_score = faithfulness_chain.invoke({
        "context": "\n\n".join(retrieved_docs),
        "answer": answer
    })
    answer_relevance_score = answer_relevance_chain.invoke({
        "question": "What are the UX pain points in the storyboard panels?",
        "answer": answer
    })

    return {
        "context_relevance": context_relevance_score,
        "faithfulness": faithfulness_score,
        "answer_relevance": answer_relevance_score
    }

# run rag triad 20 times
def eval_prompts(panels, retrieved_docs, answer, chat_model, iterations=20):
    results = []
    for i in range(iterations):
        result = run_rag_triad(panels, retrieved_docs, answer, chat_model)
        results.append(result)
    return results
