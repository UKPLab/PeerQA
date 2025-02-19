SYSTEM_PROMPT = "You are a helpful scientific research assistant. Directly answer the question and keep your answers short and concise."

PROMPTS = {
    "answerability-full-text": 'Read the following paper and answer the question. If the paper does not answer the question, answer with "No Answer".\n\nQuestion: {question}\n\nPaper: {paper}\n\nAnswer:',
    "answerability-rag": 'Read the following paragraphs of a paper and answer the question. If the paragraphs do not provide any information to answer the question, answer with "No Answer".\n\nQuestion: {question}\n\nParagraphs:\n{paragraphs}\n\nAnswer:',
    "full-text": "Read the following paper and answer the question.\n\nQuestion: {question}\n\nPaper: {paper}\n\nAnswer:",
    "rag": "Read the following paragraphs of a paper and answer the question.\n\nQuestion: {question}\n\nParagraphs:\n{paragraphs}\n\nAnswer:",
}
