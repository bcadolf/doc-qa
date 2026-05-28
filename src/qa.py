# sends retrieved context + question to LLM 

from openai import OpenAI

SERVER_URL = "http://localhost:8080/v1"
MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

client = OpenAI(
    base_url=SERVER_URL,
    api_key="local-model"
)


def answer_question(question: str, context_chunks: list[str]) -> str:
    """Inject chunks into prompt and send to LLM for answer"""
    context = "\n\n---\n\n".join(context_chunks)


    prompt = f"""You are a helpful assistant. Answer the user's question using only the context provided below. If the answer is not in the context, say so clearly reather than guessing.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content