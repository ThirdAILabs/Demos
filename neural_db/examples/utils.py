from openai import OpenAI


def query_gpt(prompt, model_name, client):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0
    )
    return response.choices[0].message.content


def generate_answers(query, references):
    openai_client = OpenAI()
    context = "\n\n".join(references[:3])

    prompt = f"Answer the following question in about 50 words using the context given: \nQuestion : {query} \nContext: {context}"

    return query_gpt(prompt, "gpt-3.5-turbo", client=openai_client)
