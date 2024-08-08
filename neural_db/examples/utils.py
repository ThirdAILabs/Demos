from openai import OpenAI


def generate_answers(query, references):
    openai_client = OpenAI()
    context = "\n\n".join(references[:3])

    prompt = f"Answer the following question in about 50 words using the context given: \nQuestion : {query} \nContext: {context}"

    messages = [{"role": "user", "content": prompt}]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message.content
