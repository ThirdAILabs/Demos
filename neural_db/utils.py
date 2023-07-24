import openai

def ask_gpt(messages):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=messages, 
    )

    response_content = response.choices[0].message.content
    return response_content


def get_capabilities(db, task, context, top_k=1):
    initial_message = {"role": "user", "content": task + "\n" + context}

    response = ask_gpt([initial_message])
    broad_action_items_message = {"role": "assistant", "content": response}

    generate_queries_str = f"""
    This request comes from a company with proprietary data. Suppose you had a semantic
    search system that contains a variety of information from its website, including
    products, applications, research, design support, community, etc. Come up with a
    list of 5-10 short queries (10-20 words) that would collect all of the necessary
    auxiliary information to complete the above task. You should return ONLY these
    queries such that I might be able to split the resulting string with string.split('\n')
    in python. Do not include numbers as prefixes and do not add quotes to each query.
    """

    generate_queries_message = {"role": "user", "content": generate_queries_str}

    response = ask_gpt([initial_message, broad_action_items_message, generate_queries_message])

    queries = response.split("\n")

    source_information = ""
    for query in queries:
        search_results = db.search(
        query=query,
        top_k=top_k,
        on_error=lambda error_msg: print(f"Error! {error_msg}"))
        source_information += search_results[0].text + "\n"
    
    gpts_queries_message = {"role": "assistant", "content": response}

    refine_action_items_str = """
    Here are the answers to all of your questions. Revise the set of capabilities given this auxiliary information.\n\n
    """
    refine_action_items_str += source_information
    refine_action_items_message = {"role": "user", "content": refine_action_items_str}

    response = ask_gpt([initial_message, broad_action_items_message, generate_queries_message, gpts_queries_message, refine_action_items_message])
    
    return response