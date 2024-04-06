from transformers import GPT2Tokenizer
from thirdai import bolt, licensing


# Please register for a free license at https://www.thirdai.com/try-bolt/
licensing.activate("")

MODEL_PATH = ""
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = bolt.GenerativeModel.load(MODEL_PATH)


def generate_response(user_input):
    input_tokens = tokenizer.encode(user_input)
    generated_output = model.generate(
        input_tokens=input_tokens,
        max_predictions=100,
        beam_width=3,
        temperature=1.2,
    )
    return tokenizer.decode(generated_output)


while True:
    user_input = input("Enter your text (or 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    response = generate_response(user_input)
    print("\nGenerated Output:")
    print("------------------")
    print(response)
    print("------------------\n")
