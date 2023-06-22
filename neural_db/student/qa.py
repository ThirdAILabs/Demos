from typing import Callable, List


class QA:
    def answer(self, question: str, answers: List[str], on_error: Callable) -> str:
        raise NotImplementedError()


class T5(QA):
    def __init__(self, **kwargs):
        # from transformers import AutoTokenizer, AutoModelForConditionalGeneration, AutoModelWithLMHead
        # t5_model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
        # self.model = AutoModelWithLMHead.from_pretrained(t5_model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        t5_model_name = "t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def answer(self, question: str, answers: List[str], on_error: Callable):
        to_summarize = f"question: {question} context: {' '.join(answers)}"
        encoded_input = self.tokenizer(
            [to_summarize], return_tensors="pt", max_length=512, truncation=True
        )

        output = self.model.generate(
            input_ids=encoded_input.input_ids,
            attention_mask=encoded_input.attention_mask,
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class OpenAI(QA):
    def __init__(self, key, **kwargs) -> None:
        if not key:
            raise ValueError("OpenAI key required.")

        from langchain.chat_models import ChatOpenAI
        from paperqa.qaprompts import make_chain, qa_prompt

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=key
        )
        self.chain = make_chain(prompt=qa_prompt, llm=llm)

    def answer(self, question: str, answers: List[str], on_error: Callable) -> str:
        return self.chain.run(
            question=question, context_str=" ".join(answers), length="abt 100 words"
        )


class Dolly(QA):
    def __init__(self, **kwargs) -> None:
        from langchain import LLMChain, PromptTemplate
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline

        generate_text = pipeline(
            model="databricks/dolly-v2-3b",
            trust_remote_code=True,
            device_map="auto",
            return_full_text=True,
        )
        llm = HuggingFacePipeline(pipeline=generate_text)
        prompt_with_context = PromptTemplate(
            input_variables=["instruction", "context"],
            template="{instruction}\n\nInput:\n{context}",
        )
        self.chain = LLMChain(llm=llm, prompt=prompt_with_context)

    def answer(self, question: str, answers: List[str], on_error: Callable) -> str:
        return self.chain.predict(
            instruction="answer from context: " + question, context=" ".join(answers)
        ).lstrip()


class UDTEmbedding(QA):
    def __init__(self, get_model, get_query_col, **kwargs) -> None:
        self.get_model = get_model
        self.get_query_col = get_query_col

    def answer(self, question: str, answers: List[str], on_error: Callable) -> str:
        # ignore question
        from parsing_utils import summarize

        summaries = [
            summarize.summarize(
                a, self.get_model(), query_col=self.get_query_col()
            ).strip()
            for a in answers
        ]
        return " ".join(
            [
                sent.strip()
                for s in summaries
                for sent in summarize.nlkt_sent_tokenize(s)
            ]
        )


class ContextArgs:
    def __init__(self, **kwargs):
        # Set default context arguments
        setattr(self, "num_references", 1)

        # Set input context args
        for key in kwargs:
            setattr(self, key, kwargs[key])


OPENAI_CG_MODULE = "Open AI for Content Generation"
OPENAI_QA_MODULE = "Open AI for Question Answering"
DOLLY_MODULE = "Dolly"
T5_MODULE = "T5"
UDTEMB = "UDT Embedding"
NONE = "None"

# Map from answering module to tuple of QA class and ContextArgs class
# Every ContextArg class defines "num_references" = 1 by default, which can be overridden.
# Currently, the only other supported arg is "chunk_radius"
modules = {
    T5_MODULE: (T5, ContextArgs()),
    OPENAI_CG_MODULE: (OpenAI, ContextArgs(chunk_radius=3, num_references=5)),
    OPENAI_QA_MODULE: (OpenAI, ContextArgs()),
    # DOLLY_MODULE: (Dolly, {}),
    UDTEMB: (UDTEmbedding, ContextArgs()),
    NONE: None,
}
