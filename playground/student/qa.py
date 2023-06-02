class QA:
    def answer(self, question, answers) -> str:
        raise NotImplementedError()


class T5(QA):
    def __init__(self, **kwargs):
        # from transformers import AutoTokenizer, AutoModelForConditionalGeneration, AutoModelWithLMHead
        # t5_model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
        # self.model = AutoModelWithLMHead.from_pretrained(t5_model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        t5_model_name = "t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def answer(self, question, answers):
        to_summarize = f"question: {question} context: {' '.join(answers)}"
        encoded_input = self.tokenizer(
            [to_summarize],
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        output = self.model.generate(
            input_ids = encoded_input.input_ids,
            attention_mask = encoded_input.attention_mask,
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class OpenAI(QA):
    def __init__(self, key, **kwargs) -> None:
        if not key:
            raise ValueError("OpenAI key required.")

        from langchain.chat_models import ChatOpenAI
        from paperqa.qaprompts import qa_prompt, make_chain
        
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo', 
            temperature=0.1, 
            openai_api_key=key
        )
        self.chain = make_chain(prompt=qa_prompt, llm=llm)

    def answer(self, question, answers) -> str:
        return self.chain.run(question=question, context_str=' '.join(answers),length="abt 100 words")
    

class Dolly(QA):
    def __init__(self, **kwargs) -> None:
        from transformers import pipeline
        from langchain.llms import HuggingFacePipeline
        from langchain import PromptTemplate, LLMChain

        generate_text = pipeline(model="databricks/dolly-v2-3b",
                         trust_remote_code=True, device_map="auto", return_full_text=True)
        llm = HuggingFacePipeline(pipeline=generate_text)
        prompt_with_context = PromptTemplate(
            input_variables=["instruction", "context"],
            template="{instruction}\n\nInput:\n{context}"
        )
        self.chain = LLMChain(llm=llm, prompt=prompt_with_context)

    def answer(self, question, answers) -> str:
        return self.chain.predict(
            instruction='answer from context: '+ question, 
            context=' '.join(answers)
        ).lstrip()


class UDTEmbedding(QA):
    def __init__(self, model, **kwargs) -> None:
        self.model = model
    
    def answer(self, question, answers) -> str:
        # ignore question
        from parsing_utils import summarize
        summaries = [summarize.summarize(a, self.model).strip() for a in answers]
        return " ".join([
            sent.strip() 
            for s in summaries 
            for sent in summarize.nlkt_sent_tokenize(s)
        ])
    

OPENAI_MODULE = "Open AI"
DOLLY_MODULE = "Dolly"
T5_MODULE = "T5"
UDTEMB = "UDT Embedding"
NONE = "None"

modules = {
    T5_MODULE: T5,
    OPENAI_MODULE: OpenAI,
    # DOLLY_MODULE: Dolly,
    UDTEMB: UDTEmbedding,
    NONE: lambda **kwargs: None,
}