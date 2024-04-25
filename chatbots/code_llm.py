"""
codellama for python coding
"""

import panel as pn
from langchain.chains import LLMChain
# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

pn.extension()

MODEL_KWARGS = {
    "llama": {
        "model": "TheBloke/CodeFuse-CodeLlama-34B-GGUF",
        "model_file": "codefuse-codellama-34b.Q4_K_S.gguf",
    },

}
llm_chains = {}

TEMPLATE = """<s>[INST] You are good python programmer
user:
{user_input} [/INST] </s>
"""

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    config = {"max_new_tokens": 1024, "temperature": 0.0}

    for model in MODEL_KWARGS:
        if model not in llm_chains:
            instance.placeholder_text = (
                f"Downloading {model}, this may take a few minutes,"
                f"or longer, depending on your internet connection."
            )
            llm = CTransformers(**MODEL_KWARGS[model], config=config)
            prompt = PromptTemplate(template=TEMPLATE, input_variables=["user_input"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            llm_chains[model] = llm_chain
        instance.send(
            await llm_chains[model].apredict(user_input=contents),
            user=model.title(),
            respond=False,
        )


chat_interface = pn.chat.ChatInterface(callback=callback, placeholder_threshold=0.1)
chat_interface.send(
    "Send your msg for generate code!",
    user="System",
    respond=False,
)
chat_interface.servable()