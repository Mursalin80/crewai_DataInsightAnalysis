import os

from crewai import LLM
from icecream import ic
from langchain_together import Together

# openai model
llm_chatOpenAi = LLM(
    model="gpt-4",
    temperature=0.5,
    max_tokens=150,
)

# google gemini model
llm_google_gemini = LLM(model="gemini/gemini-1.5-pro-latest", temperature=0.5)
#  anthropic model
llm_anthropic = LLM(model="anthropic/claude-3-sonnet-20240229-v1:0", temperature=0.5)
# groq model
llm_groq_llama_3 = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.5)


# TODO: Removing of error while using in crew agent.
# huggingface model, having error when using in agent while in sample call its work fine
llm_haggingsface = LLM(
    model="huggingface/HuggingFaceH4/zephyr-7b-beta",
    base_url=os.getenv("HUGGINGFACE_BASE_URL"),
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

# TODO: Removing of error while using in crew agent.
# together model for chat, having error when using in agent while in sample call its work fine
llm_together = Together(
    model="meta-llama/Llama-Vision-Free",  # chat model
    base_url=os.getenv("TOGETHER_BASE_URL"),
    api_key=os.getenv("TOGETHER_API_KEY"),
    # model="black-forest-labs/FLUX.1-schnell-Free", image generation model
)
