import os

import requests
from crewai import LLM
from icecream import ic
from langchain_together import Together
from litellm import completion

from src.data_insight_analysis.llm.llms import llm_together


def call_huggingface_llm_api(
    model_name: str = "huggingface/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    input: str = "Today is a great day",
):
    base_url = "https://api-inference.huggingface.co/models/"
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        f"{base_url}{model_name}",
        headers=headers,
        json={
            "inputs": input,
        },
    )
    ic(response.status_code)
    ic(response.json())


# # Example function to handle API call and errors
def call_huggingface_llm(
    model_name: str = "huggingface/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    input: str = "Today is a great day",
):

    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": input}],
        )
        ic(response)
    except Exception as e:
        ic(f"An error occurred: {e}")


def call_together_llm(
    input: str = "Tell about 5K running. What are the benefits of running 5K daily?",
):
    ic(input)
    try:
        response = llm_together.invoke(input)
        ic(response)
    except Exception as e:
        ic(f"An error occurred: {e}")


# # Call the function to test
call_together_llm()


#  free models to test with the API are:
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "HuggingFaceH4/zephyr-7b-beta"
# "TheBloke/Llama-2-7B-GGUF"  # Free version of Llama-2
# "microsoft/phi-2"  # 2.7B parameters
