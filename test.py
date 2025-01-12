import os

import requests
from crewai import LLM
from icecream import ic
from litellm import completion


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


# # Call the function to test
call_huggingface_llm()


#  free models to test with the API are:
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "HuggingFaceH4/zephyr-7b-beta"
# "TheBloke/Llama-2-7B-GGUF"  # Free version of Llama-2
# "microsoft/phi-2"  # 2.7B parameters
