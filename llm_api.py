import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from requests.exceptions import Timeout
import time

def get_llm_api_key():
    load_dotenv()
    return os.getenv("LLM_API_KEY")

def generate_text_by_llm_api_via_openai(messages, model_name, logprobs=0, temperature=0.2):
    api_key = get_llm_api_key()
    client = OpenAI(
        base_url='https://xxx',
        api_key=api_key
    )
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # logprobs=logprobs,
                temperature=temperature,
            )
            return response  
        except Timeout:  
            retries += 1
            print(f"Time out, retry...({retries}/{max_retries})")
            time.sleep(2)  
        except Exception as e:  
            print(f"error: {e}")
            raise  

    raise Exception("request failed, please check your internet or api service.")  

if __name__ == '__main__':
    api_key = get_llm_api_key()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
    # model_name = "gpt-3.5-turbo-1106"
    model_name = "gpt-4o-mini"
    response = generate_text_by_llm_api_via_openai(messages, model_name, logprobs=1)
    print(type(response))
    # calcuate_perplexity(response)
    # print(response.choices[0].logprobs.content)