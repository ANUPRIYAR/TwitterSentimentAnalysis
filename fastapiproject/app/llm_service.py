import openai
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
import config

from app.utils import  prompt_instruction


openai_api_credentials = {
    "OPENAI_API_KEY": config.OPENAI_API_KEY,
    "OPENAI_API_TYPE": config.OPENAI_API_TYPE,
    "OPENAI_API_VERSION": config.OPENAI_API_VERSION,
}

# embedding_config = {
#     "azure_get_embedding": get_configuration_property("AZURE_OPEN_AI_GET_EMBEDDING"),
#     "azure_llm_model_ada": get_configuration_property("AZURE_OPEN_AI_LLM_MODEL_ADA"),
# }

openai.api_type = openai_api_credentials["OPENAI_API_TYPE"]
openai.api_version = openai_api_credentials["OPENAI_API_VERSION"]
openai.api_key = openai_api_credentials["OPENAI_API_KEY"]
base_url = config.OPENAI_API_BASE


def get_answer(query):
    llm_type = 'gpt-4'
    development_name = config.AZURE_OPEN_AI_DEPLOYMENT_NAME_GPT_4

    model = AzureChatOpenAI(
        azure_endpoint=base_url,
        openai_api_version=openai_api_credentials["OPENAI_API_VERSION"],
        deployment_name=development_name,
        openai_api_key=openai_api_credentials["OPENAI_API_KEY"],
        openai_api_type=openai_api_credentials["OPENAI_API_TYPE"],

    )

    prompt = f"{prompt_instruction}\nText: {query}\nLabel:"

    response = model(
        [HumanMessage(content=prompt)],
        max_tokens=4000
    )
    print(response.content)
    generated_response = response.content

    return generated_response
