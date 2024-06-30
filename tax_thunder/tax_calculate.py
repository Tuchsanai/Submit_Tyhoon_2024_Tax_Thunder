import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate



def typhoon_instruct_OPENAPI_complete(
    system_prompt, user_prompt, temperature=0.5, max_tokens=3200, checked_json=False
):

    def is_json(my_string):
        try:
            json_object = json.loads(my_string)
        except ValueError as e:
            return False
        return True

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("human", "{input}"),
        ]
    )

    completion = prompt | ChatOpenAI(
        api_key="xxxxxxx",
        openai_api_base="https://api.opentyphoon.ai/v1",
        model="typhoon-v1.5x-70b-instruct",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output = completion.invoke(
        {"system_prompt": system_prompt, "input": user_prompt}
    ).content

    if checked_json == False:
        return output
    else:
        if is_json(output):
            return output
        else:
            return None


def tax_calculator(user_input: str):
    result_json = typhoon_instruct_OPENAPI_complete(
        system_prompt=SYSTEM_MESSAGE_TYPHOON_JSON_TAX,
        user_prompt=user_input,
        temperature=0,
        max_tokens=2048,
        checked_json=True,
    )

    if isinstance(result_json, str) and is_json(result_json):
        output, final_tax = calculate_thai_income_tax(result_json)
        return {"result": output}
    else:
        return {"result": "Error in processing the input\n"}
