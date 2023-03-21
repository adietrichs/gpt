import tiktoken

from gpt_types import *


TOKEN_PARAMS = {
    "gpt-3.5-turbo": {"per_message": 5},
    "gpt-4": {"per_message": 4},
}


def calculate_prompt_tokens(messages: list[Message], model: Model) -> int:
    embedding = tiktoken.encoding_for_model(str(model))
    return 2 + sum(
        len(embedding.encode(message.content))
        + TOKEN_PARAMS[model.family]["per_message"]
        for message in messages
    )


def calculate_completion_tokens(choices: list[Choice], model: Model) -> int:
    embedding = tiktoken.encoding_for_model(str(model))
    return sum(len(embedding.encode(choice.message.content)) + 1 for choice in choices)
