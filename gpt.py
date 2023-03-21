import openai
import tiktoken

from gpt_types import *
from utils import *


class Chat:
    def __init__(self, system_prompt="", model_version_or_family="gpt-4", **params):
        self.messages = [Message.system(system_prompt)]
        self.model = Model(model_version_or_family)
        self.params = params
        self.latest_response = None

    def pop_dirty(self, expect_user):
        if (self.messages[-1].role is Role.USER) is not expect_user:
            self.messages.pop()

    def send(self, content, **params):
        self.pop_dirty(False)
        self.messages.append(Message.user(content))
        try:
            raw = openai.ChatCompletion.create(
                messages=[message.to_json() for message in self.messages],
                model=str(self.model),
                **{**self.params, **params},
            )
        except openai.OpenAIError:
            self.pop_dirty(False)
            return None
        self.latest_response = Response.from_json(raw)
        if self.model.version is None:
            self.model.version = self.latest_response.model.version
        else:
            assert self.model.version == self.latest_response.model.version

        predicted_prompt_tokens = calculate_prompt_tokens(self.messages, self.model)
        predicted_completion_tokens = calculate_completion_tokens(
            self.latest_response.choices, self.model
        )
        assert predicted_prompt_tokens == self.latest_response.usage.prompt_tokens, (
            predicted_prompt_tokens,
            self.latest_response.usage.prompt_tokens,
        )
        assert (
            predicted_completion_tokens == self.latest_response.usage.completion_tokens
        ), (
            predicted_completion_tokens,
            self.latest_response.usage.completion_tokens,
        )

        self.messages.append(self.latest_response.choices[0].message)
        return self.messages[-1].content

    def resend(self, content=None):
        self.pop_dirty(True)
        popped_content = self.messages.pop().content
        return self.send(content or popped_content)
