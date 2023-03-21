import openai

from gpt_types import *


class Chat:
    def __init__(self, system_prompt="", **params):
        self.messages = [Message.system(system_prompt)]
        self._params = params
        self.debug = False
        self.latest_response = None

    @property
    def params(self):
        return {"model": "gpt-4", "max_tokens": 512, **self._params}

    def pop_dirty(self, expect_user):
        if (self.messages[-1].role is Role.USER) is not expect_user:
            self.messages.pop()

    def send(self, content, **params):
        self.pop_dirty(False)
        self.messages.append(Message.user(content))
        try:
            raw = openai.ChatCompletion.create(
                messages=[message.to_json() for message in self.messages],
                **{**self.params, **params}
            )
        except openai.OpenAIError:
            self.pop_dirty(False)
            return None
        self.latest_response = Response.from_json(raw)
        self.messages.append(self.latest_response.choices[0].message)
        self._params["model"] = self.latest_response.model.model_str
        return self.messages[-1].content

    def resend(self, content=None):
        self.pop_dirty(True)
        popped_content = self.messages.pop().content
        return self.send(content or popped_content)
