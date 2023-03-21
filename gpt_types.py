from enum import Enum
import re
from typing import Any, Self


Json = dict[str, Any]


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    NULL = "null"


class DataClass:
    def dict(self) -> dict[str, Any]:
        return self.__dict__

    def __repr__(self) -> str:
        return self.dict().__repr__()


class Message(DataClass):
    def __init__(self, role: Role, content: str) -> None:
        super().__init__()
        self.role = role
        self.content = content

    def to_json(self) -> Json:
        return {"role": self.role.value, "content": self.content}

    def __repr__(self) -> str:
        return f'<{self.role.name}: "{self.content}">'

    @classmethod
    def from_json(cls, raw: Json) -> Self:
        return cls(Role(raw["role"]), raw["content"])

    @classmethod
    def system(cls, content: str) -> Self:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> Self:
        return cls(Role.USER, content)


class Choice(DataClass):
    def __init__(self, finish_reason: FinishReason, message: Message) -> None:
        super().__init__()
        self.finish_reason = finish_reason
        self.message = message

    @classmethod
    def from_json(cls, raw: Json) -> Self:
        return cls(
            FinishReason(raw["finish_reason"]), Message.from_json(raw["message"])
        )


class Model(DataClass):
    def __init__(self, version_or_family: str) -> None:
        super().__init__()
        if re.match(r".*-\d{4}$", version_or_family):
            self.version = version_or_family
            self.family = version_or_family[:-5]
        else:
            self.version = None
            self.family = version_or_family

    def __str__(self) -> str:
        return self.version or self.family

    def __repr__(self) -> str:
        return f"<Model {self}>"


class Usage(DataClass):
    def __init__(self, completion_tokens: int, prompt_tokens: int) -> None:
        super().__init__()
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens

    @classmethod
    def from_json(cls, raw: Json) -> Self:
        assert raw["total_tokens"] == raw["completion_tokens"] + raw["prompt_tokens"]
        return cls(raw["completion_tokens"], raw["prompt_tokens"])


class Response(DataClass):
    def __init__(
        self, choices: list[Choice], created: int, model: Model, usage: Usage
    ) -> None:
        super().__init__()
        self.choices = choices
        self.created = created
        self.model = model
        self.usage = usage

    def dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in ["choices", "usage"]}

    @classmethod
    def from_json(cls, raw: Json) -> Self:
        return cls(
            [Choice.from_json(raw_choice) for raw_choice in raw["choices"]],
            raw["created"],
            Model(raw["model"]),
            Usage.from_json(raw["usage"]),
        )
