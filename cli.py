import openai
import os
from colorama import Fore, Style, init
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory


def chat_with_openai(messages):
    print()

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=messages, max_tokens=512, stream=True
    )

    response_text = ""
    for chunk in response:
        choice = chunk["choices"][0]
        delta = choice["delta"]
        finish_reason = choice["finish_reason"]

        if finish_reason is None:
            assert delta.get("role", "assistant") == "assistant"
            content = delta.get("content")
            if content:
                print(Fore.MAGENTA + content + Style.RESET_ALL, end="")
                response_text += content
        else:
            print()
            if finish_reason != "stop":
                print(
                    Fore.RED
                    + "Unexpected finish reason: "
                    + finish_reason
                    + Style.RESET_ALL
                )
            print()

    return response_text


def chat_loop():
    print(
        Fore.YELLOW
        + "\nWelcome to the GPT-4 Chatbot. Start by entering the system message.\n"
        + Style.RESET_ALL
    )

    system_history_file = "system_history.txt"
    system_session = PromptSession(history=FileHistory(system_history_file))
    system_message = system_session.prompt()
    messages = [{"role": "system", "content": system_message}]

    print(
        Fore.YELLOW + "\nHave fun chatting! Press Ctrl + D to exit.\n" + Style.RESET_ALL
    )

    history_file = "chat_history.txt"
    session = PromptSession(history=FileHistory(history_file))
    while True:
        try:
            user_input = session.prompt()
        except EOFError:
            break

        messages.append({"role": "user", "content": user_input})
        response_text = chat_with_openai(messages)
        messages.append({"role": "assistant", "content": response_text})


def main():
    init(autoreset=True)

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.organization = os.environ.get("OPENAI_ORG_ID")

    chat_loop()


if __name__ == "__main__":
    main()
