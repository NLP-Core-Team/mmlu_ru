import abc
import typing as tp

        
class Conversation(abc.ABC):
    """
    Inspired by https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
    """
    def __init__(self, system_prompt: str, roles: tp.Tuple[str, str]):
        self.system_prompt = system_prompt
        self.roles = roles
        self.messages: tp.List[tp.Tuple[str, str]] = []

    @abc.abstractmethod
    def get_prompt(self) -> str:
        pass

    def update_last_message(self, text: str) -> None:
        self.messages[-1] = (self.messages[-1][0], text)

    def append_message(self, role: str, text: str) -> None:
        self.messages.append([role, text])

        
class LlamaConversation(Conversation):

    def __init__(self):
        super().__init__(
            system_prompt="",  # faked for compatibility
            roles=("", ""),
        )

    def get_prompt(self) -> str:
        prompt = self.system_prompt
        for role, text in self.messages:
            if text:
                prompt += f"{role}{text}"
            else:
                prompt += f"{role}"
        return prompt


class SaigaConversation(Conversation):
    def __init__(self):
        super().__init__(
            system_prompt="<s>system\nТы — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s>\n",
            roles=("user", "bot"),
        )

    def get_prompt(self) -> str:
        prompt = self.system_prompt
        for role, text in self.messages:
            if text:
                prompt += f"<s>{role}\n{text}</s>\n"
            else:
                prompt += f"<s>{role}"
        return prompt


conversation_classes = {
    "saiga": SaigaConversation,
    "llama": LlamaConversation,
}


if __name__ == "__main__":
    utterances = [
        "А тот второй парень — он тоже терминатор?",
        "Не совсем. Модель T-1000, усовершенствованный прототип.",
        "То есть, он более современная модель?",
        "Да. Мимикрирующий жидкий металл.",
        "И что это значит?",
    ]
    print("-=-=-=- Saiga -=-=-=-")
    conv = SaigaConversation()
    conv.append_message(conv.roles[0], utterances[0])
    conv.append_message(conv.roles[1], utterances[1])
    conv.append_message(conv.roles[0], utterances[2])
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
    print("-=-=-=-=-=-=-=-=-=-=-")
