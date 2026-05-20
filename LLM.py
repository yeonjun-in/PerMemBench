from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "claude", "together", "gemini", "vllm"]

DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-5.1-mini",
    "claude": "claude-sonnet-4-5",
    "together": "openai/gpt-oss-120b",
    "gemini": "gemini-3-flash-preview",
    "vllm": "Qwen/Qwen3-14B",
}

ENV_KEYS: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "together": "TOGETHER_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "vllm": "",  # vLLM does not require an API key (any value works)
}


class UnifiedLLM:
    """Unified chat client for OpenAI / Claude / Together AI APIs."""

    def __init__(
        self,
        provider: Provider = "openai",
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.base_url = base_url  # custom base URL (e.g. vLLM server)
        self.temperature = temperature
        self._api_key = os.environ.get(ENV_KEYS[provider]) if ENV_KEYS[provider] else None
        if not self._api_key and provider != "vllm":
            raise ValueError(
                f"Set {ENV_KEYS[provider]} in .env or pass api_key to use {provider}."
            )
        self.model = model or DEFAULT_MODELS[provider]
        self._openai_client = None
        self._together_client = None
        self._anthropic_client = None
        self._gemini_client = None
        self._vllm_client = None

    def _get_openai_client(self):
        from openai import OpenAI
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self._api_key)
        return self._openai_client

    def _get_together_client(self):
        from openai import OpenAI
        if self._together_client is None:
            self._together_client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.together.xyz/v1",
            )
        return self._together_client

    def _get_anthropic_client(self):
        from anthropic import Anthropic
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=self._api_key)
        return self._anthropic_client

    def _get_gemini_client(self):
        from google import genai
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._api_key)
        return self._gemini_client

    def _get_vllm_client(self):
        from openai import OpenAI
        if self._vllm_client is None:
            base = self.base_url or "http://localhost:8000/v1"
            # vLLM OpenAI-compatible endpoint
            if not base.endswith("/v1"):
                base = base.rstrip("/") + "/v1"
            self._vllm_client = OpenAI(api_key="vllm", base_url=base)
        return self._vllm_client

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> str:

        model = model or self.model

        if self.provider == "openai":
            return self._chat_openai(prompt, system, model)
        if self.provider == "together":
            return self._chat_together(prompt, system, model)
        if self.provider == "claude":
            return self._chat_claude(prompt, system, model)
        if self.provider == "gemini":
            return self._chat_gemini(prompt, system, model)
        if self.provider == "vllm":
            return self._chat_vllm(prompt, system, model)
        raise ValueError(f"Unknown provider: {self.provider}")

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> str:

        model = model or self.model
        if self.provider == "openai" and tools is not None:
            return self._chat_messages_openai_responses(messages, system, model, tools)
        if self.provider == "openai":
            return self._chat_messages_openai(messages, system, model)
        if self.provider == "together":
            return self._chat_messages_together(messages, system, model)
        if self.provider == "claude":
            return self._chat_messages_claude(messages, system, model)
        if self.provider == "gemini":
            return self._chat_messages_gemini(messages, system, model)
        if self.provider == "vllm":
            return self._chat_messages_vllm(messages, system, model)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_messages_openai(
        self, messages: list[dict], system: str | None, model: str
    ) -> str:
        client = self._get_openai_client()
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        try:
            r = client.chat.completions.create(model=model, messages=full, temperature=self.temperature)
        except Exception as e:
            if not self._is_temperature_unsupported_error(e):
                raise
            r = client.chat.completions.create(model=model, messages=full)
        return (r.choices[0].message.content or "").strip()

    def _chat_messages_openai_responses(
        self,
        messages: list[dict],
        system: str | None,
        model: str,
        tools: list[dict],
    ) -> str:
        """Use OpenAI Responses API (e.g. web_search tool)."""
        client = self._get_openai_client()
        parts = []
        if system:
            parts.append(f"[System]\n{system}\n\n[Conversation]")
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        input_text = "\n\n".join(parts)
        r = client.responses.create(
            model=model,
            tools=tools,
            input=input_text,
        )
        return (r.output_text or "").strip()

    def _chat_messages_together(
        self, messages: list[dict], system: str | None, model: str
    ) -> str:
        client = self._get_together_client()
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        try:
            r = client.chat.completions.create(model=model, messages=full, temperature=self.temperature)
        except Exception as e:
            if not self._is_temperature_unsupported_error(e):
                raise
            r = client.chat.completions.create(model=model, messages=full)
        return (r.choices[0].message.content or "").strip()

    def _chat_messages_claude(
        self, messages: list[dict], system: str | None, model: str
    ) -> str:
        client = self._get_anthropic_client()
        kwargs = {"model": model, "max_tokens": 4096, "messages": messages, "temperature": self.temperature}
        if system:
            kwargs["system"] = system
        r = client.messages.create(**kwargs)
        if not r.content:
            return ""
        return r.content[0].text.strip()

    def _chat_messages_gemini(
        self, messages: list[dict], system: str | None, model: str
    ) -> str:
        from google.genai import types
        client = self._get_gemini_client()
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
        )
        r = client.models.generate_content(model=model, contents=contents, config=config)
        return (r.text or "").strip()

    @staticmethod
    def _is_temperature_unsupported_error(e: Exception) -> bool:
        msg = str(e)
        return (
            "temperature" in msg
            and ("unsupported" in msg.lower() or "does not support" in msg.lower())
        )

    def _chat_openai(self, prompt: str, system: str | None, model: str) -> str:
        client = self._get_openai_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            if not self._is_temperature_unsupported_error(e):
                raise
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        return response.choices[0].message.content or ""

    def _chat_together(self, prompt: str, system: str | None, model: str) -> str:
        client = self._get_together_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            if not self._is_temperature_unsupported_error(e):
                raise
            response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    def _chat_claude(self, prompt: str, system: str | None, model: str) -> str:
        client = self._get_anthropic_client()
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        # content is a ContentBlock list; text is content[0].text
        if not response.content:
            return ""
        return response.content[0].text

    def _chat_gemini(self, prompt: str, system: str | None, model: str) -> str:
        from google.genai import types
        client = self._get_gemini_client()
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def _chat_vllm(self, prompt: str, system: str | None, model: str) -> str:
        client = self._get_vllm_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=model, messages=messages, temperature=self.temperature)
        return (response.choices[0].message.content or "").strip()

    def _chat_messages_vllm(
        self, messages: list[dict], system: str | None, model: str
    ) -> str:
        client = self._get_vllm_client()
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        response = client.chat.completions.create(model=model, messages=full, temperature=self.temperature)
        return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    # usage example (only one provider may work depending on configured API keys)
    for provider in ("openai", "claude", "together", "gemini", "vllm"):
        try:
            llm = UnifiedLLM(provider=provider)
            out = llm.chat("Say hello in one sentence.")
            print(f"[{provider}] {out}")
        except ValueError as e:
            print(f"[{provider}] skip: {e}")
