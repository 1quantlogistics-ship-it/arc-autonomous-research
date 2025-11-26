"""
LLM Client: OpenAI-compatible API client for LLM interactions
==============================================================

Enhanced version with multi-model support and retry logic.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List


class LLMClient:
    """
    Generic LLM client for OpenAI-compatible APIs.

    Supports:
    - vLLM endpoints
    - OpenAI API
    - Local model servers
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/generate",
        model_name: str = "deepseek-r1",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize LLM client.

        Args:
            endpoint: API endpoint URL
            model_name: Model identifier
            api_key: API key (if required)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
        """
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop generation at these sequences
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": self.model_name,
            **kwargs
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()

                # Handle different response formats
                if "text" in data:
                    return data["text"][0] if isinstance(data["text"], list) else data["text"]
                elif "choices" in data:
                    return data["choices"][0]["text"]
                else:
                    return str(data)

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"LLM request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"LLM request failed after {self.max_retries} attempts: {e}")

        return ""

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate JSON response from prompt.

        Automatically extracts JSON from response (handles think tags).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(prompt, max_tokens, temperature)
        return self.extract_json(response)

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from text response.

        Handles:
        - Think tags (<think>...</think>)
        - JSON wrapped in code blocks
        - Pure JSON

        Args:
            text: Response text

        Returns:
            Parsed JSON dictionary
        """
        # Remove think tags
        if "<think>" in text:
            # Extract content after think tags
            parts = text.split("</think>")
            if len(parts) > 1:
                text = parts[-1]

        # Find JSON object with proper brace matching
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in response")

        # Find matching closing brace
        brace_count = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if brace_count != 0:
            raise ValueError("Unmatched braces in JSON response")

        json_str = text[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

    def health_check(self) -> bool:
        """
        Check if LLM endpoint is healthy.

        Returns:
            True if endpoint is reachable
        """
        try:
            response = requests.get(
                self.endpoint.replace("/generate", "/health"),
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def __repr__(self) -> str:
        return f"<LLMClient model={self.model_name} endpoint={self.endpoint}>"
