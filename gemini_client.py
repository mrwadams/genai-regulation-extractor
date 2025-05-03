"""Wrapper utilities for interacting with the Gemini GenAI API.

Two public helpers are exposed:
  • :func:`get_structured_data_from_llm` – single-call extraction (optionally
    streamed internally).
  • :func:`stream_structured_data_from_llm` – true streaming generator that
    yields parts of the response incrementally, ideal for updating the UI in
    real-time within Streamlit.

Both helpers share the same prompt-building logic via the private
:func:`_build_prompt` function.
"""

import time

from google import genai
from google.genai import types
from typing import Literal


def _build_prompt(text_content: str, output_format: str) -> str:
    """Returns a formatted prompt for the extraction task."""
    return f"""
You are an expert in regulatory analysis. Extract the hierarchical structure (sections, subsections) and all explicit requirements from the following regulation text.

Return **only** a single {output_format.upper()} object following this schema:
  - document_title
  - sections (with id, title, text_content, requirements, subsections)
  - requirements (with req_id, text, keywords)

Text to analyze:
{text_content}
"""


def get_structured_data_from_llm(
    text_content: str,
    output_format: Literal["JSON", "YAML"],
    api_key: str,
    *,
    stream: bool = False,
    max_output_tokens: int | None = 8192,
    retries: int = 3,
) -> str:
    """Calls Gemini to extract structured regulation data.

    Args:
        text_content: Raw regulation text (chunk).
        output_format: "JSON" or "YAML".
        api_key: Gemini API key.
        stream: Whether to use streaming endpoint.
        max_output_tokens: Optional upper bound for output tokens.
        retries: Number of times to retry on transient failures.

    Returns:
        The model's response text (string).
    """

    client = genai.Client(api_key=api_key)

    request_prompt = _build_prompt(text_content, output_format)

    # Build config; None means use defaults.
    config = types.GenerateContentConfig(max_output_tokens=max_output_tokens) if max_output_tokens else None

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            if stream:
                # Streaming response collection
                response_stream = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=[request_prompt],
                    **({"config": config} if config else {}),
                )
                collected: list[str] = []
                for chunk in response_stream:
                    if chunk.text:
                        collected.append(chunk.text)
                return "".join(collected)
            else:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[request_prompt],
                    **({"config": config} if config else {}),
                )
                return response.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Exponential backoff for transient errors
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                break

    # If we exit the loop without returning, raise the last error
    raise RuntimeError(f"Gemini API call failed after {retries} attempts: {last_err}")


def stream_structured_data_from_llm(
    text_content: str,
    output_format: Literal["JSON", "YAML"],
    api_key: str,
    *,
    max_output_tokens: int | None = 8192,
):
    """Yields chunks of the model response as they arrive (streaming).

    Example usage::

        accumulated = ""
        for part in stream_structured_data_from_llm(text, "JSON", key):
            accumulated += part
            st.code(accumulated)

    Args are the same as :func:`get_structured_data_from_llm` but without retries.
    """

    client = genai.Client(api_key=api_key)

    request_prompt = _build_prompt(text_content, output_format)

    config = types.GenerateContentConfig(max_output_tokens=max_output_tokens) if max_output_tokens else None

    response_stream = client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=[request_prompt],
        **({"config": config} if config else {}),
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text 