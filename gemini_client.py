from google import genai
from typing import Literal

def get_structured_data_from_llm(text_content: str, output_format: Literal["JSON", "YAML"], api_key: str) -> str:
    """
    Sends text_content to Gemini with a prompt to extract regulatory structure and requirements.
    Returns the raw model response (string).
    """
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are an expert in regulatory analysis. Extract the hierarchical structure (sections, subsections) and all explicit requirements from the following regulation text. 
Return the result as a single {output_format.upper()} object following this schema:
- document_title
- sections (with id, title, text_content, requirements, subsections)
- requirements (with req_id, text, keywords)

Text to analyze:
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt + text_content]
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}") 