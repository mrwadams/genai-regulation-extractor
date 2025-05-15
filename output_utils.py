"""Utility functions for parsing and post-processing Gemini model output.

This module is responsible for:
  • Stripping markdown code fences from the model output.
  • Parsing the raw JSON/YAML strings into Python dictionaries with robust
    fall-back logic to salvage partially malformed JSON.
  • Merging a list of chunk-level structures into a single document-level
    structure.  The merging strategy is intentionally simple for this PoC and
    can be replaced with a more sophisticated deduplication/merging algorithm
    in future versions.
"""

import json
import yaml
from typing import Any, Dict, List


def _strip_code_fences(text: str) -> str:
    """Remove common markdown code fences from the model output."""
    lines = text.splitlines()
    cleaned_lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(cleaned_lines).strip()


def parse_llm_output(raw_text: str, output_format: str) -> Dict[str, Any]:
    """Parse raw LLM output string into a Python dictionary.

    Args:
        raw_text: The raw text returned by the LLM for a chunk.
        output_format: "JSON" or "YAML" (case-insensitive).

    Returns:
        Parsed dictionary representation of the structured data.

    Raises:
        ValueError: If parsing fails or output is not valid.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty output from LLM")

    cleaned = _strip_code_fences(raw_text)

    fmt_upper = output_format.upper()

    def _try_json(text: str):
        """Try standard JSON parse first, then trim to the outermost braces and retry."""
        try:
            return json.loads(text)
        except Exception:
            # Attempt salvage: extract substring from first '{' to last '}'
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    return json.loads(text[first : last + 1])
                except Exception:
                    pass

            # Advanced salvage: walk through text to find a balanced JSON object
            balanced = _extract_complete_json(text)
            if balanced:
                try:
                    return json.loads(balanced)
                except Exception:
                    pass
            raise

    def _extract_complete_json(text: str) -> str | None:
        """Return the first balanced JSON object found in *text* or None."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(text[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
            else:
                if ch == "\"":
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        return candidate
        return None

    if fmt_upper == "JSON":
        try:
            return _try_json(cleaned)
        except Exception as e:  # noqa: BLE001
            # As a fallback, attempt YAML parsing (YAML is a superset of JSON)
            try:
                return yaml.safe_load(cleaned)
            except Exception:
                raise ValueError(f"Failed to parse JSON: {e}") from e
    elif fmt_upper == "YAML":
        try:
            return yaml.safe_load(cleaned)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to parse YAML: {e}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def merge_structures(chunk_structs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of chunk-level structures into a single document structure.

    The merging strategy is intentionally simple for the PoC: concatenate the
    `sections` lists in order. Future versions may deduplicate or merge based on
    section IDs.
    """
    if not chunk_structs:
        return {}

    # Use the first chunk as the base structure
    merged = dict(chunk_structs[0])
    merged_sections: List[dict] = merged.get("sections", [])

    for struct in chunk_structs[1:]:
        sections = struct.get("sections", [])
        merged_sections.extend(sections)

    merged["sections"] = merged_sections
    return merged


def flatten_requirements(struct: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the hierarchical *struct* into a list of requirement rows.

    Each row contains the section information alongside the requirement fields so
    that it can be easily exported as CSV/Excel.
    """
    rows: List[Dict[str, Any]] = []

    def _recurse(sections: List[Dict[str, Any]], parent_path: str = "") -> None:
        for sec in sections or []:
            sec_id = sec.get("id", "")
            sec_title = sec.get("title", "")
            path = sec_id if not parent_path else f"{parent_path}.{sec_id}"
            for req in sec.get("requirements", []) or []:
                rows.append(
                    {
                        "section_id": path,
                        "section_title": sec_title,
                        "requirement_id": req.get("req_id", ""),
                        "requirement_text": req.get("text", ""),
                        "keywords": (
                            ", ".join(req.get("keywords", []))
                            if isinstance(req.get("keywords"), list)
                            else req.get("keywords", "")
                        ),
                    }
                )
            # Recurse into any nested children (handle both "subsections" and "sections")
            child_sections = []
            if isinstance(sec.get("subsections"), list):
                child_sections.extend(sec["subsections"])
            if isinstance(sec.get("sections"), list):
                child_sections.extend(sec["sections"])

            if child_sections:
                _recurse(child_sections, path)

    _recurse(struct.get("sections", []))
    return rows 