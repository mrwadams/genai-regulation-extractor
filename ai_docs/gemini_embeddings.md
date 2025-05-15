# Gemini API - Embeddings Documentation

The Gemini API supports several embedding models that generate embeddings for words, phrases, code, and sentences. The resulting embeddings can then be used for tasks such as semantic search, text classification, and clustering, among many others.

## What are embeddings?

Embeddings are numerical representations of text (or other media formats) that capture relationships between inputs. Text embeddings work by converting text into arrays of floating point numbers, called _vectors_. These vectors are designed to capture the meaning of the text. The length of the embedding array is called the vector's _dimensionality_. A passage of text might be represented by a vector containing hundreds of dimensions.

Embeddings capture semantic meaning and context, which results in text with similar meanings having "closer" embeddings. For example, the sentence "I took my dog to the vet" and "I took my cat to the vet" would have embeddings that are close to each other in the vector space.

## Generate embeddings

Use the `embedContent` method to generate text embeddings:

### Python
```python
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?")

print(result.embeddings)
```

### JavaScript
```javascript
import { GoogleGenAI } from "@google/genai";

async function main() {
    const ai = new GoogleGenAI({ apiKey: "GEMINI_API_KEY" });

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-exp-03-07',
        contents: 'What is the meaning of life?',
    });

    console.log(response.embeddings);
}

main();
```

## Task Types

Task types enable you to generate optimized embeddings for specific tasks. Here are the supported task types:

| Task type | Description |
| --- | --- |
| `SEMANTIC_SIMILARITY` | Used to generate embeddings that are optimized to assess text similarity. |
| `CLASSIFICATION` | Used to generate embeddings that are optimized to classify texts according to preset labels. |
| `CLUSTERING` | Used to generate embeddings that are optimized to cluster texts based on their similarities. |
| `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `QUESTION_ANSWERING`, `FACT_VERIFICATION` | Used to generate embeddings that are optimized for document search or information retrieval. |
| `CODE_RETRIEVAL_QUERY` | Used to retrieve a code block based on a natural language query. Embeddings of the code blocks are computed using `RETRIEVAL_DOCUMENT`. |

### Example with Task Type:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?",
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
print(result.embeddings)
```

## Use Cases

Text embeddings are used in a variety of common AI use cases:

1. **Information retrieval:** Use embeddings to retrieve semantically similar text given a piece of input text.
2. **Clustering:** Compare groups of embeddings to identify hidden trends.
3. **Vector database:** Store embeddings in a vector database for production use cases.
4. **Classification:** Train models using embeddings to classify documents into categories.

## Available Models

The Gemini API offers three models that generate text embeddings:

- gemini-embedding-exp-03-07
- text-embedding-004
- embedding-001

Source: [Google AI for Developers - Embeddings Documentation](https://ai.google.dev/gemini-api/docs/embeddings)
Last updated: 2025-04-03 UTC 