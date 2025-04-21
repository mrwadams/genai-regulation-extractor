# Text generation

## Text input

The simplest way to generate text using the Gemini API is to provide the model
with a single text-only input, as shown in this example:

[Python](https://ai.google.dev/gemini-api/docs/text-generation#python)[JavaScript](https://ai.google.dev/gemini-api/docs/text-generation#javascript)[Go](https://ai.google.dev/gemini-api/docs/text-generation#go)[REST](https://ai.google.dev/gemini-api/docs/text-generation#rest)[Apps Script](https://ai.google.dev/gemini-api/docs/text-generation#apps-script)More

```
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)
print(response.text)

```

## Image input

The Gemini API supports multimodal inputs that combine text and media files.
The following example shows how to generate text from text and image input:

[Python](https://ai.google.dev/gemini-api/docs/text-generation#python)[JavaScript](https://ai.google.dev/gemini-api/docs/text-generation#javascript)[Go](https://ai.google.dev/gemini-api/docs/text-generation#go)[REST](https://ai.google.dev/gemini-api/docs/text-generation#rest)[Apps Script](https://ai.google.dev/gemini-api/docs/text-generation#apps-script)More

```
from PIL import Image
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, "Tell me about this instrument"]
)
print(response.text)

```

## Streaming output

By default, the model returns a response after completing the entire text
generation process. You can achieve faster interactions by using streaming to
return instances of
[`GenerateContentResponse`](https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse)
as they're generated.

[Python](https://ai.google.dev/gemini-api/docs/text-generation#python)[JavaScript](https://ai.google.dev/gemini-api/docs/text-generation#javascript)[Go](https://ai.google.dev/gemini-api/docs/text-generation#go)[REST](https://ai.google.dev/gemini-api/docs/text-generation#rest)[Apps Script](https://ai.google.dev/gemini-api/docs/text-generation#apps-script)More

```
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"]
)
for chunk in response:
    print(chunk.text, end="")

```

## Multi-turn conversations

The Gemini SDK lets you collect multiple rounds of questions and responses into
a chat. The chat format enables users to step incrementally toward answers and
to get help with multipart problems. This SDK implementation of chat provides an
interface to keep track of conversation history, but behind the scenes it uses
the same
[`generateContent`](https://ai.google.dev/api/generate-content#method:-models.generatecontent) method
to create the response.

The following code example shows a basic chat implementation:

[Python](https://ai.google.dev/gemini-api/docs/text-generation#python)[JavaScript](https://ai.google.dev/gemini-api/docs/text-generation#javascript)[Go](https://ai.google.dev/gemini-api/docs/text-generation#go)[REST](https://ai.google.dev/gemini-api/docs/text-generation#rest)[Apps Script](https://ai.google.dev/gemini-api/docs/text-generation#apps-script)More

```
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
chat = client.chats.create(model="gemini-2.0-flash")

response = chat.send_message("I have 2 dogs in my house.")
print(response.text)

response = chat.send_message("How many paws are in my house?")
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}',end=": ")
    print(message.parts[0].text)

```

## Configuration parameters

Every prompt you send to the model includes parameters that control how the
model generates responses. You can configure these parameters, or let the model
use the default options.

The following example shows how to configure model parameters:

[Python](https://ai.google.dev/gemini-api/docs/text-generation#python)[JavaScript](https://ai.google.dev/gemini-api/docs/text-generation#javascript)[Go](https://ai.google.dev/gemini-api/docs/text-generation#go)[REST](https://ai.google.dev/gemini-api/docs/text-generation#rest)[Apps Script](https://ai.google.dev/gemini-api/docs/text-generation#apps-script)More

```
from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1
    )
)
print(response.text)

```

## System instructions

System instructions let you steer the behavior of a model based on your specific
use case. When you provide system instructions, you give the model additional
context to help it understand the task and generate more customized responses.
The model should adhere to the system instructions over the full interaction
with the user, enabling you to specify product-level behavior separate from the
prompts provided by end users.

You can set system instructions when you initialize your model:

[Python](https://ai.google.dev/gemini-api/docs/text-generation#python)[JavaScript](https://ai.google.dev/gemini-api/docs/text-generation#javascript)[Go](https://ai.google.dev/gemini-api/docs/text-generation#go)[REST](https://ai.google.dev/gemini-api/docs/text-generation#rest)[Apps Script](https://ai.google.dev/gemini-api/docs/text-generation#apps-script)More

```
from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko."),
    contents="Hello there"
)

print(response.text)

```

## Supported models

The entire Gemini family of models supports text generation. To learn more
about the models and their capabilities, see [Models](https://ai.google.dev/gemini-api/docs/models).

## Prompting tips

For basic text generation use cases, your prompt might not need to include any
output examples, system instructions, or formatting information. This is a
[zero-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#zero-shot-prompts)
approach. For some use cases, a
[one-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#one-shot-prompts) or
[few-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#few-shot-prompts) prompt
might produce output that's more aligned with user expectations. In some cases,
you might also want to provide [system instructions](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions) to
help the model understand the task or follow specific guidelines.

## What's next

- Try the
[Gemini API getting started Colab](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb).
- Learn how to use Gemini's
[vision understanding](https://ai.google.dev/gemini-api/docs/vision) to process images and videos.
- Learn how to use Gemini's [audio understanding](https://ai.google.dev/gemini-api/docs/audio) to
process audio files.
- Learn about multimodal
[file prompting strategies](https://ai.google.dev/gemini-api/docs/file-prompting-strategies).



Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-04-04 UTC. 