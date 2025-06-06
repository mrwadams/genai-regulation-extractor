# GenAI Regulation Extraction Tool (Proof of Concept)

## Overview

The GenAI Regulation Extraction Tool is a web-based application built with Streamlit that assists users in extracting structured information (such as requirements, sections, and keywords) from PDF documents containing regulatory text. It leverages Google's Gemini generative AI model to parse and interpret the content of the uploaded PDFs, providing the output in either JSON or YAML format, as well as a tabular view of extracted requirements.

This tool is designed as a Proof of Concept (PoC) to demonstrate the potential of AI in automating parts of the regulatory analysis workflow.

## Features

*   **PDF Upload:** Easily upload regulation documents in PDF format.
*   **Selective Page Processing:** Option to specify page ranges for targeted extraction.
*   **Configurable Chunking:** Adjust the number of pages processed per AI call to manage API limits and improve output quality.
*   **AI-Powered Extraction:** Utilizes Google Gemini for understanding and structuring information from text.
*   **Structured Output:** Provides results in user-selected JSON or YAML formats.
*   **Tabular View:** Displays extracted requirements in an interactive table.
*   **Data Download:** Allows downloading of the full structured output (JSON/YAML) and the requirements table (CSV).
*   **User-Friendly Interface:** Simple and intuitive UI built with Streamlit.

## Setup and Running Locally

Follow these steps to set up and run the GenAI Regulation Extraction Tool on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate a Python Virtual Environment:**
    (Recommended to avoid conflicts with other Python projects)
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file in your project root with the necessary packages. If not, create one with at least the following:
    ```
    streamlit
    google-generativeai
    PyPDF2
    python-dotenv
    PyYAML
    pandas
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    The tool requires a Google Gemini API key. You can set this as an environment variable named `GOOGLE_API_KEY`.
    Create a `.env` file in the project root:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
    The application will load this key automatically. Alternatively, you can enter the API key directly in the UI.

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This will start the application, and you can access it in your web browser (usually at `http://localhost:8501`).

## Configuration

The application provides several configuration options directly in the user interface:

*   **Upload PDF:** Select the regulation PDF document to process.
*   **(Optional) Page Range:** Specify particular pages or ranges of pages to process (e.g., `1-3, 5, 7-8`). Leave empty to process all pages.
*   **Pages per Chunk:** Define how many PDF pages are sent to the AI model in each request. Smaller values (e.g., 1 or 2) can help with large documents or if you encounter output length errors from the AI, but may take longer.
*   **Output Format:** Choose between JSON and YAML for the structured data output.
*   **Gemini API Key:** Enter your Google Gemini API key if not set via the `.env` file.

## Important Considerations & Disclaimer

*   **AI-Powered Extraction:** This tool utilizes a Generative AI model (Google Gemini). The accuracy and completeness of the extracted information are dependent on the clarity of the PDF document, the complexity of the regulatory text, and the inherent capabilities and limitations of the AI model.
*   **User Review is Crucial:** **Always** carefully review the output generated by this tool. AI models can make mistakes, misinterpret nuances, or omit information. This tool should be used as an assistant to augment human review, not as a replacement for it.
*   **Proof of Concept (PoC):** This application is a Proof of Concept. It is intended to demonstrate a potential use case for AI in regulatory analysis and may have limitations. It is not recommended for production use without further rigorous development, testing, and validation.
*   **API Costs:** Be mindful of the costs associated with using the Google Gemini API, especially when processing large documents or many files.

## Potential Future Enhancements

*   Support for other document formats (e.g., .docx, .txt).
*   More sophisticated chunking strategies (e.g., by section rather than page count).
*   Advanced error handling and retry mechanisms for API calls.
*   Integration with vector databases for semantic search over extracted requirements.
*   User accounts and history of processed documents.
*   Batch processing of multiple documents.
*   Fine-tuning options or prompt engineering UI for more tailored extractions.

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to contribute to the code, please feel free to fork the repository, make your changes, and submit a pull request. You can also open an issue to report bugs or suggest new features.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.