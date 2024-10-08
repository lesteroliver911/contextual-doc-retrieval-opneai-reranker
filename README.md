
# Contextual Document Retrieval with LlamaParse, OpenAI GPT-4o, Contextual chunking, BM25, Query Expansion and Cohere Re-ranking

This repository provides a Python-based system for accurate document retrieval using advanced AI techniques. It combines OpenAI's GPT-4o model with Cohere for re-ranking and query expansion to enhance search precision. The system utilizes BM25 for keyword-based search and contextual chunking for improved understanding of documents, ensuring high-quality, context-aware results.

## Features
- **PDF Parsing**: Extracts content from PDFs using LlamaParse.
- **Contextual Chunking**: Splits documents into manageable chunks and provides contextual summaries using OpenAI's GPT-4.
- **BM25 Search**: Implements a BM25 search index for efficient keyword-based retrieval.
- **Cohere Re-ranking**: Enhances search results by re-ranking them using Cohere's reranking model.
- **Query Expansion**: Expands search queries using AI to improve retrieval performance.
- **Error Handling**: Robust exception handling ensures reliable document processing.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/contextual-doc-retrieval-opneai-reranker.git
    cd contextual-doc-retrieval-opneai-reranker
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file with the following API keys:
        ```bash
        OPENAI_API_KEY=your_openai_api_key
        COHERE_API_KEY=your_cohere_api_key
        LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
        ```

## Usage

1. **Run the Script**:
    ```bash
    python main.py
    ```

2. **Load PDF Document**:
    The script will prompt you to enter the path to your PDF file.

3. **Perform Document Search**:
    You can input your search queries, and the system will return the relevant results from the document using BM25 and re-ranking with Cohere.

## Example Query

- Original query: "Summarize the full document and explain the Fixture limits in detail."
- The system will provide both an original and expanded version of the query for better retrieval accuracy.

## Dependencies
- Python 3.x
- [LangChain](https://github.com/hwchase17/langchain)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Cohere](https://cohere.ai/)
- [BM25Okapi](https://github.com/dorianbrown/rank_bm25)
- [LlamaParse](https://github.com/your-username/llama_parse) (For PDF parsing)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or enhancements.
