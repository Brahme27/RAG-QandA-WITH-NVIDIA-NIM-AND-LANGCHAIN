# PDF Q\&A App using NVIDIA NIM + LangChain + Streamlit

This Streamlit app allows you to **ask questions from a folder of PDF files** using **NVIDIA NIM LLMs** and **LangChain**. It reads and processes PDF documents, converts them into vector embeddings using NVIDIA’s embedding model, and answers your questions based on the document context.


## Features

* Loads your **NVIDIA API key** from `.env`
* Loads PDFs from the `./us_census` folder
* Splits text into chunks for efficient processing
* Uses NVIDIA LLM (`meta/llama3-70b-instruct`) to answer your questions
* Creates a FAISS vector database for fast document retrieval
* You can **ask questions directly in the UI**


## How it Works

1. **Document Embedding**:
   Click the `Document Embedding` button to:

   * Load PDFs
   * Split them into chunks
   * Create vector embeddings using NVIDIA
   * Store them in a FAISS vector store

2. **Ask Questions**:
   Type your question related to the documents in the text box.
   The app retrieves relevant chunks and asks the LLM to answer based on that context.

## Setup Instructions

1. **Clone the repo** and install dependencies:

```cmd
pip install -r requirements.txt
```

2. **Create a `.env` file** in the project root:

```.env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

3. **Add your PDFs** to a folder named `us_census` in the project directory.

4. **Run the app**:

```cmd
streamlit run app.py
```

## Credits

* **LangChain** – for building the retrieval chain and document processing.
* **NVIDIA NIM** – for high-performance embeddings and LLM responses.
* **Streamlit** – for the interactive user interface.