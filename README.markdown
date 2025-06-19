# RAG-Based QA System

## Overview
This is a Retrieval-Augmented Generation (RAG) Question-Answering system that uses a single book as its source of truth. The system runs locally or on a private server, ensuring no external model knowledge leakage and compliance with privacy constraints. It uses FAISS for vector storage, Sentence Transformers for embeddings, and LangChain with Ollama for local LLM inference.

## Setup & Installation
1. **Prerequisites**:
   - Python 3.8+
   - Install Ollama and pull the `llama3.1` or `mistral` model: `ollama pull llama3.1` or mistal
   - Ensure the Ollama server is running at local server e.g.`http://192.168.11.204:11434`
   - A PDF book file (e.g., `book.pdf`)

2. **Create Virtual Environment**:
   ```bash
   python -m venv my_env
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
   ```

4. **Place the Book**:
   - Save your book PDF as `book_path.pdf` in the project directory.

## Deployment Steps
1. **Start the Ollama Server**:
   - Ensure the Ollama server is running at `http://192.168.11.204:11434` with the required model available.

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   - Access the UI at `http://localhost:8501`.

3. **Preprocessing**:
   - On first run, the system extracts text from `book.pdf`, chunks it, and creates a FAISS index (`faiss_index.bin`) and JSON file (`book_chunks.json`).
   - Subsequent runs load these files for faster startup.

## Usage
- Open the Streamlit app in your browser.
- Enter a question in the input field.
- The system retrieves relevant book chunks and generates an answer using the local LLM via LangChain.
- View the answer, retrieved context, and metadata (page numbers).

## Sample Inputs and Outputs
1. **Question**: What is the expected number of High BP cases by 2025?
   - **Answer**: around one billion individuals suffer from high BP worldwide and the number is expected to surge up to 1.56 billion by 2025.
   - **Context**: [Relevant chunk]
2. **Question**: What's the most widely used search engine for the methodology?
   - **Answer**: According to the text, "At the time of our research study, based on usage of all kinds of electronic devices, the topmost search engines around the globe are Google (77%), Baidu (14%) and Bing (4%)."
   - **Context**: [Relevant chunk]

3. **Question**: For statistical analysis which software was used?
   - **Answer**: IBM SPSS Statistics v-20 software.
   - **Context**: [Empty or irrelevant chunk]

4. **Question**: Give me the category wise percentage of websites?
   - **Answer**: Information 36% Government 24% Institution 24% Non-Profit Organization 16%

   - **Context**: [Relevant chunk]

5. **Question**: what does silent killer mean in the book?
   - **Answer**: In the context of the book, "silent killer" refers to high blood pressure (hypertension). According to the text, it is mentioned that "seventy percent of the research studies related to the evaluation of websites providing health information conclude that the clarity of sources of such sites is poor." It also says that only 3 out of 25 websites provided clear sources of information about high blood pressure, which had obtained a maximum score for this indicator.
   - **Context**: [Relevant chunk]


## Limitations
- The system is limited to the content of the provided book.
- Accuracy depends on the quality of text extraction from the PDF.
- The local LLM (e.g., Llama, Mistral) may have performance constraints based on hardware.
- Requires a running Ollama server at the specified URL e.g.server IP (`http://192.168.11.204:11434`).

## Security & Privacy
- All processing (text extraction, embedding, inference) occurs locally or on the private server.
- No external API calls or internet access at runtime beyond the specified Ollama server.
- Data is stored in local files (`book_chunks.json`, `faiss_index.bin`).