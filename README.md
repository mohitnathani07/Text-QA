# Text-QA
Here’s a README file for GitHub with the title and description:

---

# Text Extraction

This project extracts text from each page of a PDF file. It converts PDF pages into images and applies OCR (Optical Character Recognition) using EasyOCR to capture text from each image. The extracted text is saved to a file, with separate entries for each page, enabling efficient text extraction from PDF documents.

# Vector Database
This project demonstrates an end-to-end pipeline for preparing text data for vector search. It chunks text into overlapping segments, converts each chunk into vector embeddings using SentenceTransformer, and stores these embeddings in a Pinecone vector database. The process enables efficient similarity search and retrieval of text data based on vector embeddings.

# RAG with Evaluation and Frontend

Here's an overview of the provided code:

### Project Overview
This code is designed to implement a **Retrieval-Augmented Generation (RAG) system** with evaluation metrics and a user-friendly frontend. It integrates various machine learning libraries such as Hugging Face `transformers`, `sentence_transformers`, `pinecone`, `nltk`, and `gradio` for a complete RAG workflow that includes data retrieval, model inference, and interactive user querying.

### Code Breakdown

1. **Authentication and Setup**:
   ```python
   from huggingface_hub import login
   login('YOUR LOGIN KEY')
   ```
   - Logs into Hugging Face to access pre-trained models.

2. **Library Imports and Initialization**:
   - Imports libraries for sentence embedding (`sentence_transformers`), vector storage (`pinecone`), and question-answering pipelines (`transformers`).
   - Sets up Pinecone for storing and querying high-dimensional vectors.
   ```python
   api_key = 'YOUR API KEY'
   pinecone_client = Pinecone(api_key=api_key)
   ```

3. **Pinecone Index Management**:
   ```python
   if index_name not in pinecone_client.list_indexes().names():
       pinecone_client.create_index(
           name=index_name,
           dimension=384,
           metric='cosine',
           spec=ServerlessSpec(cloud='aws', region='us-east-1')
       )
   index = pinecone_client.Index(index_name)
   ```
   - Connects to or creates a Pinecone index to store embeddings with a specified dimensionality and similarity metric.

4. **Embedding and Model Loading**:
   ```python
   embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
   model_id = "meta-llama/Llama-3.2-3B-Instruct"
   llama_pipeline = transformers.pipeline("text-generation", model=model_id, ...)
   ```
   - Loads an embedding model (`all-MiniLM-L6-v2`) and sets up a LLaMA model pipeline for text generation.

5. **Query Processing Functions**:
   - **`generate_query_embedding()`**: Encodes the user query into an embedding.
   - **`retrieve_relevant_chunks()`**: Queries Pinecone for top-K most similar text chunks based on the embedding.
   - **`construct_prompt()`**: Constructs a prompt for the LLaMA model using retrieved chunks.
   - **`get_answer_from_llama()`**: Generates a response from the LLaMA model using a structured prompt.
   - **`respond_to_query()`**: Combines the steps to generate a final response for a user query.

6. **Evaluation Metrics**:
   ```python
   def evaluation_matrics(response, context):
       bleu_score = sentence_bleu([context], response)
       response_vector = model.encode(response)
       context_vector = model.encode(context)
       similarity = cosine_similarity(response_vector.reshape(1, -1), context_vector.reshape(1, -1))
       return bleu_score, similarity[0][0]
   ```
   - **BLEU Score**: Measures how closely the generated response matches the context.
   - **Cosine Similarity**: Compares the semantic similarity between the response and the context.

7. **Frontend with Gradio**:
   ```python
   demo = gr.Interface(
       fn=run,
       inputs=["text"],
       outputs=["text"],
   )
   ```
   - **Gradio Interface**: Provides an interactive frontend where users can input their queries and receive responses. The `run()` function handles the query processing, context retrieval, and evaluation.

### Summary of Workflow
1. The user inputs a query in the Gradio interface.
2. The query is encoded and relevant text chunks are retrieved from Pinecone.
3. A prompt is constructed and passed to the LLaMA model for generating a response.
4. The generated response is evaluated using BLEU and cosine similarity metrics.
5. The result is displayed to the user in the Gradio frontend.

This code establishes a complete pipeline for handling user queries, retrieving relevant information, generating answers, and evaluating the results with a user-friendly interface.
