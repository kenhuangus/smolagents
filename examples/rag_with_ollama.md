# AI Agent with SmolAgent and OLLAMA's LLAMA3.3 Local Model and RAG

This code example show how to use Huggingface SmolAgent with Ollama's LLAMA3.3 local mode and the RAG. The agent retrieves relevant documentation from a dataset and employs a language model to generate informative responses.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Code Explanation](#code-explanation)
- [Running the Code](#running-the-code)
- [Sample Output](#sample-output)
- [License](#license)

## Prerequisites

Ensure you have the following installed on your machine:

- Python 3.12
- pip (Python package installer)

## Setup Instructions

1. **Clone the Repository**

   Open your terminal and clone this repository:


2. **Create a Virtual Environment**

   Create a virtual environment using Python 3.12:

   ```
   python3.12 -m venv venv
   ```

3. **Activate the Virtual Environment**

   - On macOS and Linux:

     ```
     source venv/bin/activate
     ```

   - On Windows:

     ```
     venv\Scripts\activate
     ```

4. **Install Required Packages**

   Install the necessary packages using pip:

   ```
   pip install datasets langchain langchain-community smolagents requests
   ```

5. **Set Up Your API Key**

   In the code, replace `"your-api-key"` in the `LiteLLMModel` initialization with your actual API key for the language model service you are using.

## Code Explanation

Here is the complete code for the project:

```
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents.agents import ToolCallingAgent
from smolagents import Tool, LiteLLMModel

# Load the knowledge base dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Process the documents from the dataset
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant documentation about using Transformers in Hugging Face."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        
        # Retrieve documents based on the query
        docs = self.retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize the language model
model = LiteLLMModel(
    model_id="ollama_chat/llama3.3",
    api_base="http://localhost:11434",  # replace with remote OpenAI compatible server if necessary
    api_key="your-api-key"  # replace with API key if necessary
)

model.set_verbose = True

# Create an instance of the retriever tool
retriever_tool = RetrieverTool(docs_processed)

# Create an agent that combines the retriever tool with the language model
agent = ToolCallingAgent(tools=[retriever_tool], model=model)

# Example usage of the agent to conduct research on using Transformers in Hugging Face
if __name__ == "__main__":
    topic_input = "How to use Transformers in Hugging Face"  # Updated topic for research
    agent_output = agent.run(f"Research on: {topic_input}")
    
    print("Final output:")
    print(agent_output)
```

### Code Breakdown

1. **Importing Libraries**:
   - The code begins by importing necessary libraries such as `datasets`, `langchain`, and `smolagents`. These libraries provide functionalities for data handling, document processing, and building AI agents.

2. **Loading Knowledge Base**:
   - The knowledge base is loaded from a dataset available on Hugging Face. The dataset is filtered to include only documents related to Transformers.

3. **Processing Documents**:
   - Each document is processed into a format suitable for retrieval using `Document`, which includes content and metadata.
   - The `RecursiveCharacterTextSplitter` is used to split documents into smaller chunks (500 characters) while maintaining some overlap (50 characters) for context.

4. **Creating the Retriever Tool**:
   - A custom tool class `RetrieverTool` is defined, which uses BM25 retrieval for semantic search.
   - The `forward` method retrieves relevant documents based on a user query.

5. **Initializing Language Model**:
   - An instance of `LiteLLMModel` is created, which represents the language model that will generate responses based on retrieved documents.

6. **Creating Agent**:
   - An agent is created by combining the retriever tool and the language model.

7. **Running the Agent**:
   - When executed, the agent takes a specified topic (in this case, how to use Transformers in Hugging Face) and retrieves relevant information from the knowledge base.

## Running the Code

Once you have set up your environment and installed the required packages, you can run the code as follows:

1. Open your preferred code editor or IDE.
2. Create a new Python file (e.g., `main.py`) and copy the provided code into this file.
3. Run the script:

   ```
   python main.py
   ```

## Sample Output

When you run the code, it will output information related to using Transformers in Hugging Face based on retrieval from the dataset. 
Here’s an example of what you might see in the output:

```
Final output:
Retrieved documents:

===== Document 0 =====
To use Transformers in Hugging Face, first install the library using pip:
```bash
pip install transformers
```
Then, import it in your Python script:
```python
from transformers import pipeline
```
You can create various pipelines such as text generation, sentiment analysis, etc.
...

===== Document 1 =====
The Hugging Face Transformers library provides pre-trained models for various NLP tasks...
...
```

### Notes:
- Replace `"your-api-key"` with your actual API key for accessing remote language model service.
- Ensure that all necessary dependencies are accurately listed based on your implementation.
- Adjust any specific instructions or configurations needed based on your actual setup or requirements.
