## 🚀 Beyond Basics: Unpacking Hugging Face's Smol Agent Framework and Building a RAG-Powered Research Assistant with Ollama's Llama 3!

Start my new year with the AI Agents framework. I wrote about Dspy and CrewAI agent in my previous Linkedin post. In this article, I will talk about smolagents framework from Hugging Face. The framework simplifies the creation of sophisticated, multi-step agents that can use tools and interact intelligently. Let's go beyond the surface and dissect the core `agent.py` code to see how it works, then connect it to a real-world example: a research agent powered by **Retrieval-Augmented Generation (RAG)** and **Ollama's Llama 3**!

**Diving Deep into `agent.py`**

At its heart, `agent.py` is the core of Smol Agent Framework. Here's a breakdown of the key components:

1. **`MultiStepAgent` - The Foundation:**
    
    *   This is the base class for agents that operate in a step-by-step manner, following the **ReAct** framework (Reasoning and Acting).
    *   **Key Methods:**
        *   `run()`: The main loop that orchestrates the agent's actions, including planning, tool execution, and interaction.
        *   `step()`: An abstract method (to be implemented by subclasses) that defines a single step in the agent's process.
        *   `planning_step()`:  Periodically called to allow the agent to strategize its next moves, leveraging prompts like `SYSTEM_PROMPT_PLAN` and `SYSTEM_PROMPT_FACTS_UPDATE`. This is where the agent thinks before it acts!
        *   `execute_tool_call()`:  Handles the execution of a tool, including argument substitution and error handling.
        *   `write_inner_memory_from_logs()`:  Transforms the agent's logs into a conversational format, providing context for the LLM.
2. **`ToolCallingAgent` - Leveraging LLM Tool Calling:**
    
    *   Inherits from `MultiStepAgent`.
    *   Designed for agents that use LLMs with built-in tool-calling capabilities (like those that can output function calls in a structured format).
    *   **`step()` Implementation:**
        *   Uses `model.get_tool_call()` to get the LLM's suggested tool and arguments.
        *   Executes the tool via `execute_tool_call()`.
        *   Handles final answers and updates observations.
3. **`CodeAgent` - For Code-Generating Agents:**
    
    *   Also inherits from `MultiStepAgent`.
    *   Specialized for agents that generate and execute code.
    *   **`step()` Implementation:**
        *   Gets code from the LLM using `model()`.
        *   Parses the code using `parse_code_blob()`.
        *   Executes the code with either a local or remote Python interpreter.
        *   Handles `final_answer` extraction from the code.
4. **`Tool` and `Toolbox` - Expanding Capabilities:**
    
    *   `Tool` is a class defining the interface for tools that agents can use.
    *   `Toolbox` manages a collection of tools, making them available to the agent.

**Example: Building a RAG Research Agent**

Now, let's see how these pieces fit together in our RAG example:

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents.agents import ToolCallingAgent
from smolagents import Tool, LiteLLMModel

# Load and process a knowledge base (Hugging Face documentation)
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))
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

# Create a custom tool for document retrieval
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

# Initialize Ollama's Llama 3 model using LiteLLMModel
model = LiteLLMModel(
    model_id="ollama_chat/llama3.3",
    api_base="http://localhost:11434", # replace with remote OpenAI compatible server if necessary
    api_key="your-api-key"  # Replace with your API key if needed
)

# Create a ToolCallingAgent with the retriever tool
retriever_tool = RetrieverTool(docs_processed)
agent = ToolCallingAgent(tools=[retriever_tool], model=model)

# Ask the agent a question!
topic_input = "How to use Transformers in Hugging Face"
agent_output = agent.run(f"Research on: {topic_input}")

print("Final output:")
print(agent_output)
```

**Connecting the Dots:**

1. **`RetrieverTool` (Custom Tool):** We define a `RetrieverTool` that inherits from `Tool`. This tool encapsulates the logic for retrieving relevant documents using `BM25Retriever`. The `agent.py` framework allows us to define custom tools easily.
2. **`ToolCallingAgent`:** We create a `ToolCallingAgent` and provide it with our `RetrieverTool`. This agent will use Ollama's Llama 3 (via `LiteLLMModel`) to decide when to call the `RetrieverTool`.
3. **`LiteLLMModel`:**  This class from `smolagents` acts as a bridge to the Ollama API, allowing our agent to interact with Llama 3.
4. **The `run()` Loop:** When we call `agent.run()`, the `MultiStepAgent`'s core logic kicks in. The agent will use the LLM to generate thoughts, decide to use the `RetrieverTool`, execute it via `execute_tool_call()`, get the results, and repeat until it has a final answer or reaches the maximum iteration limit. The agent may also decide to call the `planning_step()` if you set the `planning_interval` when initialize `ToolCallingAgent`.
5. **RAG in Action:** The combination of the `RetrieverTool` and Llama 3 enables RAG. The agent retrieves relevant information from the knowledge base and uses it to generate a more informed and accurate response.

**Why This Matters:**

Understanding the inner workings of `agent.py` empowers you to:

*   **Build custom agents:**  Tailor agents to your specific needs by creating custom tools and implementing the `step()` method.
*   **Debug effectively:**  Trace the execution flow and identify bottlenecks or areas for improvement.
*   **Extend the framework:**  Contribute to the `smolagents` project by adding new features and functionalities.

**Getting Started:**

1. **Install:** `pip install datasets langchain langchain-community smolagents requests`
2. **Set up Ollama:** Download and run Llama 3 locally using Ollama.
3. **Run the Example:** Adapt the code above, and run it!

**The `smolagents` framework, combined with the power of local LLMs like Llama 3, is a potent combination for building the next generation of intelligent agents. Dive into the code, experiment, and unleash your creativity!**

**Let me know your thoughts and what you build in the comments!**

**#AI #agents #LLMs #Ollama #Llama3 #HuggingFace #transformers #RAG #developers #machinelearning #deeplearning #opensource #python #coding**
