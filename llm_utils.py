"""
Unified LLM interface for SI405 Applied AI course.
Supports both local Ollama and remote vLLM backends.

This module provides a consistent interface for students to use different
LLM backends without changing their code. Just toggle USE_REMOTE!
"""


class UnifiedLLM:
    """
    Wrapper that provides a consistent interface for Ollama and vLLM.

    Usage:
        from llm_utils import UnifiedLLM

        # Local Ollama with default model (llama3.2:3b)
        llm = UnifiedLLM(use_remote=False)

        # Specify model size
        llm = UnifiedLLM(use_remote=False, model="3b")   # llama3.2:3b (default)
        llm = UnifiedLLM(use_remote=False, model="1b")   # llama3.2:1b

        # Use Qwen3 models (note: has thinking mode that can't be fully disabled)
        llm = UnifiedLLM(use_remote=False, model="qwen-4b")   # qwen3:4b
        llm = UnifiedLLM(use_remote=False, model="qwen-0.6b") # qwen3:0.6b

        # Use Qwen3.5 models
        llm = UnifiedLLM(use_remote=False, model="qwen3.5-4b") # qwen3.5:4b

        # Or specify full model name directly
        llm = UnifiedLLM(use_remote=False, model="mistral:7b")

        # Remote vLLM server
        llm = UnifiedLLM(use_remote=True)

        # Use like any LangChain LLM
        response = llm.invoke("Hello!")
    """

    # Default server configurations
    REMOTE_VLLM_URL = "http://burger.si.umich.edu:8001/v1"
    LOCAL_OLLAMA_URL = "http://localhost:11434"

    # Model naming conventions differ between backends
    # Students can specify size (e.g., "3b", "4b") or full model name
    # Default is llama3.2:3b which doesn't have thinking mode issues
    MODEL_MAP = {
        "ollama": {
            # Llama 3.2 models (recommended - no thinking mode issues)
            "3b": "llama3.2:3b",
            "1b": "llama3.2:1b",
            # Qwen3 models (have thinking mode that can't be fully disabled)
            "qwen-4b": "qwen3:4b",
            "qwen-0.6b": "qwen3:0.6b",
            # Qwen3.5 models
            "qwen3.5-4b": "qwen3.5:4b",
            # Legacy/alternate aliases
            "4b": "qwen3:4b",
            "0.6b": "qwen3:0.6b",
            "qwen:4b": "qwen3:4b",
            "qwen:0.6b": "qwen3:0.6b",
            "qwen3:4b": "qwen3:4b",
            "qwen3:0.6b": "qwen3:0.6b",
            "qwen3.5:4b": "qwen3.5:4b",
            "llama3.2:3b": "llama3.2:3b",
            "llama3.2:1b": "llama3.2:1b",
            # Default model
            "default": "llama3.2:3b"
        },
        "vllm": {
            "3b": "meta-llama/Llama-3.2-3B-Instruct",
            "1b": "meta-llama/Llama-3.2-1B-Instruct",
            "qwen-4b": "Qwen/Qwen3-4B",
            "qwen-8b": "Qwen/Qwen3-8B",
            "qwen-0.6b": "Qwen/Qwen3-0.6b",
            "qwen3.5-4b": "Qwen/Qwen3.5-4B",
            "4b": "Qwen/Qwen3-4B",
            "8b": "Qwen/Qwen3-8B",
            "0.6b": "Qwen/Qwen3-0.6b",
            # Ollama-style names (students often try these with remote)
            "qwen:4b": "Qwen/Qwen3-4B",
            "qwen:8b": "Qwen/Qwen3-8B",
            "qwen:0.6b": "Qwen/Qwen3-0.6b",
            "qwen3:4b": "Qwen/Qwen3-4B",
            "qwen3-4b": "Qwen/Qwen3-4B",
            "qwen3:8b": "Qwen/Qwen3-8B",
            "qwen3-8b": "Qwen/Qwen3-8B",
            "qwen3:0.6b": "Qwen/Qwen3-0.6b",
            "qwen3-0.6b": "Qwen/Qwen3-0.6b",
            "qwen3.5:4b": "Qwen/Qwen3.5-4B",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
            "default": "meta-llama/Llama-3.2-3B-Instruct"
        }
    }

    def __init__(self, use_remote=False, model="3b", temperature=0.7, max_tokens=1024, stop=None, thinking=False):
        """
        Initialize the LLM wrapper.

        Args:
            use_remote: True for vLLM server, False for local Ollama
            model: Model size ("3b", "1b", "qwen-4b") or full model name
                   Default is "3b" which uses llama3.2:3b
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response length
            stop: List of stop sequences (e.g., ["Observation:"] for ReAct agents)
            thinking: Show thinking/reasoning output for qwen3 models (default: False)
                      Only relevant for Qwen models. Reasoning is always enabled
                      internally for better quality; this flag controls whether
                      the <think> output is visible or stripped.
        """
        self.use_remote = use_remote
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_thinking = thinking

        # Resolve model name from size or use as-is if full name provided
        backend_key = "vllm" if use_remote else "ollama"
        model_map = self.MODEL_MAP[backend_key]

        # Case-insensitive lookup so "Qwen3:8b", "QWEN3:8B", etc. all work
        model_lower = model.lower() if model else None
        if model_lower in model_map:
            self.model_name = model_map[model_lower]
        elif model is None:
            self.model_name = model_map["default"]
        else:
            # Assume full model name provided
            self.model_name = model

        if use_remote:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key="EMPTY",
                openai_api_base=self.REMOTE_VLLM_URL,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            self.backend = "vLLM"
            self.url = self.REMOTE_VLLM_URL
        else:
            # Use ChatOllama for better compatibility with chat-optimized models
            from langchain_ollama import ChatOllama

            # Only set reasoning mode for Qwen models (other models don't use it)
            is_qwen = 'qwen' in self.model_name.lower()

            ollama_kwargs = {
                "model": self.model_name,
                "base_url": self.LOCAL_OLLAMA_URL,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            if stop:
                ollama_kwargs["stop"] = stop

            # Always enable reasoning for Qwen models (produces better quality).
            # Thinking output is stripped in invoke() unless show_thinking=True.
            # With reasoning=False, Qwen still leaks thinking text but WITHOUT
            # proper <think> tags, making it impossible to strip cleanly.
            # Reasoning tokens consume num_predict budget, so we increase it
            # to leave room for the actual answer after thinking.
            if is_qwen:
                ollama_kwargs["reasoning"] = True
                ollama_kwargs["num_predict"] = max_tokens + 4096

            self._llm = ChatOllama(**ollama_kwargs)
            self.backend = "Ollama"
            self.url = self.LOCAL_OLLAMA_URL

        print(f"Using {self.backend} at {self.url}")
        print(f"Model: {self.model_name}")
        if use_remote:
            print("  (Requires campus network or VPN)")

    def invoke(self, prompt):
        """
        Send prompt to LLM and return response as string.
        Handles response type differences between backends.
        """
        import re
        response = self._llm.invoke(prompt)
        # Normalize: ChatOpenAI returns AIMessage, OllamaLLM returns str
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        if not self.show_thinking:
            # With reasoning=True, ChatOllama separates thinking into
            # additional_kwargs['reasoning_content'] and puts the answer
            # in content. If the model ran out of tokens while thinking,
            # content may be empty — fall back to reasoning_content.
            if not content and hasattr(response, 'additional_kwargs'):
                reasoning = response.additional_kwargs.get('reasoning_content', '')
                if reasoning:
                    # Try to extract just the code from reasoning text.
                    # Look for markdown code blocks first, then function defs.
                    blocks = re.findall(r'```(?:python)?\n(.*?)```', reasoning, re.DOTALL)
                    if blocks:
                        content = blocks[-1].strip()
                    else:
                        # Find the last function/class definition as a heuristic
                        lines = reasoning.split('\n')
                        code_start = None
                        for i, line in enumerate(lines):
                            if re.match(r'^(def |class )', line.strip()):
                                code_start = i
                        if code_start is not None:
                            content = '\n'.join(lines[code_start:])
                        else:
                            content = reasoning

            # Strip any remaining <think>...</think> tags as a safety net
            if content:
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)

            # Qwen3 often wraps its answer in **answer** or **Answer:** markers.
            # Extract the content after the marker if present.
            if content:
                answer_match = re.search(
                    r'\*\*[Aa]nswer:?\*\*\s*\n?(.*)',
                    content, flags=re.DOTALL
                )
                if answer_match:
                    content = answer_match.group(1).strip()

        return content

    def __or__(self, other):
        """Support pipe operator for LangChain chains."""
        return self._llm | other

    def __ror__(self, other):
        """Support reverse pipe operator."""
        return other | self._llm

    @property
    def llm(self):
        """Access the underlying LangChain LLM object for advanced use."""
        return self._llm


def get_llm(use_remote=False, **kwargs):
    """Factory function for quick LLM creation."""
    return UnifiedLLM(use_remote=use_remote, **kwargs)


def get_chat_model(use_remote=False, model="3b", temperature=0.7, thinking=False):
    """
    Get a chat model that supports native tool binding via bind_tools().

    This returns a ChatOllama or ChatOpenAI instance that can be used with
    LangChain's native tool calling:

        from llm_utils import get_chat_model
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))

        chat_model = get_chat_model()
        model_with_tools = chat_model.bind_tools([calculator])
        response = model_with_tools.invoke("What is 15 * 7?")

    Args:
        use_remote: True for vLLM server, False for local Ollama
        model: Model size ("3b", "1b", "qwen-4b") or full model name
               Default is "3b" which uses llama3.2:3b
        temperature: Sampling temperature (0-1)
        thinking: Show thinking/reasoning output for qwen3 models (default: False)
                  Only relevant for Qwen models. Reasoning is always enabled
                  internally for better quality; this controls visibility.
                  Use get_thinking_stripped_parser() to strip tags in chains.

    Returns:
        ChatOllama or ChatOpenAI instance with bind_tools() support
    """
    # Resolve model name (case-insensitive)
    backend_key = "vllm" if use_remote else "ollama"
    model_map = UnifiedLLM.MODEL_MAP[backend_key]

    model_lower = model.lower() if model else None
    if model_lower in model_map:
        model_name = model_map[model_lower]
    elif model is None:
        model_name = model_map["default"]
    else:
        model_name = model

    if use_remote:
        from langchain_openai import ChatOpenAI
        chat_kwargs = {
            "model": model_name,
            "openai_api_key": "EMPTY",
            "openai_api_base": UnifiedLLM.REMOTE_VLLM_URL,
            "temperature": temperature,
        }
        # Disable thinking mode for Qwen3.5 on vLLM — much faster responses
        if 'qwen' in model_name.lower() and not thinking:
            chat_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        chat_model = ChatOpenAI(**chat_kwargs)
        print(f"Using vLLM at {UnifiedLLM.REMOTE_VLLM_URL}")
    else:
        from langchain_ollama import ChatOllama

        is_qwen = 'qwen' in model_name.lower()

        ollama_kwargs = {
            "model": model_name,
            "base_url": UnifiedLLM.LOCAL_OLLAMA_URL,
            "temperature": temperature,
        }

        # Always enable reasoning for Qwen models for better quality.
        # Use get_thinking_stripped_parser() in chains to strip tags.
        if is_qwen:
            ollama_kwargs["reasoning"] = True

        chat_model = ChatOllama(**ollama_kwargs)
        print(f"Using Ollama at {UnifiedLLM.LOCAL_OLLAMA_URL}")

    print(f"Model: {model_name}")
    if thinking:
        print("  (thinking mode enabled — slower but higher quality reasoning)")
    return chat_model


def get_thinking_model(use_remote=False, model="qwen3.5:4b", temperature=0.3):
    """
    Get a chat model with thinking/reasoning enabled.

    Use this for tasks that benefit from step-by-step reasoning:
    planning, synthesis, analysis, evaluation.

    Slower but produces higher quality responses for complex tasks.

    Usage:
        from llm_utils import get_thinking_model
        thinker = get_thinking_model(use_remote=True)
        response = thinker.invoke("Plan which tools to use for...")
    """
    return get_chat_model(use_remote=use_remote, model=model, temperature=temperature, thinking=True)


def get_fast_model(use_remote=False, model="qwen3.5:4b", temperature=0.3):
    """
    Get a chat model with thinking disabled for fast responses.

    Use this for simple tasks that don't need deep reasoning:
    classification, routing, short answers, formatting.

    Much faster than thinking mode — ideal for agent routing and simple LLM calls.

    Usage:
        from llm_utils import get_fast_model
        fast = get_fast_model(use_remote=True)
        response = fast.invoke("Classify this query as product/outfit/scene: ...")
    """
    return get_chat_model(use_remote=use_remote, model=model, temperature=temperature, thinking=False)


def get_embeddings(use_remote=False, model="nomic-embed-text"):
    """
    Get an embedding model for local or remote inference.

    Usage:
        from llm_utils import get_embeddings

        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(["Hello", "World"])
        query_vector = embeddings.embed_query("Hello World")

    Args:
        use_remote: True for remote server, False for local Ollama
        model: Embedding model name (default: nomic-embed-text)

    Returns:
        LangChain Embeddings instance (OllamaEmbeddings or compatible)
    """
    if use_remote:
        # Remote embedding endpoint - using Ollama protocol
        # Note: vLLM doesn't serve embeddings the same way, so we use Ollama
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model=model,
            base_url=UnifiedLLM.LOCAL_OLLAMA_URL  # Use local for embeddings even in remote mode
        )
        print(f"Using Ollama embeddings at {UnifiedLLM.LOCAL_OLLAMA_URL}")
    else:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model=model,
            base_url=UnifiedLLM.LOCAL_OLLAMA_URL
        )
        print(f"Using Ollama embeddings at {UnifiedLLM.LOCAL_OLLAMA_URL}")

    print(f"Embedding model: {model}")
    return embeddings


def get_vectorstore(persist_directory, collection_name="documents", use_remote=False):
    """
    Get or load a ChromaDB vector store.

    Usage:
        from llm_utils import get_vectorstore

        # Load existing vector store
        vectorstore = get_vectorstore("./chroma_db")

        # Search
        results = vectorstore.similarity_search("query", k=3)

        # With metadata filter
        results = vectorstore.similarity_search(
            "query",
            k=3,
            filter={"source": "langchain"}
        )

    Args:
        persist_directory: Path to ChromaDB storage
        collection_name: Name of the collection (default: "documents")
        use_remote: Whether to use remote embeddings (currently uses local)

    Returns:
        Chroma vector store instance
    """
    from langchain_chroma import Chroma

    embeddings = get_embeddings(use_remote=use_remote)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    print(f"Loaded ChromaDB from: {persist_directory}")
    print(f"Collection: {collection_name}")

    return vectorstore


def _clean_qwen_output(text):
    """
    Clean Qwen3 model output by stripping thinking tags and **answer** markers.

    Handles two patterns:
    1. <think>...</think> tags (reasoning output)
    2. **answer**/**Answer:** markers (Qwen3 answer formatting)

    Args:
        text: Raw text from Qwen3 model

    Returns:
        Cleaned text with only the answer content
    """
    import re
    if not text:
        return text
    # Strip <think>...</think> tags
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    # Extract content after **answer** marker if present
    answer_match = re.search(
        r'\*\*[Aa]nswer:?\*\*\s*\n?(.*)',
        text, flags=re.DOTALL
    )
    if answer_match:
        text = answer_match.group(1).strip()
    return text


def get_thinking_stripped_parser():
    """
    Get an output parser that strips <think> tags and **answer** markers
    from Qwen3 output.

    Usage:
        from llm_utils import get_llm, get_thinking_stripped_parser

        llm = get_llm()
        parser = get_thinking_stripped_parser()

        chain = prompt | llm.llm | parser
        result = chain.invoke({"query": "hello"})

    Returns:
        A LangChain output parser that strips thinking tags and answer markers
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    def strip_thinking(text):
        if isinstance(text, str):
            return _clean_qwen_output(text)
        return text

    # Chain StrOutputParser with thinking tag stripper
    return StrOutputParser() | RunnableLambda(strip_thinking)


def strip_thinking_tags(text: str) -> str:
    """
    Clean Qwen3 output by stripping <think> tags and **answer** markers.

    Utility function for manually cleaning Qwen3 output.

    Args:
        text: Text potentially containing thinking tags or answer markers

    Returns:
        Cleaned text with only the answer content
    """
    return _clean_qwen_output(text)


def vision_chat(image_path: str, prompt: str, use_remote=False, model="qwen3.5:4b") -> str:
    """
    Send an image + text prompt to a vision-capable model and return the response.

    Abstracts the difference between local Ollama and remote vLLM so callers
    use the same interface regardless of backend.

    Usage:
        from llm_utils import vision_chat

        # Local (Ollama)
        result = vision_chat("photo.jpg", "Describe this image.")

        # Remote (vLLM)
        result = vision_chat("photo.jpg", "Describe this image.", use_remote=True)

    Args:
        image_path: Path to the image file
        prompt: Text prompt / question about the image
        use_remote: True for vLLM server, False for local Ollama
        model: Model name (Ollama-style, e.g. "qwen3.5:4b"). Automatically
               resolved to the vLLM name when use_remote=True.

    Returns:
        Model's text response about the image
    """
    import base64

    # Resolve model name for the chosen backend
    backend_key = "vllm" if use_remote else "ollama"
    model_map = UnifiedLLM.MODEL_MAP[backend_key]
    model_lower = model.lower() if model else None
    resolved = model_map.get(model_lower, model)

    if use_remote:
        from openai import OpenAI

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/jpeg")

        client = OpenAI(base_url=UnifiedLLM.REMOTE_VLLM_URL, api_key="EMPTY")
        response = client.chat.completions.create(
            model=resolved,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
            # Disable thinking mode for vision — keeps responses fast and clean
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _clean_qwen_output(response.choices[0].message.content or "")
    else:
        import ollama as _ollama
        response = _ollama.chat(
            model=resolved,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        )
        return _clean_qwen_output(response["message"]["content"])


def get_ollama_client(use_remote=False):
    """
    Get an Ollama client for direct API access.

    Useful for embeddings and other direct Ollama operations.

    Usage:
        from llm_utils import get_ollama_client

        client = get_ollama_client()

        # Generate embeddings
        response = client.embed(model="nomic-embed-text", input=["Hello"])
        embeddings = response["embeddings"]

        # Generate text
        response = client.generate(model="qwen3:4b", prompt="Hello!")

    Returns:
        ollama.Client instance
    """
    from ollama import Client

    url = UnifiedLLM.LOCAL_OLLAMA_URL
    client = Client(host=url)

    print(f"Ollama client connected to: {url}")
    return client
