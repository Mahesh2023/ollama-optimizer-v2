"""Standard prompt suite for LLM benchmarking."""
from dataclasses import dataclass


@dataclass
class BenchmarkPrompt:
    name: str
    prompt: str
    category: str
    expected_tokens: int  # Expected output length


STANDARD_PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        name="short_qa",
        category="qa",
        prompt="What is the capital of France?",
        expected_tokens=20,
    ),
    BenchmarkPrompt(
        name="medium_essay",
        category="generation",
        prompt="Write a 300-word essay on why GPUs are better than CPUs for deep learning.",
        expected_tokens=400,
    ),
    BenchmarkPrompt(
        name="code_gen",
        category="code",
        prompt="Write a Python function to implement binary search. Include docstring and type hints.",
        expected_tokens=200,
    ),
    BenchmarkPrompt(
        name="summary",
        category="summarization",
        prompt=(
            "Summarize this text in 3 bullet points: "
            "The NVIDIA H100 GPU is based on the Hopper architecture and features 80GB of HBM3 memory. "
            "It offers up to 3TB/s of memory bandwidth and supports FP8 precision for AI workloads. "
            "The H100 is designed for large-scale AI training and inference workloads. "
            "It includes Transformer Engine for accelerating LLM operations."
        ),
        expected_tokens=100,
    ),
    BenchmarkPrompt(
        name="reasoning",
        category="reasoning",
        prompt=(
            "A train leaves Station A at 3pm going 60mph. Another train leaves Station B at 4pm going 90mph toward Station A. "
            "If the stations are 300 miles apart, at what time do they meet? Think step by step."
        ),
        expected_tokens=300,
    ),
    BenchmarkPrompt(
        name="long_context",
        category="long_context",
        prompt="Below is a technical document. Extract all mentioned technologies:\n\n" + ("Lorem ipsum " * 200),
        expected_tokens=150,
    ),
    BenchmarkPrompt(
        name="translation",
        category="translation",
        prompt="Translate to French: 'The future of AI inference lies in edge devices with specialized hardware.'",
        expected_tokens=30,
    ),
]


def get_prompt(name: str) -> BenchmarkPrompt | None:
    for p in STANDARD_PROMPTS:
        if p.name == name:
            return p
    return None


def get_prompts_by_category(category: str) -> list[BenchmarkPrompt]:
    return [p for p in STANDARD_PROMPTS if p.category == category]
