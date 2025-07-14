# 03 Closed Source LLMs

## Introduction

This tutorial covers Closed Source Large Language Models (LLMs), their characteristics, and how they compare to Open Source LLMs.

## Overview of Closed Source LLMs

Closed Source LLMs are language models whose code and weights are not publicly available. They are typically developed and maintained by private companies. Here's a comparison of some of the leading models:

| Model | Developer | Key Strengths |
| --- | --- | --- |
| GPT-4 | OpenAI | State-of-the-art performance in reasoning and language understanding. |
| Gemini | Google | Multimodal capabilities (text, images, audio, video) and strong integration with Google services. |
| Claude 3 | Anthropic | Focus on safety, steerability, and a large context window for processing long documents. |

## Advantages of Closed Source LLMs

*   **High Performance:** Often achieve state-of-the-art results due to extensive resources and proprietary training data.
*   **Ease of Use:** Typically offer user-friendly APIs and interfaces.
*   **Reliability and Support:** Backed by dedicated teams and infrastructure.

## Disadvantages of Closed Source LLMs

*   **Lack of Transparency:** Limited access to model details and training data.
*   **Cost:** Usage often involves subscription fees or pay-per-use pricing.
*   **Limited Customization:** Fine-tuning options may be restricted.
*   **Vendor Lock-in:** Dependence on a specific provider.

## Accessing and Using Closed Source LLMs

*   **APIs:** Most closed-source LLMs are accessed through APIs (e.g., OpenAI API, Google AI Platform).
*   **SDKs:** Software Development Kits (SDKs) may be available to simplify integration.
*   **Web Interfaces:** Some providers offer web-based interfaces for interacting with their models.

## Code Example: Using a Closed Source LLM API for text summarization (Python with API calls - Example using OpenAI)

```python
import openai

# Set your API key
openai.api_key = "YOUR_API_KEY"  # Replace with your actual API key

def summarize_text(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or another suitable engine
            prompt=f"Summarize the following text:\n{text}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
text_to_summarize = """
Large Language Models (LLMs) are a type of artificial intelligence (AI) model that can understand and generate human-like text. They are trained on massive datasets of text and code, allowing them to perform a wide range of natural language processing (NLP) tasks. LLMs are revolutionizing various fields due to their ability to generate human-quality text, understand and respond to complex queries, and automate tasks.
"""
summary = summarize_text(text_to_summarize)
print(summary)
```

## Assignment

Compare the performance of an Open Source LLM and a Closed Source LLM on a specific task, considering factors like cost, speed, and accuracy. Document your findings, including the models used, the task, the evaluation metrics, and your conclusions.

## Interview Question

What are the trade-offs between Open Source and Closed Source LLMs?

## Exercises

1.  **Research Closed Source LLMs:** List three different Closed Source LLMs and briefly describe their key features and pricing models.
2.  **API Exploration:** Choose one Closed Source LLM and explore its API documentation. Describe the available features and limitations.
3.  **Cost Analysis:** Compare the cost of using a Closed Source LLM API with the cost of running an Open Source LLM on your own hardware (consider factors like hardware costs, electricity, and maintenance).
4.  **Ethical Considerations:** Discuss the ethical implications of using Closed Source LLMs, considering issues like data privacy, bias, and transparency.
