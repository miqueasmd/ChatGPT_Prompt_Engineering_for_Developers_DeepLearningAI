# ChatGPT Prompt Engineering for Developers

This project demonstrates various prompting techniques and best practices for working with Large Language Models (LLMs). It includes examples of different prompting principles, tactics, and common use cases.

## Overview

The project covers several key areas of LLM interaction:

- **Prompting Principles**
  - Writing clear and specific instructions
  - Giving the model time to "think"

- **Key Techniques**
  - Text summarization
  - Inference tasks
  - Text transformation
  - Format conversion
  - Chat-based interactions

## Features

- **Structured Output Generation**: Examples of generating JSON, HTML, and other structured formats
- **Text Analysis**: Sentiment analysis, emotion detection, and topic inference
- **Language Translation**: Multi-language translation capabilities
- **Grammar & Spelling**: Text correction and improvement
- **OrderBot Demo**: Interactive pizza ordering chatbot example

## Prerequisites

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository
2. Create a `.env` file in the root directory
3. Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

The project is organized into several Jupyter notebooks demonstrating different prompting techniques:

1. `prompting_guidelines.ipynb`: Main examples of prompting principles and tactics
2. `helper_functions.py`: Utility functions for API interactions

## Examples

### Basic Prompting
```python
prompt = f"""
Summarize the text delimited by triple backticks into a single sentence.
```{text}```
"""
response = get_llm_response(prompt)
```

### Chat Format
```python
messages = [  
    {'role':'system', 'content':'You are a helpful assistant.'},    
    {'role':'user', 'content':'Hi, my name is Isa'}  
]
response = get_completion_from_messages(messages)
```
## ☕ Support Me

If you like my work, consider supporting my studies!

Your contributions will help cover fees and materials for my **Computer Science and Engineering studies  at UoPeople** starting in September 2025.

Every little bit helps—you can donate from as little as $1.

<a href="https://ko-fi.com/miqueasmd"><img src="https://ko-fi.com/img/githubbutton_sm.svg" /></a>

## Acknowledgements

This project is inspired by the DeepLearning.AI courses. Please visit [DeepLearning.AI](https://www.deeplearning.ai/) for more information and resources.
