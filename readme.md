# ChatGPT Prompt Engineering for Developers

This project demonstrates various prompting techniques and best practices for working with Large Language Models (LLMs). It includes examples of different prompting principles, tactics, and common use cases.

## Overview

The project covers several key areas of LLM interaction:

- **Prompting Principles**
  - Writing clear and specific instructions
  - Giving the model time to "think"
  - Using delimiters for distinct inputs
  - Structured output generation

- **Key Applications**
  - Text summarization and extraction
  - Language translation and tone transformation
  - Format conversion (JSON, HTML)
  - Grammar and spelling correction
  - Interactive chat interfaces

## Features

- **Interactive Components**
  - Chat interface with styled message display
  - File upload widget for text analysis
  - Temperature control for response randomness
  - Cost calculation for API usage

- **Utility Functions**
  - Text processing and format conversion
  - Multi-language support
  - File handling (CSV, TXT)
  - Data visualization helpers

## Prerequisites

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository
2. Create a `.env` file in the root directory
3. Add your environment variables:
```
OPENAI_API_KEY=your_api_key_here
BASE_PATH=your_base_path_here
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

### Interactive Components
```python
from helper_functions import open_chatbot, upload_txt_file_widget

# Launch interactive chatbot
open_chatbot()

# Create file upload widget
upload_txt_file_widget()
```

## Project Structure

- `helper_functions.py`: Core utilities and API interactions
- `prompting_guidelines.ipynb`: Interactive examples and demonstrations
- Various example files for testing and demonstration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ☕ Support Me

If you find this project helpful, consider supporting me on Ko-fi:
Your contributions will help cover fees and materials for my **Computer Science and Engineering studies at UoPeople** starting in September 2025.

<a href="https://ko-fi.com/miqueasmd"><img src="https://ko-fi.com/img/githubbutton_sm.svg" /></a>

## Acknowledgements

This project is inspired by the DeepLearning.AI courses. Please visit [DeepLearning.AI](https://www.deeplearning.ai/) for more information and resources.
