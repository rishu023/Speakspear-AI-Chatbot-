Speakspear AI Chatbot

This project is an educational implementation of a custom GPT prototype, the Speakspear AI Chatbot, designed to generate Shakespearean-style text. The chatbot is built using PyTorch, with a Bigram Model and a simplified GPT model based on Transformer architecture. This project was created solely for learning purposes and is not intended for commercial use.
Project Overview

    Model Type: Bigram Model with Transformer Architecture
    Framework: PyTorch
    Training Data: Shakespeare’s writings (sample text in input_data.txt)
    Objective: To build a text generation model that mimics Shakespearean dialogue.

Features

    Bigram Language Model: A simple model with 1M parameters leveraging a Transformer architecture to generate text outputs.
    Frequency and Positional Encoding: Text data is preprocessed with frequency encoding and positional encoding to aid context understanding.
    Self-Attention Mechanism: Uses multi-head self-attention blocks to capture relationships between words in a sequence, enhancing context comprehension.

Disclaimer

    This code is intended for educational purposes only. It is not meant for commercial applications, and any resemblance to copyrighted works is purely coincidental and unintentional.

Files

    bigram_model.py: Contains the implementation of the Bigram language model, including training and text generation functions.
    gpt_model.py: Implements a simplified GPT model using Transformer blocks, with multiple layers of self-attention and feed-forward layers.
    input_data.txt: Sample text data (excerpts from Shakespeare) used to train the model.

Requirements

    Python 3.x
    PyTorch
    Additional libraries: torch, torch.nn, torch.optim

Getting Started

    Clone the repository:

git clone https://github.com/your-username/speakspear-ai-chatbot.git
cd speakspear-ai-chatbot

Install dependencies:

    pip install -r requirements.txt

    Run the Model:
        To train the Bigram model, execute bigram_model.py.
        To train the GPT model, execute gpt_model.py.

    Generate Text:
        Use the generate() method in either BigramLanguageModel or GPTLanguageModel to create Shakespearean-style text samples.

Reference

This code and project were inspired by a YouTube tutorial. You can view the video here:https://youtu.be/kCc8FmEb1nY?si=3-K-rxIfwQ6sN21B
Future Improvements

    Extend the model to handle longer context sizes.
    Experiment with different encoding techniques for improved style matching.

License

This project is licensed under the MIT License. See the LICENSE file for more details.
