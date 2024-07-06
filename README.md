# Chatbot

This project is a simple chatbot that responds to user queries using natural language processing (NLP) techniques. It reads from a text file (`chatbot.txt`), preprocesses the text, and generates responses based on user input.

## Features

- Responds to greetings with predefined responses.
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to find the most relevant response to user queries.
- Handles user input and provides appropriate responses.

## Prerequisites

- Python 3.x
- NLTK library
- scikit-learn library
- numpy library

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chatbot.git
    ```

2. Navigate to the project directory:
    ```bash
    cd chatbot
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your text file (`chatbot.txt`) in the same directory as `main.py`.

2. Run the chatbot:
    ```bash
    python main.py
    ```

3. Interact with the chatbot in the terminal. Type `bye` to exit the chatbot.

## Code Overview

- **main.py**: Contains the main logic for the chatbot, including text preprocessing, greeting responses, and generating responses based on user input.

## Acknowledgements

- [NLTK](https://www.nltk.org/) library for natural language processing.
- [scikit-learn](https://scikit-learn.org/) library for machine learning algorithms.

