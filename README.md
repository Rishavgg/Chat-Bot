# Portfolio Companion Chatbot

The Portfolio Companion Chatbot is a simple text-based chatbot that provides responses based on predefined patterns and sentence similarity. It is created using Python and utilizes the NLTK library for text processing and Scikit-learn's TfidfVectorizer for calculating cosine similarity between sentences.

## Features

- **Greeting and Farewell Responses**: The chatbot greets users with friendly messages and bids farewell when users indicate they are done chatting.

- **User Queries**: The chatbot responds to user queries about its status, availability, and well-being, providing programmed responses.

- **Response Generation**: The core functionality of the chatbot involves generating responses based on the similarity of the user's input with sentences from a provided text. The chatbot calculates cosine similarity to identify the most relevant sentence from the text.

## Getting Started

1. Clone the repository: `git clone https://github.com/Rishavgg/Chat-Bot.git`
2. Install the required libraries: `pip install nltk scikit-learn`
3. Place your text content in the 'information.txt' file.
4. Run the chatbot: `python app.py`
5. Interact with the chatbot by entering text queries.

## Example Usage
I will add soon

## Limitations

- The chatbot's responses are limited to the provided patterns and the content of the 'information.txt' file.
- It may not handle complex queries or understand context outside of predefined patterns.

## Contributing

Contributions are welcome! If you'd like to enhance the chatbot's functionality or fix any issues, feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [GNU General Public License v2.0](LICENSE) file for details.
