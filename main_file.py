import requests
from bs4 import BeautifulSoup
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Define the URL of the college website to scrape
url = "https://vnrvjiet.ac.in/admission/"

# Send an HTTP GET request to the website
response = requests.get(url)

# Parse the HTML content of the webpage
soup = BeautifulSoup(response.text, "html.parser")

# Extract information from the webpage
# For example, let's extract the title of the webpage
page_title = soup.title.string
print("Page Title:", page_title)


# Create a chatbot instance
chatbot = ChatBot("CollegeChatbot")

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot using ChatterBot's built-in English language data
trainer.train("chatterbot.corpus.english")

# Define a function to interact with the chatbot
def chat_with_bot(user_input):
    response = chatbot.get_response(user_input)
    return response.text

# Example usage:
user_query = "Tell me about the admission process."
bot_response = chat_with_bot(user_query)
print("Bot Response:", bot_response)

