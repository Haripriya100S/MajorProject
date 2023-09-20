from bs4 import BeautifulSoup
import requests
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS

url = "https://vnrvjiet.ac.in/admission/"
html_content = requests.get(url).text

# Assuming 'html_content' contains your scraped HTML
soup = BeautifulSoup(html_content, 'html.parser')
text = soup.get_text()

# print(text)


nlp = spacy.load("en_core_web_sm")



# Remove non-alphanumeric characters
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)


# Remove extra whitespaces
text = re.sub(r'\s+', ' ', text).strip()

text = text.lower()
print("text here")
print(text)


doc = nlp(text)
tokens = [token.text for token in doc]

