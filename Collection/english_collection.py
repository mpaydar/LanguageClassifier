from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

english_wikipedia = "https://en.wikipedia.org/wiki/Special:Random"

output_file = "English_collected_sentences.txt"  # File to save the collected sentences
sentences_to_collect = 1000  # Desired number of sentences to collect
words_per_page = 150  # Maximum number of words to collect from a single page
desired_sentence_length = 15  # Number of words in each sentence

# Create a Chrome WebDriver instance
driver = webdriver.Chrome()

# Initialize a counter for collected sentences
collected_sentences_count = 0

# Function to split text into words
def split_into_words(text):
    return re.findall(r'\w+', text)

# Function to divide words into sentences of desired length
def divide_words(words, length):
    for i in range(0, len(words), length):
        segment = words[i:i + length]
        if len(segment) == length:
            yield ' '.join(segment)

# Open the output file in write mode
with open(output_file, "w", encoding="utf-8") as file:
    # Keep collecting sentences until the desired count is reached
    while collected_sentences_count < sentences_to_collect:
        # Navigate to a random English Wikipedia page
        driver.get(english_wikipedia)

        # Wait for the page to load and find all paragraphs
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "p")))
        paragraphs = driver.find_elements(By.TAG_NAME, "p")

        collected_words = []
        # Process each paragraph for word extraction
        for paragraph in paragraphs:
            collected_words.extend(split_into_words(paragraph.text))
            if len(collected_words) >= words_per_page:
                collected_words = collected_words[:words_per_page]
                break

        # Break the collected words into sentences of desired length
        for chunk in divide_words(collected_words, desired_sentence_length):
            file.write(chunk + "\n")
            collected_sentences_count += 1
            if collected_sentences_count >= sentences_to_collect:
                break

        if collected_sentences_count >= sentences_to_collect:
            break

# Close the WebDriver when done
driver.quit()
