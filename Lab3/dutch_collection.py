from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re





dutch_wikipedia = "https://nl.wikipedia.org/wiki/Speciaal:Willekeurige_pagina"
temp_storage=""
driver=webdriver.Chrome()
driver.get(dutch_wikipedia)
while len(temp_storage)<=15000:
    menu=driver.find_element(By.CLASS_NAME,'vector-main-menu-landmark')
    menu.click()
    random_article=driver.find_element(By.ID,'n-randompage')
    article=random_article.click()
    paragraphs=driver.find_element(By.TAG_NAME,'p')
    # print(paragraphs.text)
    temp_storage+=paragraphs.text
# print(temp_storage)


def edit_stored_text(text_string):
    with open("dutchText.txt", "w", encoding="utf-8") as file:
        words = text_string.split()  # Split the string into words
        temp_text = ""
        word_count = 0

        for word in words:
            temp_text += word + ' '
            word_count += 1

            if word_count == 15:
                file.write(temp_text.strip() + '\n')  # Write the line to the file
                temp_text = ""  # Reset the temporary string
                word_count = 0  # Reset the word count

        # Write any remaining text that didn't make up a full 15 words
        if temp_text:
            file.write(temp_text.strip() + '\n')

# Example usage
edit_stored_text(temp_storage)



# print(len(temp_storage)/15)
# print(temp_storage)
print(temp_storage)
edit_stored_text(temp_storage)