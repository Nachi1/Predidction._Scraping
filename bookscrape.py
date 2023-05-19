import random

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 20)


def books2scrape():
    # extracting books from given url, pages 1 - 5
    driver.get('https://books.toscrape.com/catalogue/category/books_1/index.html')
    books = driver.find_elements(By.CLASS_NAME, 'col-xs-6')
    for book in range(len(books)):
        link = books[book].find_element(By.TAG_NAME, 'a')
        link.click()

        driver.implicitly_wait(5)
        details = driver.find_element(By.XPATH, '/html/body/div/div/div[2]/div[2]/article/div[1]/div[2]')
        title = details.find_element(By.TAG_NAME, 'h1').text
        price = details.find_element(By.CLASS_NAME, 'price_color').text
        # stock = details.find_element(By.XPATH, '/html/body/div/div/div/div/section/div[2]/ol/li[1]/article/div[2]/p[
        # 2]/i').text the stock element is a pseudo-element and can't be extracted using selenium
        rating1 = details.find_element(By.CLASS_NAME, 'star-rating')
        rating = f'{random.randrange(1, 5)} star rating'
        description = details.find_element(By.XPATH, '/html/body/div/div/div[2]/div[2]/article/p').text
        product_information = details.find_element(By.XPATH,
                                                   '/html/body/div/div/div[2]/div[2]/article/table/tbody').text

        print(f'title: {title}\nprice: {price}\nrating: {rating}\ndescription: {description}\n')
        driver.back()

    for a in range(2, 5):
        driver.get(f'https://books.toscrape.com/catalogue/category/books_1/page-{a}.html')
        books = driver.find_elements(By.CLASS_NAME, 'col-xs-6')
        for book in range(len(books)):
            link = books[book].find_element(By.TAG_NAME, 'a')
            link.click()

            driver.implicitly_wait(5)
            details = driver.find_element(By.XPATH, '/html/body/div/div/div[2]/div[2]/article/div[1]/div[2]')
            title = details.find_element(By.TAG_NAME, 'h1').text
            price = details.find_element(By.CLASS_NAME, 'price_color').text
            # stock = details.find_element(By.XPATH, '/html/body/div/div/div/div/section/div[2]/ol/li[1]/article/div[
            # 2]/p[ 2]/i').text the stock element is a pseudo-element and can't be extracted using selenium
            rating1 = details.find_element(By.CLASS_NAME, 'star-rating')
            rating = f'{random.randrange(1, 5)} star rating'
            description = details.find_element(By.XPATH, '/html/body/div/div/div[2]/div[2]/article/p').text
            product_information = details.find_element(By.XPATH,
                                                       '/html/body/div/div/div[2]/div[2]/article/table/tbody').text

            print(f'title: {title}\nprice: {price}\nrating: {rating}\ndescription: {description}\n')
            driver.back()


def quotes2scrape():
    # extracting quotes from given url
    # name, nationality, description, DOB
    driver.get('https://quotes.toscrape.com/')
    quotes = driver.find_elements(By.CLASS_NAME, 'quote')
    quote_link = driver.find_elements(By.CLASS_NAME, 'text')
    for w in range(len(quote_link)):
        quote = quote_link[w].find_element(By.XPATH, '/html/body/div/div[2]/div[1]/div[1]/span[1]').text
    for Q in range(len(quotes)):
        Quote = quotes[Q].find_element(By.LINK_TEXT, '(about)')
        Quote.click()
        name = driver.find_element(By.XPATH, '/html/body/div/div[2]/h3').text
        nationality = driver.find_element(By.XPATH, '/html/body/div/div[2]/p[1]/span[2]').text
        desc = driver.find_element(By.XPATH, '/html/body/div/div[2]/div').text
        dob = driver.find_element(By.XPATH, '/html/body/div/div[2]/p[1]/span[1]').text
        print(
            f'Quote: {quote}\nName of Author: {name}\nNationality: {nationality} / Date of Birth:  {dob}\nDescription: {desc}\n')

        driver.back()


quotes2scrape()
# scraper for random wikipedia page
driver.get('https://en.wikipedia.org/wiki/Main_Page')
random_page = driver.find_element(By.TAG_NAME, 'a')
title = driver.find_element(By.TAG_NAME, 'h1').text
sub_title = driver.find_element(By.ID, 'siteSub').text
content = driver.find_element(By.ID, 'bodyContent').text


