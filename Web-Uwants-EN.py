#############################################################################################
## Note: This program is written in Chinese Before, and translate to English without test. ##
#############################################################################################

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
import random
from datetime import datetime
from tqdm import tqdm
import os
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException

real_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (HH:MM:SS)")

start_time = time.time()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# prefs = {
#     "profile.managed_default_content_settings.images": 2,
#     "permissions.default.stylesheet": 2,
# }

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.page_load_strategy = 'none'
# options.add_experimental_option("prefs", prefs)

# set chromedriver path
driver_path = 'chromedriver.exe'

# Create a new browser windows
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=options)

columns = ["Page Number when Search", "Title", "URL", "Message Content", "Category", "Author", "Date", "Reply", "Watch", "Last Time Update"]
df = pd.DataFrame(columns=columns)

search_query = input("Please input the keyword for search：")

if os.path.exists(search_query):
    print(f"The name with：'{search_query}' is exits，the file will save at this place")
else:
    print(f"The name with： '{search_query}' isn't exits，creating and redirecting...")
    os.mkdir(search_query)
    print("Redirecting Finish, Keep Going...")

# Access first page to get total page number
first_page_url = f"https://www.uwants.com/search.php?searchsubmit=true&srchtxt={search_query}&orderby=most_relevant&page=1"
driver.get(first_page_url)
random_sleep_time = random.uniform(3, 5)
print(f"Randomly delay, wating for page to load, delay time：{random_sleep_time}Second")
time.sleep(random_sleep_time)  # Random Delay, waiting for page to load

# Find the element standard for total page and extract the number
try:
    last_page_element = driver.find_element(By.CLASS_NAME, "last")
    last_page_number = int(last_page_element.text.strip().replace('...', '').strip())
except:
    while True:
        try:
            last_page_number = int(input("The page is not enough, can not recognize, please input manually: "))
            break
        except ValueError:
            print("Input illegal, please only input number")
print(f"Total search page number is：{last_page_number}")

# Loop all page
for number in tqdm(range(1, last_page_number + 1), desc="Search every page"):
    driver.get(
        f"https://www.uwants.com/search.php?searchsubmit=true&srchtxt={search_query}&orderby=most_relevant&page={number}")
    random_sleep_time_1 = random.uniform(2, 4)
    print(f"Random delay, delay time：{random_sleep_time_1}秒")
    time.sleep(random_sleep_time_1)

    main_element = driver.find_element(By.CLASS_NAME, "search-result__results")
    parent_element = main_element.find_element(By.TAG_NAME, "tbody")
    rows = parent_element.find_elements(By.TAG_NAME, "tr")

    for row in tqdm(rows, desc=f"Loop Signal Page {number}"):
        new_row = {"Page Number when Search": number}
        try:
            title_element = row.find_element(By.CSS_SELECTOR, ".search-result-subject a")
            new_row["Title"] = title_element.text
            new_row["URL"] = title_element.get_attribute("href")
        except:
            new_row["Title"] = ""
            new_row["URL"] = ""

        try:
            message_content = row.find_element(By.CSS_SELECTOR, ".search-result-message a").text
            new_row["Message Content"] = message_content
        except:
            new_row["Message Content"] = ""

        try:
            forum_category = row.find_element(By.CSS_SELECTOR, ".search-result-forum a").text
            new_row["Category"] = forum_category
        except:
            new_row["Category"] = ""

        try:
            author = row.find_element(By.CSS_SELECTOR, ".search-result-author a").text
            new_row["Author"] = author
        except:
            new_row["Author"] = ""

        try:
            date = row.find_element(By.CSS_SELECTOR, ".search-result-author .date").text
            date = date.replace('(', '').replace(')', '')
            new_row["Date"] = date
        except:
            new_row["Date"] = ""

        try:
            nums = row.find_element(By.CSS_SELECTOR, ".search-result-nums").text.split(" / ")
            new_row["Reply"] = nums[0]
            new_row["Watch"] = nums[1]
        except:
            new_row["Reply"] = ""
            new_row["Watch"] = ""

        try:
            last_post_info = row.find_element(By.CSS_SELECTOR, ".search-result-lastpost em a").text
            new_row["Last Time Update"] = last_post_info
        except:
            new_row["Last Time Update"] = ""

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(new_row)
        # print(f"Title: {title}, URL: {title_url}, Message Content: {message_content}, Category: {forum_category}, 作者: {author}, Date: {date}, Reply: {replies}, Watch: {views}, Last Time Update: {last_post_info}")

search_info_text_path = os.path.join(search_query, f"Uwants-SearchInfo-{search_query}-{current_time}.csv")
search_info_text = df.to_csv(index=False, encoding='utf-8')
with open(search_info_text_path, "w", encoding='utf-8', newline='') as f:
    f.write("\ufeff" + search_info_text)
print(f"Uwants-SearchInfo-{search_query}-{current_time}.csv Generate Finished")
time.sleep(1)

columns = ['URL', 'Whether have other page under this title', 'Inner Page', 'Inner URL', 'Comment Author', 'Publish Time', 'Order', 'Main Content', 'Whether Reference',
           'Reference Content']
content = pd.DataFrame(columns=columns)

inner_page_number = 1
inner_url = ""

random_sleep_time_2 = 10

for url in tqdm(df['URL'], desc="Loop For Single URL"):
    driver.get(url)
    inner_url = url
    while True:
        driver.get(inner_url)
        print("Delay...")
        print(f"Random Delay，Delay Time：{random_sleep_time_2}Second")
        time.sleep(random_sleep_time_2)
        random_sleep_time_2 = random.uniform(3, 6)

        rows = driver.find_elements(By.CSS_SELECTOR, ".mainbox.viewthread")
        print(f"Totally found {len(rows)} post.")

        for row in rows:
            new_row = {"URL": url, "Whether have other page under this title": "Yes" if inner_page_number > 1 else "No",
                       "Inner Page": inner_page_number, "Inner URL": "" if str(inner_url) == str(url) else inner_url}
            try:
                postauthor_element = row.find_element(By.CSS_SELECTOR, ".postauthor cite a")
                new_row["Comment Author"] = postauthor_element.text
            except:
                new_row["Comment Author"] = ""

            try:
                maincontent_element = row.find_element(By.CSS_SELECTOR, 'div[id^="postmessage_"]')
                new_row["Main Content"] = maincontent_element.text
            except:
                new_row["Main Content"] = ""

            try:
                public_time = row.find_element(By.CSS_SELECTOR, ".postinfo")
                new_row["Publish Time"] = public_time.text.split('Publish at:')[-1].strip()
            except:
                new_row["Publish Time"] = ""

            # try:
            #    title = row.find_element(By.CSS_SELECTOR, ".postmessage.defaultpost h2")
            #    new_row["Title"] = title.text
            # except:
            #    new_row["Title"] = ""

            try:
                order = row.find_element(By.CSS_SELECTOR, '.postinfo > strong')
                new_row["Order"] = order.text
            except:
                new_row["Order"] = ""

            try:
                reference = row.find_elements(By.CSS_SELECTOR, '.quote blockquote')
                later_process = row.find_element(By.CSS_SELECTOR, 'div[id^="postmessage_"]')
                new_row["Whether Reference"] = "Yes"
                new_row["Reference Content"] = reference[0].text
                new_row["Main Content"] = later_process.text.replace(new_row["Reference Content"], "").replace("Reference:", "").strip()
            except:
                new_row["Whether Reference"] = "No"
                new_row["Reference Content"] = " "

            print(new_row)
            content = pd.concat([content, pd.DataFrame([new_row])], ignore_index=True)

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, 'div.pages_btns div.pages a.next')
            driver.execute_script("arguments[0].click();", next_button)
            # next_button.click()
            print("Enter to the Next Page...")
            time.sleep(2)
            inner_page_number += 1
            inner_url = driver.current_url
            continue
        except NoSuchElementException:
            print("This post don't have inner page, keep looking for other post...")
            inner_url = ""
            inner_page_number = 1
            break

Content_Info_path = os.path.join(search_query, f"Uwants-Content-{search_query}-{current_time}.csv")
Content_Info = content.to_csv(index=False, encoding='utf-8')
with open(Content_Info_path, "w", encoding='utf-8', newline='') as f:
    f.write("\ufeff" + Content_Info)
print(f"Uwants-Content-{search_query}-{current_time}.csv Generate Finish")

driver.quit()

print("Ready For Combine DataFrame....")
merged_df = pd.merge(df, content, on="URL", how="outer")
print("DataFrame Combine Finished，Ready for Output")
final_information_Info_path = os.path.join(search_query, f"Uwants-{search_query}-{current_time}.csv")
final_information = merged_df.to_csv(index=False, encoding='utf-8')
with open(final_information_Info_path, "w", encoding='utf-8', newline='') as f:
    f.write("\ufeff" + final_information)
print(f"Uwants-{search_query}-{current_time}.csv Generate Finished")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"All Done，Total Cost:{elapsed_time}Second")
finished_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (HH:MM:SS)")
print(f"Start at：{real_start_time} \nFinished at：{finished_time}")
