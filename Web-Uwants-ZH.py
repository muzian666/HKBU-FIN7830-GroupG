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

real_start_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S (HH:MM:SS)")

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

# 指定 chromedriver 的路径
driver_path = 'chromedriver.exe'

# 创建一个新的浏览器窗口
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=options)

columns = ["搜索时页面数", "标题", "URL", "消息内容", "论坛类别", "贴文作者", "日期", "回复", "查看", "最后发表信息"]
df = pd.DataFrame(columns=columns)

# 用户输入搜索内容
search_query = input("请输入您要搜索的内容：")

if os.path.exists(search_query):
    print(f"名称为：'{search_query}' 的文件夹已存在，生成文件将存放于此处。")
else:
    print(f"名称为： '{search_query}' 的文件夹不存在，正在创建并重导向...")
    os.mkdir(search_query)
    print("重导向完成，程序继续运行")

# 访问第一页以获取总页数
first_page_url = f"https://www.uwants.com/search.php?searchsubmit=true&srchtxt={search_query}&orderby=most_relevant&page=1"
driver.get(first_page_url)
random_sleep_time = random.uniform(3, 5)
print(f"随机延迟中，等待页面加载，延迟时长：{random_sleep_time}秒")
time.sleep(random_sleep_time)  # 随机延迟，等待页面加载

# 找到表示总页数的元素并提取其中的数字
try:
    last_page_element = driver.find_element(By.CLASS_NAME, "last")
    last_page_number = int(last_page_element.text.strip().replace('...', '').strip())
except:
    while True:
        try:
            last_page_number = int(input("页面数较少，无法识别，请手动输入总页面数："))
            break
        except ValueError:
            print("输入内容不合法，请输入纯数字")
print(f"总搜索页数为：{last_page_number}")

# 循环遍历所有页
for number in tqdm(range(1, last_page_number + 1), desc="遍历搜索页"):
    driver.get(
        f"https://www.uwants.com/search.php?searchsubmit=true&srchtxt={search_query}&orderby=most_relevant&page={number}")
    random_sleep_time_1 = random.uniform(2, 4)
    print(f"随机延迟中，延迟时间：{random_sleep_time_1}秒")
    time.sleep(random_sleep_time_1)

    main_element = driver.find_element(By.CLASS_NAME, "search-result__results")
    parent_element = main_element.find_element(By.TAG_NAME, "tbody")
    rows = parent_element.find_elements(By.TAG_NAME, "tr")

    for row in tqdm(rows, desc=f"遍历单页内容 {number}"):
        new_row = {"搜索时页面数": number}
        try:
            title_element = row.find_element(By.CSS_SELECTOR, ".search-result-subject a")
            new_row["标题"] = title_element.text
            new_row["URL"] = title_element.get_attribute("href")
        except:
            new_row["标题"] = ""
            new_row["URL"] = ""

        try:
            message_content = row.find_element(By.CSS_SELECTOR, ".search-result-message a").text
            new_row["消息内容"] = message_content
        except:
            new_row["消息内容"] = ""

        try:
            forum_category = row.find_element(By.CSS_SELECTOR, ".search-result-forum a").text
            new_row["论坛类别"] = forum_category
        except:
            new_row["论坛类别"] = ""

        try:
            author = row.find_element(By.CSS_SELECTOR, ".search-result-author a").text
            new_row["贴文作者"] = author
        except:
            new_row["贴文作者"] = ""

        try:
            date = row.find_element(By.CSS_SELECTOR, ".search-result-author .date").text
            date = date.replace('(', '').replace(')', '')
            new_row["日期"] = date
        except:
            new_row["日期"] = ""

        try:
            nums = row.find_element(By.CSS_SELECTOR, ".search-result-nums").text.split(" / ")
            new_row["回复"] = nums[0]
            new_row["查看"] = nums[1]
        except:
            new_row["回复"] = ""
            new_row["查看"] = ""

        try:
            last_post_info = row.find_element(By.CSS_SELECTOR, ".search-result-lastpost em a").text
            new_row["最后发表信息"] = last_post_info
        except:
            new_row["最后发表信息"] = ""

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(new_row)
        # print(f"标题: {title}, URL: {title_url}, 消息内容: {message_content}, 论坛类别: {forum_category}, 作者: {author}, 日期: {date}, 回复: {replies}, 查看: {views}, 最后发表信息: {last_post_info}")

search_info_text_path = os.path.join(search_query, f"Uwants-SearchInfo-{search_query}-{current_time}.csv")
search_info_text = df.to_csv(index=False, encoding='utf-8')
with open(search_info_text_path, "w", encoding='utf-8', newline='') as f:
    f.write("\ufeff" + search_info_text)
print(f"Uwants-SearchInfo-{search_query}-{current_time}.csv 生成完成")
time.sleep(1)

columns = ['URL', '该贴下是否有别的页面', '内部页数', '内部URL', '评论作者', '发布时间', '排序', '主要内容', '是否引用',
           '引用内容']
content = pd.DataFrame(columns=columns)

inner_page_number = 1
inner_url = ""

random_sleep_time_2 = 10

for url in tqdm(df['URL'], desc="遍历单URL"):
    driver.get(url)
    inner_url = url
    while True:
        driver.get(inner_url)
        print("延迟中...")
        print(f"随机延迟中，延迟时间：{random_sleep_time_2}秒")
        time.sleep(random_sleep_time_2)
        random_sleep_time_2 = random.uniform(3, 6)

        rows = driver.find_elements(By.CSS_SELECTOR, ".mainbox.viewthread")
        print(f"在本页中共找到 {len(rows)} 篇post.")

        for row in rows:
            new_row = {"URL": url, "该贴下是否有别的页面": "是" if inner_page_number > 1 else "否",
                       "内部页数": inner_page_number, "内部URL": "" if str(inner_url) == str(url) else inner_url}
            try:
                postauthor_element = row.find_element(By.CSS_SELECTOR, ".postauthor cite a")
                new_row["评论作者"] = postauthor_element.text
            except:
                new_row["评论作者"] = ""

            try:
                maincontent_element = row.find_element(By.CSS_SELECTOR, 'div[id^="postmessage_"]')
                new_row["主要内容"] = maincontent_element.text
            except:
                new_row["主要内容"] = ""

            try:
                public_time = row.find_element(By.CSS_SELECTOR, ".postinfo")
                new_row["发布时间"] = public_time.text.split('發表於')[-1].strip()
            except:
                new_row["发布时间"] = ""

            # try:
            #    title = row.find_element(By.CSS_SELECTOR, ".postmessage.defaultpost h2")
            #    new_row["标题"] = title.text
            # except:
            #    new_row["标题"] = ""

            try:
                order = row.find_element(By.CSS_SELECTOR, '.postinfo > strong')
                new_row["排序"] = order.text
            except:
                new_row["排序"] = ""

            try:
                reference = row.find_elements(By.CSS_SELECTOR, '.quote blockquote')
                later_process = row.find_element(By.CSS_SELECTOR, 'div[id^="postmessage_"]')
                new_row["是否引用"] = "是"
                new_row["引用内容"] = reference[0].text
                new_row["主要内容"] = later_process.text.replace(new_row["引用内容"], "").replace("引用:", "").strip()
            except:
                new_row["是否引用"] = "否"
                new_row["引用内容"] = " "

            print(new_row)
            content = pd.concat([content, pd.DataFrame([new_row])], ignore_index=True)

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, 'div.pages_btns div.pages a.next')
            driver.execute_script("arguments[0].click();", next_button)
            # next_button.click()
            print("进入该贴的下一页...")
            time.sleep(2)
            inner_page_number += 1
            inner_url = driver.current_url
            continue
        except NoSuchElementException:
            print("该帖子无其他内部页面，继续寻找其他帖子...")
            inner_url = ""
            inner_page_number = 1
            break

Content_Info_path = os.path.join(search_query, f"Uwants-Content-{search_query}-{current_time}.csv")
Content_Info = content.to_csv(index=False, encoding='utf-8')
with open(Content_Info_path, "w", encoding='utf-8', newline='') as f:
    f.write("\ufeff" + Content_Info)
print(f"Uwants-Content-{search_query}-{current_time}.csv 生成完成")

driver.quit()

print("准备合并DataFrame....")
merged_df = pd.merge(df, content, on="URL", how="outer")
print("DataFrame合并完成，准备输出文件")
final_information_Info_path = os.path.join(search_query, f"Uwants-{search_query}-{current_time}.csv")
final_information = merged_df.to_csv(index=False, encoding='utf-8')
with open(final_information_Info_path, "w", encoding='utf-8', newline='') as f:
    f.write("\ufeff" + final_information)
print(f"Uwants-{search_query}-{current_time}.csv 生成完成")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"全部完成，总耗时:{elapsed_time}秒")
finished_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S (HH:MM:SS)")
print(f"开始于：{real_start_time} \n完成于：{finished_time}")
