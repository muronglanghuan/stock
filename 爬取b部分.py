import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chardet
import csv
import time
from tqdm import tqdm  # 进度条库，可选安装


def get_news_links(stock_code):
    """爬取指定股票代码的新闻列表"""
    base_url = "https://news.stockstar.com/info/dstock_{}_c10_p{}.shtml"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    page = 1
    news_list = []

    with tqdm(desc="爬取新闻列表") as pbar:
        while True:
            url = base_url.format(stock_code, page)
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # 提取新闻条目
                for content in soup.find_all('div', class_='newslist_content'):
                    items = content.find_all('li')
                    for i in range(0, len(items), 2):
                        if i + 1 >= len(items):
                            continue

                        link_item = items[i].find('a')
                        time_item = items[i + 1]
                        if link_item and 'li_time' in time_item.get('class', []):
                            news_list.append({
                                'title': link_item.get_text(strip=True),
                                'link': link_item['href'],
                                'time': time_item.get_text(strip=True)
                            })

                # 分页处理
                next_page = page + 1
                page_control = soup.find('div', class_='newslist_page')
                if page_control and not page_control.find('a', href=f'dstock_{stock_code}_c10_p{next_page}.shtml'):
                    break
                page = next_page
                pbar.update(1)
                time.sleep(1)

            except Exception as e:
                print(f"\n爬取终止：{str(e)}")
                break

    return news_list


def advanced_crawler(url):
    """高级页面抓取器（自动处理静态/动态内容）"""
    # 第一次尝试静态抓取
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        encoding = chardet.detect(response.content)['encoding'] or 'utf-8'
        response.encoding = encoding
        soup = BeautifulSoup(response.text, 'lxml')
        content_div = soup.find('div', class_='article_content')
        if content_div:
            return content_div.get_text().strip()
    except:
        pass

    # 动态渲染备选方案
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(2)  # 等待页面加载
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()
        content_div = soup.find('div', class_='article_content')
        return content_div.get_text().strip() if content_div else ""
    except:
        return ""


def save_to_file(data, filename, filetype='csv'):
    """保存数据到文件"""
    if filetype == 'csv':
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'link', 'time', 'content'])
            writer.writeheader()
            writer.writerows(data)
    elif filetype == 'txt':
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(
                    f"标题：{item['title']}\n链接：{item['link']}\n时间：{item['time']}\n内容：{item['content']}\n{'=' * 50}\n")


def b(stock_code):
    # 获取新闻列表
    news_list = get_news_links(stock_code)
    print(f"共发现 {len(news_list)} 条新闻")

    # 爬取详细内容
    full_data = []
    for news in tqdm(news_list, desc="抓取新闻内容"):
        # 补全相对链接
        if news['link'].startswith('//'):
            news['link'] = 'https:' + news['link']
        elif news['link'].startswith('/'):
            news['link'] = 'https://news.stockstar.com' + news['link']

        content = advanced_crawler(news['link'])
        full_data.append({**news, 'content': content})
        time.sleep(0.5)  # 控制请求频率

    # 保存结果
    filename = f"{stock_code}_news"
    save_to_file(full_data, filename + '.csv')
    save_to_file(full_data, filename + '.txt', 'txt')
    print(f"数据已保存到 {filename}.[csv/txt]")


if __name__ == "__main__":
    stock_code = "900941"  # 修改为需要的股票代码
    b(stock_code)






