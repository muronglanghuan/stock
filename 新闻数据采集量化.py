# import akshare as ak
#
# # 获取A股所有上市公司的代码和名称
# stock_info_a_code_name_df = ak.stock_info_a_code_name()
#
# # 指定日期范围
# start_date = "20000101"
# end_date = "20250101"
#
# # 循环获取每一家公司的股票数据并保存
# for index, row in stock_info_a_code_name_df.iterrows():
#     stock_code = row['code']
#     stock_name = row['name']
#     file_name = f"{stock_code}.csv"
#
#     try:
#         stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date)
#         stock_data.to_csv(f"data/{file_name}", index=False)
#         print(f"{index}-{stock_name}（{stock_code}）的数据已成功保存到 {file_name}")
#     except Exception as e:
#         print(f"获取 {index}-{stock_name}（{stock_code}）的数据时出现错误：{str(e)}")
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
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
    return full_data
import requests
from bs4 import BeautifulSoup


def a(date):
    """
    爬取新闻联播指定日期的文字内容
    参数：
        date: 日期字符串，格式为YYYYMMDD（如20230307）
    返回：
        包含所有新闻内容的字符串（若爬取失败返回错误信息）
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    url = f"https://cn.govopendata.com/xinwenlianbo/{date}/"

    try:
        # 发送HTTP请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查HTTP状态码

        # 解析网页内容
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main', class_='news-content')

        if not main_content:
            return "未找到新闻内容，请检查日期格式或网页结构是否变化"

        # 提取所有文章
        articles = main_content.find_all('article', class_='mb-4')
        news_text = []

        for article in articles:
            # 提取标题
            title = article.find('h2', class_='h4').get_text(strip=True)
            # 提取正文
            content = article.find('p', class_='text-justify').get_text(strip=True)
            news_text.append(f"{title}\n{content}")

        return '\n\n'.join(news_text) if news_text else "当日无新闻内容"

    except requests.exceptions.RequestException as e:
        return f"网络请求失败: {str(e)}"
    except Exception as e:
        return f"处理时发生错误: {str(e)}"

def get_stock_text(stock_code):
    url = f"https://stock.quote.stockstar.com/corp_{stock_code}.shtml"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查HTTP状态码

        # 检测页面编码（部分页面可能使用gbk编码）
        if response.encoding == 'ISO-8859-1':
            response.encoding = 'gbk' if 'charset=gbk' in response.text.lower() else 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')
        main_div = soup.find('div', id='sta_3')

        if not main_div:
            return f"未找到股票代码 {stock_code} 的公司资料内容，请确认代码有效性"

        # 提取所有文本并清理
        text = main_div.get_text(separator='\n', strip=True)
        # 去除多余空行
        cleaned_text = '\n'.join([line for line in text.split('\n') if line.strip()])

        return cleaned_text

    except requests.exceptions.RequestException as e:
        return f"请求失败: {str(e)}"
    except Exception as e:
        return f"处理时发生错误: {str(e)}"
import re

def extract_abc(text):
    """
    从文本中提取由两个分号间隔的3个小数
    返回包含abc的元组或None（格式示例：1.23;-4.56;7.89）
    """
    pattern = r"""
        (-?            # 负号（可选）
        \d+            # 整数部分
        (?:\.\d+)?)    # 小数部分（可选）
        \s*;\s*        # 分号间隔（允许空格）
        (-?\d+(?:\.\d+)?)
        \s*;\s*
        (-?\d+(?:\.\d+)?)
    """
    match = re.search(pattern, text, re.VERBOSE)
    return match.groups() if match else None
def jibenmian(code,date):
    aa=a(date)
    # bb=b(code)
    cc=str(get_stock_text(code))
    text=f'请执行事件影响量化分析：\n[分析规则]\n1. 基于特征矩阵的三维度分析：\n   - 短期：事件关键词匹配度（0.3权重）+情感倾向（0.7权重）\n   - 中期：行业趋势一致性（0.6权重）+竞争格局影响（0.4权重）\n   - 长期：政策方向契合度（0.5权重）+技术替代风险（0.5权重）\n\n2. 输出格式严格遵循：三个浮点数用分号分隔，范围[-1,1]\n[输出示例]\n0.7;-0.3;0.2\n'

    # client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="")
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")#填入自己的api
    # 第一步：使用 deepseek-r1-zero 进行初步处理
    prompt_1 ="请执行新闻经济价值提取：\n"
    "[处理规则]\n"
    "1. 先分类后提炼，输出格式：分类标签 | 关键信息\n"
    "2. 分类标准：\n"
    "   1-重大经济事件（直接影响上市公司）\n"
    "   2-行业/区域政策\n"
    "   3-企业运营动态\n"
    "   4-无关信息\n"
    "3. 提炼要求：\n"
    "   - 保留金额/百分比/时间/政策条款\n"
    "   - 过滤非经济内容"
    response_1 = client.chat.completions.create(
        # model="deepseek/deepseek-r1-zero:free",
        model="deepseek-chat",
        # messages=[{"role": "user", "content": prompt_1 + aa}]
        messages=[{"role": "system", "content": prompt_1},{"role": "user", "content":aa}]

    )
    processed_text = response_1.choices[0].message.content
    print(processed_text)
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")#填入自己的api

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": text},
            {"role": "user", "content": processed_text+cc},


        ],
        stream=False,
        temperature=0
    )
    p=response.choices[0].message.content
    print(p)
    a1,a2,a3=extract_abc(p)
    return a1,a2,a3

print(jibenmian('000001','20100512'))
date='20100512'
code='000001'
aa=a(date)
# bb=b(code)
cc=str(get_stock_text(code))
print(aa)
# print(bb)
print(cc)