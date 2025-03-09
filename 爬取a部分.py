# import os
# import re
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
#
#
# def sanitize_filename(filename):
#     """清理文件名中的非法字符"""
#     return re.sub(r'[\\/*?:"<>|]', '_', filename)
#
#
# # 初始化浏览器驱动（示例使用Chrome）
# driver = webdriver.Chrome()
# driver.get("你的目标网页URL")  # 替换为实际URL
#
# # 等待父容器加载完成
# parent_div = WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.CSS_SELECTOR, "div.cc-cd#node-2413"))
# )
#
# # 提取板块标题（用作文件夹名称）
# section_title = parent_div.find_element(By.CSS_SELECTOR, ".cc-cd-**-st").text
# sanitized_title = sanitize_filename(section_title)
#
# # 创建保存目录
# if not os.path.exists(sanitized_title):
#     os.makedirs(sanitized_title)
#
# # 获取所有新闻条目
# news_items = parent_div.find_elements(By.CSS_SELECTOR, "a[itemid]")
#
# for index, item in enumerate(news_items):
#     try:
#         # 获取当前窗口句柄
#         main_window = driver.current_window_handle
#
#         # 获取链接信息
#         link = item.get_attribute("href")
#         title = item.find_element(By.CSS_SELECTOR, ".t").text
#         sanitized_title = sanitize_filename(title)[:50]  # 限制文件名长度
#
#         # 在新标签页打开链接
#         driver.execute_script("window.open(arguments[0]);", link)
#
#         # 切换到新标签页
#         WebDriverWait(driver, 10).until(
#             lambda d: len(d.window_handles) > 1
#         )
#         new_window = [w for w in driver.window_handles if w != main_window][0]
#         driver.switch_to.window(new_window)
#
#         # 等待正文内容加载（可根据实际页面调整等待条件）
#         WebDriverWait(driver, 15).until(
#             EC.presence_of_element_located((By.TAG_NAME, "body"))
#         )
#
#         # 获取页面所有文本
#         page_text = driver.find_element(By.TAG_NAME, "body").text
#
#         # 保存文件
#         filename = f"{sanitized_title}_{index}.txt"
#         filepath = os.path.join(section_title, filename)
#         with open(filepath, "w", encoding="utf-8") as f:
#             f.write(page_text)
#             print(f"已保存：{filename}")
#
#     except Exception as e:
#         print(f"处理第 {index + 1} 条时出错：{str(e)}")
#
#     finally:
#         # 关闭当前标签页并切换回主窗口
#         if len(driver.window_handles) > 1:
#             driver.close()
#             driver.switch_to.window(main_window)
#
# # 关闭浏览器
# driver.quit()
import requests
from bs4 import BeautifulSoup


def crawl_xinwenlianbo(date):
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


# 使用示例
if __name__ == "__main__":
    date = "20250307"  # 替换
    result = crawl_xinwenlianbo(date)
    print(result)

