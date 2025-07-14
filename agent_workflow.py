import argparse
import os
from datetime import datetime, timedelta

import akshare as ak
import yfinance as yf
import pandas as pd
import torch
import matplotlib.pyplot as plt
from openai import OpenAI

# 使用黑体以避免中文字符显示异常
plt.rcParams["font.sans-serif"] = ["SimHei"]

from main import TransformerModel, FEATURE_COLS, HISTORY_WINDOW


class DataFetcher:
    """Fetch price data for multiple markets."""

    def fetch(self, code: str, start: str, end: str) -> pd.DataFrame:
        if code.isdigit():
            return self._fetch_cn(code, start, end)
        else:
            return self._fetch_intl(code, start, end)

    def _fetch_cn(self, code: str, start: str, end: str) -> pd.DataFrame:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end)
        df.rename(columns={"日期": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _fetch_intl(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        data = yf.download(ticker, start=start, end=end)
        data.reset_index(inplace=True)
        data.rename(columns={"Date": "date", "Open": "开盘", "Close": "收盘", "High": "最高", "Low": "最低", "Volume": "成交量"}, inplace=True)
        data["成交额"] = data["收盘"] * data["成交量"]
        data["振幅"] = (data["最高"] - data["最低"]) / data["收盘"]
        data["换手率"] = 0.0
        data["涨跌幅"] = data["收盘"].pct_change().fillna(0)
        return data


def plot_price(df: pd.DataFrame, code: str) -> str:
    """绘制并保存价格趋势图，返回图片路径"""
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["收盘"], label="收盘价")
    plt.title(f"{code}价格走势")
    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.legend()
    img_path = f"{code}_trend.png"
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    return img_path


def predict_price(df: pd.DataFrame) -> float:
    if len(df) < HISTORY_WINDOW:
        return 0.0
    model = TransformerModel(len(FEATURE_COLS))
    if os.path.exists("model_checkpoint.pth"):
        state = torch.load("model_checkpoint.pth", map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    features = df[FEATURE_COLS].tail(HISTORY_WINDOW).values
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0)
        pred = model(x).item()
    return pred


def llm_summary(code: str, pred: float, news: str) -> str:
    """调用LLM生成中文投资观点"""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "未提供 OPENAI_API_KEY"
    client = OpenAI(api_key=key)
    prompt = (
        f"你是一名能够给出投资建议的助手。\n股票代码：{code}\n预测值：{pred:.4f}\n新闻摘要：{news}\n"
        "请分别给出短期、中期和长期的简要观点，并注明风险提示。"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


NEWS_PROMPT = (
    "简要汇总最近与该证券相关的重要新闻，20字以内即可。"
)


def main():
    parser = argparse.ArgumentParser(description="智能体工作流")
    parser.add_argument("code", help="股票或其他交易品种代码，如 000001 或 AAPL")
    parser.add_argument("--start", default=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    fetcher = DataFetcher()
    df = fetcher.fetch(args.code, args.start, args.end)
    if df.empty:
        print("未找到任何数据")
        return

    img = plot_price(df, args.code)
    prediction = predict_price(df)

    summary = llm_summary(args.code, prediction, NEWS_PROMPT)

    print("生成的分析报告:\n")
    print(f"趋势图已保存至 {img}")
    print(f"模型预测值: {prediction:.4f}")
    print(summary)
    print("\n以上内容仅供研究参考, 不构成投资建议")


if __name__ == "__main__":
    main()
