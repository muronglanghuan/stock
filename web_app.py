from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from agent_workflow import DataFetcher, plot_price, predict_price, llm_summary, NEWS_PROMPT
import base64

app = FastAPI(title="股票智能助手")

@app.get("/", response_class=HTMLResponse)
async def index():
    return (
        "<html><head><meta charset='utf-8'><title>股票助手</title></head>"
        "<body><h3>输入股票代码</h3>"
        "<form action='/predict' method='post'>"
        "代码: <input name='code'/>"
        " 起始日期: <input name='start' value='20230101'/>"
        " 结束日期: <input name='end' value='20231231'/>"
        "<button type='submit'>分析</button>"
        "</form></body></html>"
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(code: str = Form(...), start: str = Form(...), end: str = Form(...)):
    fetcher = DataFetcher()
    df = fetcher.fetch(code, start, end)
    if df.empty:
        return HTMLResponse("<p>未找到数据</p>")
    img_path = plot_price(df, code)
    pred = predict_price(df)
    summary = llm_summary(code, pred, NEWS_PROMPT)
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    html = (
        "<html><head><meta charset='utf-8'><title>分析结果</title></head><body>"
        f"<img src='data:image/png;base64,{img_b64}'/><br/>"
        f"<p>模型预测值: {pred:.4f}</p>"
        f"<pre>{summary}</pre>"
        "<p style='color:red;'>以上内容仅供研究参考, 不构成投资建议</p>"
        "</body></html>"
    )
    return HTMLResponse(html)
