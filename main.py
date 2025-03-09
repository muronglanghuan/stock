import os
import signal
import sys
import time
from datetime import datetime, timedelta
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import akshare as ak

# 配置参数
HISTORY_WINDOW = 30  # 历史窗口大小
BATCH_SIZE = 512
EMBEDDING_DIM = 64  # 修正后的正确变量名
NUM_HEADS = 4
NUM_LAYERS = 3
LEARNING_RATE = 1e-4
GAMMA = 0.99  # 折扣因子
FEATURE_COLS = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '换手率']

class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM, nhead=NUM_HEADS, dim_feedforward=4*EMBEDDING_DIM
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=NUM_LAYERS)
        self.embedding = nn.Linear(input_dim, EMBEDDING_DIM)
        self.fc = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 32),  # 修正拼写错误
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq, batch, features)
        x = self.transformer(x)
        x = x[-1]  # 取最后一个时间步
        return self.fc(x).squeeze()

class StockDataset(Dataset):
    def __init__(self, data_dict, current_date):
        self.samples = []
        for code, df in data_dict.items():
            if current_date not in df.index:  # 添加日期存在性检查
                continue
            try:
                idx = df.index.get_loc(current_date)
                if idx < HISTORY_WINDOW: 
                    continue
                features = df.iloc[idx-HISTORY_WINDOW:idx][FEATURE_COLS].values
                target = df.iloc[idx]['涨跌幅']
                self.samples.append((features, target, code))
            except KeyError:
                continue
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, target, code = self.samples[idx]
        return torch.FloatTensor(features), torch.FloatTensor([target]), code
def stock_hist(code,end_date):
    df = pd.read_csv(os.path.join('data', f'{code}.csv'))
    df['日期'] = pd.to_datetime(df['日期'])
    df = df[df['日期'] <= end_date]
    return df
class DataFetcher:
    def __init__(self,online):
        self.trade_dates = self._get_trade_dates()
        stock_info = ak.stock_info_a_code_name()
        self.stock_list = stock_info['code'].tolist() if not stock_info.empty else []
        self.online=online
    def _get_trade_dates(self):
        df = ak.tool_trade_date_hist_sina()
        return pd.to_datetime(df['trade_date']).dt.date.tolist()
    
    def fetch_single(self, code, end_date):
        try:
            if self.online :
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20000101", end_date=end_date)
            else:
                df=stock_hist(code,end_date)

            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期']).dt.date
                df = df.set_index('日期')
                # 添加特征工程
                df['成交量'] = np.log(df['成交量'] + 1e-6)  # 对数转换
                df['成交额'] = np.log(df['成交额'] + 1e-6)
                return code, df
            return code, None
        except Exception as e:
            print(f"获取{code}数据失败：{str(e)}")
            return code, None

class RLFramework:
    def __init__(self, start_date, continue_train=False,online=Ture):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fetcher = DataFetcher(online=online)
        self.current_date = self._validate_date(start_date)
        self.data_cache = {}

        
        self.model = TransformerModel(len(FEATURE_COLS)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        if continue_train and os.path.exists('model.pth'):
            self.load_model()
            
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def _validate_date(self, date_str):
        try:
            target = pd.to_datetime(date_str).date()
            return min([d for d in self.fetcher.trade_dates if d >= target])
        except:
            return None
    
    def _update_data_cache(self):
        end_date = self.current_date.strftime("%Y%m%d")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(self.fetcher.fetch_single, 
                                 [(code, end_date) for code in self.fetcher.stock_list])
            
        for code, df in results:
            if df is not None and not df.empty:
                self.data_cache[code] = df[FEATURE_COLS + ['涨跌幅']]
                
    def _preprocess(self):
        for code, df in self.data_cache.items():
            # 添加滚动标准化
            for col in FEATURE_COLS:
                df[col] = (df[col] - df[col].rolling(30).mean()) / (df[col].rolling(30).std() + 1e-6)
            df.dropna(inplace=True)
            
    def train_one_day(self):
        dataset = StockDataset(self.data_cache, self.current_date)
        if len(dataset) == 0: 
            print("无有效训练数据")
            return 0
            
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        total_loss = 0
        
        for features, targets, _ in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 添加梯度裁剪
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)
    
    def evaluate(self):
        predictions = []
        with torch.no_grad():
            for code, df in self.data_cache.items():
                if self.current_date not in df.index: continue
                try:
                    idx = df.index.get_loc(self.current_date)
                    features = df.iloc[idx-HISTORY_WINDOW:idx][FEATURE_COLS].values
                    features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    pred = self.model(features).item()
                    real = df.iloc[idx]['涨跌幅']
                    predictions.append((code, pred, real))
                except:
                    continue
                    
        # 添加评估指标
        df = pd.DataFrame(predictions, columns=['code', 'pred', 'real'])
        if df.empty:
            return df
        df['error'] = (df['pred'] - df['real']).abs()
        df = df.sort_values(['pred', 'real'], ascending=[False, False])
        return df.head(10)
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'date': self.current_date
        }, 'model.pth')
        
    def load_model(self):
        checkpoint = torch.load('model.pth')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_date = checkpoint['date']
        
    def signal_handler(self, signum, frame):
        print("\n中断训练，保存模型中...")
        self.save_model()
        sys.exit(0)
        
    def run(self):
        while True:
            start_time = time.time()
            print(f"\n训练日期：{self.current_date}")
            
            # 更新数据
            self._update_data_cache()
            self._preprocess()
            
            # 训练
            loss = self.train_one_day()
            print(f"训练损失：{loss:.4f}")
            
            # 评估
            top10 = self.evaluate()
            if not top10.empty:
                print("今日Top10推荐：")
                print(top10)
            else:
                print("无有效评估数据")
            
            # 保存模型
            self.save_model()
            
            # 移动到下个交易日
            try:
                idx = self.fetcher.trade_dates.index(self.current_date)
                if idx + 1 >= len(self.fetcher.trade_dates):
                    print("已训练到最新日期")
                    break
                self.current_date = self.fetcher.trade_dates[idx + 1]
            except ValueError:
                print("日期索引错误")
                break
            
            # 控制训练速度
            elapsed = time.time() - start_time
            if elapsed < 1: 
                time.sleep(1 - elapsed)

def main(start_date="20000101", continue_train=False,online=Ture):
    framework = RLFramework(start_date, continue_train,online=Ture)
    if framework.current_date is None:
        print("无效的起始日期")
        return
    framework.run()

if __name__ == "__main__":
    # 在这里设置起始日期和是否继续训练
    main(start_date="20030515", continue_train=True,online=Ture)
