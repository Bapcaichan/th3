import torch
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.metrics import RMSE
from sklearn.metrics import mean_squared_error
# Đọc dữ liệu
data = pd.read_csv("C:/Users/ADMIN/2025_PC/datahub/energy2.csv")

# Chuyển cột 'time' thành datetime và xử lý múi giờ
data['time'] = pd.to_datetime(data['time'], utc=True).dt.tz_convert(None)

# Nội suy tuyến tính cho các cột đặc trưng
features = data[["total load actual", "generation solar", "generation wind onshore", "generation fossil gas"]]
features = features.replace('', np.nan).interpolate(method='linear', limit_direction='both')
data[["total load actual", "generation solar", "generation wind onshore", "generation fossil gas"]] = features

# Kiểm tra giá trị thiếu
print("Số giá trị thiếu sau nội suy:")
print(pd.isna(features).sum())

# Phân cụm bằng GMM
gmm = GaussianMixture(n_components=3, random_state=100)
data["cluster"] = gmm.fit_predict(features)
data["cluster_probs"] = gmm.predict_proba(features).tolist()

# Thêm đặc trưng thời gian
data["hour"] = data["time"].dt.hour
data["day_of_week"] = data["time"].dt.dayofweek

# Đảm bảo thời gian liên tục
data = data.sort_values('time')
full_time_index = pd.date_range(start=data['time'].min(), end=data['time'].max(), freq='h')
full_time_df = pd.DataFrame({'time': full_time_index})
data = full_time_df.merge(data, on='time', how='left')
data[["total load actual", "generation solar", "generation wind onshore", "generation fossil gas"]] = data[["total load actual", "generation solar", "generation wind onshore", "generation fossil gas"]].interpolate(method='linear', limit_direction='both')
data["hour"] = data["time"].dt.hour
data["day_of_week"] = data["time"].dt.dayofweek
data["cluster"] = data["cluster"].fillna(0).astype(int)
data = data.reset_index().rename(columns={'index': 'index'})


# Tạo TimeSeriesDataSet
dataset = TimeSeriesDataSet(
    data,
    time_idx="index",
    target="total load actual",
    group_ids=["cluster"],
    time_varying_known_reals=["hour", "day_of_week", "generation solar", "generation wind onshore", "generation fossil gas"],
    time_varying_unknown_reals=["total load actual"],
    max_encoder_length=24,
    max_prediction_length=24,
    allow_missing_timesteps=True
)

# Tạo DataLoader
dataloader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)

# Định nghĩa quantiles cho QuantileLoss
quantiles = [0.1, 0.5, 0.9]  # 3 quantiles để giảm tải tính toán
n_quantiles = len(quantiles)

# Khởi tạo QuantileLoss với quantiles cụ thể
quantile_loss = QuantileLoss(quantiles=quantiles)

# Khởi tạo TFT
tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=n_quantiles,  # Phải khớp với số quantiles
    loss=quantile_loss,
)


# Khởi tạo Trainer
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="cpu",
    enable_progress_bar=True,
    logger=False
)

# Huấn luyện mô hình
trainer.fit(tft, train_dataloaders=dataloader)

split_index = int(len(data) * 0.8)
training_data = data.iloc[:split_index]
testing_data = data.iloc[split_index:]

# Tạo dataset test
test_dataset = TimeSeriesDataSet(
    testing_data,
    time_idx="index",
    target="total load actual",
    group_ids=["cluster"],
    time_varying_known_reals=["hour", "day_of_week", "generation solar", "generation wind onshore", "generation fossil gas"],
    time_varying_unknown_reals=["total load actual"],
    max_encoder_length=24,
    max_prediction_length=24,
    allow_missing_timesteps=True
)

# Dataloader test
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

predictions = tft.predict(test_dataloader, mode="prediction")

print("Shape of predictions:", predictions.shape)

actuals = torch.cat([y[0] for x, y in iter(test_dataloader)], dim=0).numpy()

# Lấy median prediction (quantile 0.5)
median_predictions = predictions.reshape(-1).numpy()

# Cắt cho đúng chiều (do padding)
min_len = min(len(actuals), len(median_predictions))
actuals = actuals[:min_len]
median_predictions = median_predictions[:min_len]

# Tính RMSE
rmse = mean_squared_error(actuals, median_predictions, squared=False)
print(f"RMSE trên tập test: {rmse:.2f}")
