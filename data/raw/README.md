# 原始数据放置说明

## Home Credit Default Risk

推荐目录：

```text
data/raw/home_credit/
  application_train.csv
  application_test.csv
  bureau.csv
  bureau_balance.csv
  credit_card_balance.csv
  installments_payments.csv
  POS_CASH_balance.csv
  previous_application.csv
```

当前第 1-2 周流程优先使用 `application_train.csv` 建立主表基线，其他表后续用于扩展特征工程。

获取方式：

```powershell
kaggle competitions download -c home-credit-default-risk -p data/raw/home_credit
```

如果没有 Kaggle API 凭据，可以使用任一方式：

- 运行 `kaggle auth login`，按网页流程授权。
- 在 Kaggle 设置页生成 API token，并将 token 写入环境变量 `KAGGLE_API_TOKEN`。
- 将 token 文件放入 `data/raw/kaggle_config/`，再设置 `KAGGLE_CONFIG_DIR=data/raw/kaggle_config`。

本项目脚本会优先使用项目内的 `data/raw/kaggle_config/`，避免写入用户目录时遇到权限问题。

## UCI Default of Credit Card Clients

推荐文件：

```text
data/raw/taiwan/default of credit card clients.xls
```

可以运行：

```powershell
py src/download_data.py --uci-only
```

该数据用于小规模验证和流程联调。
