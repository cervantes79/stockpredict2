# Stock Price Prediction with LSTM

This project implements a deep learning model for stock price prediction using LSTM networks. The model is designed to analyze historical stock data and predict future price movements.

## Requirements

### Python Version
- Python 3.9 or higher

### Required Packages
```
pandas==2.2.3
tensorflow==2.18.0
yfinance==0.2.52
scikit-learn==1.6.1
matplotlib==3.9.4
ta==0.11.0
seaborn==0.13.2
statsmodels==0.14.4
keras-tuner==1.4.7
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cervantes79/stockpredict2.git
cd stock-prediction2
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
stock-prediction/
├── main.py
├── stock_prediction.py
├── requirements.txt
├── models/
├── data/
└── reports/
```

## How to Run

1. Basic usage with default parameters:
```bash
python main.py
```

2. Custom configuration:
```python
from stock_prediction import StockPredictor

predictor = StockPredictor(
    symbol='AAPL',           # Stock symbol
    start_date='2020-01-01', # Start date for training
    sequence_length=60       # Sequence length for LSTM
)

metrics, predictions, history = predictor.run_pipeline()
```

## Configuration Options

You can modify these parameters in the `Config` class:
- `SYMBOL`: Stock symbol (default: 'AAPL')
- `TIMEFRAME`: Data timeframe (default: '1d')
- `SEQUENCE_LENGTH`: Number of days for sequence (default: 60)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Maximum training epochs (default: 50)

## Output

The model will generate:
1. Trained model files in the `models/` directory
2. Performance metrics and predictions
3. Training visualization plots
4. Detailed analysis report in markdown format

## Troubleshooting

Common issues and solutions:
1. If you get a memory error, try reducing the sequence length or batch size
2. For GPU support, ensure you have CUDA installed
3. For data download issues, check your internet connection

