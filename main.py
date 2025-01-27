# main.py
from stock_prediction import StockPredictor

def main():
    # Model parametrelerini ayarlayın
    symbol = 'AAPL'  # Tahmin edilecek hisse senedi
    start_date = '2020-01-01'  # Başlangıç tarihi
    sequence_length = 60  # Giriş dizisi uzunluğu

    # Modeli oluşturun ve çalıştırın
    predictor = StockPredictor(
        symbol=symbol,
        start_date=start_date,
        sequence_length=sequence_length
    )
    
    # Pipeline'ı çalıştırın
    metrics, predictions, history = predictor.run_pipeline()
    
    # Sonuçları yazdırın
    print("\nDetaylı Sonuçlar:")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']*100:.2f}%")
    print(f"R² Score: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()