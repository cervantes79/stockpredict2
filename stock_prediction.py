import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from keras_tuner import RandomSearch, HyperParameters
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import json



class StockPredictor:
    def __init__(self, symbol='AAPL', start_date='2014-01-01', sequence_length=2000):
        self.symbol = symbol
        self.start_date = start_date  # 10 yıllık veri için 2014'ten başlıyoruz
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()  # MinMaxScaler yerine StandardScaler kullanacağız
        self.model = None
        self.tuner = None 

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            print(f"Fetching data for {self.symbol} from {self.start_date}")
            df = yf.download(self.symbol, start=self.start_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Index'i datetime'a çevir
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

    def build_tunable_model(self, hp, input_shape):
        model = Sequential()
        
        # First LSTM Layer
        model.add(LSTM(
            units=hp.Int('lstm_1_units', 128, 256, step=32),
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(LayerNormalization())
        model.add(Dropout(hp.Float('dropout_1', 0.2, 0.3)))
        
        # Second LSTM Layer
        model.add(LSTM(
            units=hp.Int('lstm_2_units', 64, 128, step=32),
            return_sequences=True
        ))
        model.add(LayerNormalization())
        model.add(Dropout(hp.Float('dropout_2', 0.2, 0.3)))
        
        # Third LSTM Layer
        model.add(LSTM(
            units=hp.Int('lstm_3_units', 32, 64, step=16),
            return_sequences=False
        ))
        model.add(LayerNormalization())
        model.add(Dropout(hp.Float('dropout_3', 0.1, 0.2)))
        
        # Dense Layers
        model.add(Dense(
            units=hp.Int('dense_1_units', 16, 32, step=8),
            activation='relu'
        ))
        model.add(Dropout(hp.Float('dropout_dense', 0.1, 0.2)))
        
        # Output
        model.add(Dense(1))
        
        # Learning rate with wider range and finer sampling
        lr = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
        
        optimizer = Adam(learning_rate=lr)
        
        model.compile(
            optimizer=optimizer,
            loss=Huber(delta=hp.Float('huber_delta', 0.1, 1.0)), 
            metrics=['mae', 'mse']
        )
        return model

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Hyperparameter optimization"""
        try:
            # Clear any previous tuner data
            if os.path.exists('hyperparameter_tuning'):
                import shutil
                shutil.rmtree('hyperparameter_tuning')
            
            tuner = RandomSearch(
                lambda hp: self.build_tunable_model(hp, (X_train.shape[1], X_train.shape[2])),
                objective='val_loss',  # Minimize validation loss
                max_trials=4,  # Deneme sayısı
                executions_per_trial=5,
                directory='hyperparameter_tuning',
                project_name=f'{self.symbol}_optimization'
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping]
            )
            
            # En iyi hyperparametreleri al
            self.best_hp = tuner.get_best_hyperparameters()[0]
            
            # En iyi modeli oluştur
            self.model = self.build_tunable_model(
                self.best_hp,
                (X_train.shape[1], X_train.shape[2])
            )
            
            return self.best_hp
            
        except Exception as e:
            print(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        try:
            # Veriyi 1-dimensional yap
            close_prices = df['Close'].squeeze()
            
            # RSI
            df['RSI'] = ta.momentum.rsi(close_prices, window=14)
            
            # MACD
            df['MACD'] = ta.trend.macd(close_prices, window_slow=26, window_fast=12)
            df['MACD_Signal'] = ta.trend.macd_signal(close_prices, window_slow=26, window_fast=12, window_sign=9)
            
            # Bollinger Bands
            df['BB_High'] = ta.volatility.bollinger_hband(close_prices, window=20, window_dev=2)
            df['BB_Low'] = ta.volatility.bollinger_lband(close_prices, window=20, window_dev=2)
            df['BB_Mid'] = ta.volatility.bollinger_mavg(close_prices, window=20)
            
            return df
            
        except Exception as e:
            print(f"Error in technical indicators calculation: {str(e)}")
            print(f"Close shape before squeeze: {df['Close'].shape}")
            print(f"Close shape after squeeze: {df['Close'].squeeze().shape}")
            raise

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        try:
            if self.model is None:
                raise ValueError("Model has not been initialized. Run hyperparameter optimization first.")
                
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            return history
                
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
        
    def prepare_data(self, df):
        try:
            # DataFrame'i doğru formata getir
            df = df.copy()
            
            # Teknik göstergeleri hesapla
            rsi = ta.momentum.RSIIndicator(df['Close'].squeeze())  # squeeze() ile 1-boyutlu yap
            df['RSI'] = rsi.rsi() / 100
            
            macd = ta.trend.MACD(df['Close'].squeeze())
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            bb = ta.volatility.BollingerBands(df['Close'].squeeze())
            df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            # Getiriler ve hedef
            df['return'] = df['Close'].pct_change()
            df['target'] = df['return'].shift(-1)
            
            # NaN değerleri temizle
            df = df.dropna()
            
            # Kullanılacak özellikler
            features = ['return', 'RSI', 'MACD', 'BB_Width']
            
            # Sequence oluştur
            X, y = [], []
            for i in range(len(df) - self.sequence_length):
                X.append(df[features].iloc[i:(i + self.sequence_length)].values)
                y.append(df['target'].iloc[i + self.sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # Son fiyatları sakla
            self.last_prices = df['Close'].values[self.sequence_length:]
            
            # Train/validation/test split
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.15)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            print("DataFrame shape:", df.shape)
            print("Close column shape:", df['Close'].shape)
            raise


    def evaluate_model(self, X_test, y_test):
        try:
            # Model tahminleri (returns)
            predicted_returns = self.model.predict(X_test).flatten()
            
            # Returns'u fiyatlara çevir
            last_prices = self.last_prices[-len(y_test):]
            predicted_prices = last_prices * (1 + predicted_returns)
            actual_prices = last_prices * (1 + y_test)
            
            # Yön tahmini
            direction_accuracy = np.mean((predicted_returns > 0) == (y_test > 0))
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(actual_prices, predicted_prices)),
                'mae': mean_absolute_error(actual_prices, predicted_prices),
                'mape': mean_absolute_percentage_error(actual_prices, predicted_prices),
                'r2': r2_score(actual_prices, predicted_prices),
                'direction_accuracy': direction_accuracy
            }
            
            return metrics, predicted_prices
                
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            raise


    def plot_results(self, y_test, predictions):
        """Plot actual vs predicted values"""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.title(f'{self.symbol} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            
            # Save plot
            os.makedirs('reports', exist_ok=True)
            plt.savefig(f'reports/{self.symbol}_predictions.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
            raise

    def save_model(self, model_path='models'):
        """Save the trained model"""
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(model_path, f'{self.symbol}_model_{timestamp}.h5')
            self.model.save(model_file)
            
            # Hyperparameters'ı da kaydet
            if hasattr(self, 'best_hp'):
                hp_file = os.path.join(model_path, f'{self.symbol}_hyperparameters_{timestamp}.json')
                with open(hp_file, 'w') as f:
                    json.dump(self.best_hp.values, f)
                    
            print(f"Model saved to {model_file}")
            return model_file
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    def generate_report(self, metrics, history, hyperparameters=None, save_path='reports'):
        """Generate enhanced report"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Training history plot
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_path = os.path.join(save_path, f'{self.symbol}_loss_plot.png')
            plt.savefig(loss_plot_path)
            plt.close()
            
            # Report content
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_content = f"""# Enhanced Stock Price Prediction Report
    ## Model Analysis for {self.symbol}

    Generated on: {timestamp}

    ### Model Configuration
    - Stock Symbol: {self.symbol}
    - Training Start Date: {self.start_date}
    - Sequence Length: {self.sequence_length}
    - Architecture: Deep LSTM Network

    ### Best Hyperparameters
    """
            if hyperparameters:
                for param, value in hyperparameters.items():
                    report_content += f"- {param}: {value}\n"

            report_content += f"""
    ### Model Performance Metrics
    - RMSE: ${metrics['rmse']:.2f}
    - MAE: ${metrics.get('mae', 0):.2f}
    - R² Score: {metrics.get('r2', 0):.4f}

    ### Training Details
    - Number of Epochs: {len(history.history['loss'])}
    - Final Training Loss: {history.history['loss'][-1]:.4f}
    - Final Validation Loss: {history.history['val_loss'][-1]:.4f}

    ### Visualizations
    ![Training Loss Plot]('{self.symbol}_loss_plot.png')

    ### Model Architecture
    {self.model.summary()}
    ### Training Data Analysis
- Total Training Samples: {metrics.get('n_train', 'N/A')}
- Training Period: {metrics.get('train_period', 'N/A')}
- Features Used: {metrics.get('features', 'N/A')}

### Recommendations
1. Consider adjusting hyperparameters for better performance
2. Experiment with different feature combinations
3. Collect more historical data if available
4. Monitor model performance over time
            """
        
            # Save report
            report_path = os.path.join(save_path, f'{self.symbol}_prediction_report.md')
            with open(report_path, 'w') as f:
                f.write(report_content)
                
            print(f"\nReport generated successfully at: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise

    def run_pipeline(self):
        """Enhanced training pipeline with model saving"""
        try:
            print("Fetching extended historical data...")
            df = self.fetch_data()
            
            print("Preparing data with enhanced features...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
            
            print("Starting hyperparameter optimization...")
            best_hp = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            
            print("Training final model with best parameters...")
            history = self.train_model(X_train, y_train, X_val, y_val)
            
            print("Evaluating model...")
            metrics, predictions = self.evaluate_model(X_test, y_test)
            
            print("Saving model...")
            model_path = self.save_model()
            
            print("Generating report...")
            self.generate_report(metrics, history, best_hp.values)
            
            print("\nBest Hyperparameters:", best_hp.values)
            print("\nModel Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            return metrics, predictions, history
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise
if __name__ == "__main__":
    try:
        predictor = StockPredictor(
            symbol='AAPL',
            start_date='2020-01-01',
            sequence_length=60
        )
        metrics, predictions, history = predictor.run_pipeline()
    except Exception as e:
        print(f"Program terminated with error: {str(e)}")