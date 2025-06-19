import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Veri Toplama: 2022-2025 TFF Harcama Limitleri (TL)
data = {
    'Season': ['2022-2023', '2022-2023', '2022-2023', '2022-2023',
               '2023-2024', '2023-2024', '2023-2024', '2023-2024',
               '2024-2025', '2024-2025', '2024-2025', '2024-2025'],
    'Club': ['Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor'] * 3,
    'Spending_Limit': [1545820482, 1378878458, 960805023, 879158943,  # 2022-2023
                       2000000000, 1800000000, 1200000000, 1000000000,  # 2023-2024
                       3437333278, 3285079836, 1869822515, 1899243096], # 2024-2025
    'Revenue': [1200000000, 1100000000, 800000000, 700000000,
                1500000000, 1400000000, 1000000000, 900000000,
                1800000000, 1700000000, 1200000000, 1100000000],
    'Debt': [2000000000, 1800000000, 1500000000, 1300000000,
             2200000000, 2000000000, 1700000000, 1500000000,
             2500000000, 2300000000, 1900000000, 1700000000]
}
df = pd.DataFrame(data)

# Veri Temizleme
df['Spending_Limit'] = df['Spending_Limit'].astype(float)
df['Revenue'] = df['Revenue'].astype(float)
df['Debt'] = df['Debt'].astype(float)
df.dropna(inplace=True)  # Eksik veriler
df = df[df['Spending_Limit'] > 0]  # Negatif veya sıfır limitler

# Basit Regresyon Analizi
X = df[['Revenue', 'Debt']]
y = df['Spending_Limit']
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'Regresyon RMSE: {rmse:.2f}')

# Zaman Serisi Analizi (ARIMA)
def arima_forecast(club_data):
    if len(club_data) < 3:  # Yeterli veri kontrolü
        return np.nan
    try:
        model = ARIMA(club_data['Spending_Limit'].values, order=(1,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
    except:
        return np.nan

forecasts = {}
for club in df['Club'].unique():
    club_data = df[df['Club'] == club].sort_values(by='Season')
    forecasts[club] = arima_forecast(club_data)
print('2025-2026 ARIMA Tahminleri (TL):')
for club, value in forecasts.items():
    print(f'{club}: {value:.2f}' if not np.isnan(value) else f'{club}: Tahmin yapılamadı')

# Senaryo Analizleri
scenarios = {
    'Optimistic': {'Revenue_Multiplier': 1.15, 'Debt_Multiplier': 0.85},
    'Pessimistic': {'Revenue_Multiplier': 0.85, 'Debt_Multiplier': 1.15}
}

scenario_results = {}
for scenario, params in scenarios.items():
    temp_df = df[df['Season'] == '2024-2025'].copy()
    temp_df['Revenue'] *= params['Revenue_Multiplier']
    temp_df['Debt'] *= params['Debt_Multiplier']
    X_scenario = temp_df[['Revenue', 'Debt']]
    y_pred_scenario = reg.predict(X_scenario)
    scenario_results[scenario] = dict(zip(temp_df['Club'], y_pred_scenario))

print('\nSenaryo Tahminleri (2025-2026, TL):')
for scenario, results in scenario_results.items():
    print(f'\n{scenario}:')
    for club, value in results.items():
        print(f'{club}: {value:.2f}')

# Geriye Dönük Test (2024-2025)
train_df = df[df['Season'] != '2024-2025']
test_df = df[df['Season'] == '2024-2025']
X_train = train_df[['Revenue', 'Debt']]
y_train = train_df['Spending_Limit']
X_test = test_df[['Revenue', 'Debt']]
y_test = test_df['Spending_Limit']
reg_test = LinearRegression()
reg_test.fit(X_train, y_train)
y_pred_test = reg_test.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'\n2024-2025 Test RMSE: {test_rmse:.2f}')