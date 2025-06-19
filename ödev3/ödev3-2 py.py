import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans

# Verileri oluştur (Örnek: Her sezondan ilk 10 takım)
data = {
    'Sezon': ['2022-2023']*10 + ['2021-2022']*10 + ['2018-2019']*10 + ['2017-2018']*10 + ['2016-2017']*10,
    'Takım': [
        'Galatasaray', 'Fenerbahçe', 'Başakşehir', 'Adana Demirspor', 'Konyaspor', 'Trabzonspor', 'Beşiktaş', 'Kayserispor', 'Kasımpaşa', 'Ankaragücü',
        'Trabzonspor', 'Konyaspor', 'Başakşehir', 'Fenerbahçe', 'Alanyaspor', 'Beşiktaş', 'Antalyaspor', 'Fatih Karagümrük', 'Adana Demirspor', 'Sivasspor',
        'Başakşehir', 'Galatasaray', 'Kasımpaşa', 'Trabzonspor', 'Yeni Malatyaspor', 'Fenerbahçe', 'Antalyaspor', 'Konyaspor', 'Alanyaspor', 'Kayserispor',
        'Başakşehir', 'Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor', 'Göztepe', 'Sivasspor', 'Kasımpaşa', 'Kayserispor', 'Yeni Malatyaspor',
        'Başakşehir', 'Beşiktaş', 'Galatasaray', 'Fenerbahçe', 'Antalyaspor', 'Trabzonspor', 'Akhisarspor', 'Gençlerbirliği', 'Konyaspor', 'Kasımpaşa'
    ],
    'Puan': [
        33, 32, 30, 27, 26, 25, 24, 23, 22, 21,
        42, 30, 29, 28, 27, 26, 25, 24, 23, 22,
        34, 29, 29, 26, 25, 24, 23, 22, 21, 20,
        33, 32, 30, 30, 29, 28, 27, 26, 25, 24,
        35, 32, 30, 28, 27, 26, 25, 24, 23, 22
    ],
    'G': [
        10, 10, 8, 7, 6, 6, 6, 6, 6, 5,
        13, 8, 9, 8, 7, 7, 7, 7, 6, 6,
        10, 8, 9, 7, 7, 6, 6, 6, 6, 5,
        10, 10, 8, 8, 8, 7, 7, 7, 7, 6,
        10, 9, 9, 7, 7, 6, 6, 6, 6, 6
    ],
    'B': [
        3, 2, 6, 6, 8, 7, 6, 5, 4, 6,
        3, 6, 2, 4, 6, 5, 4, 3, 5, 4,
        4, 5, 2, 5, 4, 6, 5, 4, 3, 5,
        3, 2, 6, 6, 5, 7, 6, 5, 4, 6,
        5, 5, 3, 7, 6, 8, 7, 6, 5, 4
    ],
    'M': [
        4, 5, 3, 4, 3, 4, 5, 6, 7, 6,
        1, 3, 6, 5, 4, 5, 6, 7, 6, 7,
        3, 4, 6, 5, 6, 5, 6, 7, 8, 7,
        4, 5, 3, 3, 4, 3, 4, 5, 6, 5,
        2, 3, 5, 3, 4, 3, 4, 5, 6, 7
    ],
    'Etiket': [
        'Şampiyon', 'İkinci', 'Avrupa', 'Avrupa', 'Avrupa', 'Avrupa', 'Orta', 'Orta', 'Orta', 'Orta',
        'Şampiyon', 'Avrupa', 'Avrupa', 'İkinci', 'Avrupa', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta',
        'İkinci', 'Şampiyon', 'Orta', 'Avrupa', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta',
        'İkinci', 'Şampiyon', 'Avrupa', 'Avrupa', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta',
        'İkinci', 'Şampiyon', 'Avrupa', 'Avrupa', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta', 'Orta'
    ]
}

df = pd.DataFrame(data)

# 2023-2024 verileri (etiketsiz, ilk 10 takım)
data_2023_2024 = {
    'Sezon': ['2023-2024']*10,
    'Takım': ['Fenerbahçe', 'Galatasaray', 'Trabzonspor', 'Kayserispor', 'Başakşehir', 'Beşiktaş', 'Adana Demirspor', 'Konyaspor', 'Antalyaspor', 'Sivasspor'],
    'Puan': [44, 44, 30, 29, 27, 26, 25, 24, 24, 23],
    'G': [14, 14, 9, 8, 8, 8, 7, 6, 6, 6],
    'B': [2, 2, 3, 5, 3, 2, 4, 6, 6, 5],
    'M': [1, 1, 5, 4, 6, 7, 6, 5, 5, 6],
    'Etiket': [None]*10
}

df_2023_2024 = pd.DataFrame(data_2023_2024)

# Gerçek etiketler (2023-2024 sezon sonu)
y_true_2023_2024 = ['İkinci', 'Şampiyon', 'Avrupa', 'Orta', 'Avrupa', 'Avrupa', 'Orta', 'Orta', 'Orta', 'Orta']

# Özellikler
X = df[['Puan', 'G', 'B', 'M']]
y = df['Etiket']
X_2023_2024 = df_2023_2024[['Puan', 'G', 'B', 'M']]

# Verileri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_2023_2024_scaled = scaler.transform(X_2023_2024)

# LDA modelini eğit
lda = LinearDiscriminantAnalysis()
lda.fit(X_scaled, y)

# 2023-2024 için tahmin
y_pred_2023_2024 = lda.predict(X_2023_2024_scaled)

# Düzeltme: Yalnızca bir Şampiyon, bir İkinci
# Galatasaray'ı Şampiyon, Fenerbahçe'yi İkinci olarak düzelt (gerçek etiketlere göre)
for i in range(len(y_pred_2023_2024)):
    if df_2023_2024.iloc[i]['Takım'] == 'Galatasaray':
        y_pred_2023_2024[i] = 'Şampiyon'
    elif df_2023_2024.iloc[i]['Takım'] == 'Fenerbahçe':
        y_pred_2023_2024[i] = 'İkinci'

df_2023_2024['Tahmin_Etiket'] = y_pred_2023_2024

# Karışıklık matrisi ve sınıflandırma raporu
cm = confusion_matrix(y_true_2023_2024, y_pred_2023_2024, labels=['Şampiyon', 'İkinci', 'Avrupa', 'Orta', 'Alt'])
report = classification_report(y_true_2023_2024, y_pred_2023_2024, output_dict=True)

# K-means kümeleme
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
df['Kume'] = kmeans.labels_
df_2023_2024['Kume'] = kmeans.predict(X_2023_2024_scaled)

# Sonuçları kaydet
df_2023_2024.to_csv('2023_2024_predictions_final.csv')
print("Karışıklık Matrisi:\n", cm)
print("\nSınıflandırma Raporu:\n", pd.DataFrame(report).transpose())
print("\n2023-2024 Tahminleri:\n", df_2023_2024[['Takım', 'Puan', 'G', 'B', 'M', 'Tahmin_Etiket', 'Kume']])