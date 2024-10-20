import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# 1. Загрузка данных для инференса (например, тестовых данных)
test_data = pd.read_csv('data/test.csv')

# 2. Предобработка данных для инференса (те же шаги, что и в обучении)
cat_list = ['sex', 'fasting_blood_sugar', 'resting_electrocardiographic_results', 'exercise_induced_angina', 'slope', 'number_of_major_vessels', 'thal']

# Преобразование категориальных переменных в дамми-переменные
for elem in cat_list:
    test_data[elem] = test_data[elem].astype(object)

# Преобразование категориальных переменных в дамми-переменные
df_test_mod = pd.get_dummies(test_data, columns=cat_list, dtype='int', drop_first=True)

# 3. Масштабирование данных
scaler = StandardScaler()
X_test = scaler.fit_transform(df_test_mod)

# 4. Загрузка лучшей нейронной сети
model = load_model('best_neural_network_model.h5')
print("Глубокая нейронная сеть загружена.")

# 5. Инференс модели
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)  # Бинаризация предсказаний

# 6. Результаты инференса
print("Предсказанные классы:\n", y_pred.flatten())
print("Предсказанные вероятности:\n", y_pred_proba.flatten())
