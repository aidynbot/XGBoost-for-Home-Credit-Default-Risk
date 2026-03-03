import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # Если понадобятся стили отдельно
from pydantic import BaseModel

matplotlib.use('Agg')

app = FastAPI(title="CreditAI Scoring System")
templates = Jinja2Templates(directory="templates")

# Пути к файлам
MODEL_PATH = '../models/xgboost_credit.json' # Проверь имя файла! Ты писал xgboost_credit.json в прошлом шаге
# Если ты не сохранял список колонок отдельно, его можно вытащить из модели (иногда),
# но лучше использовать тот pickle, если он есть. Если нет - скажи, напишу как вытащить.
# Предположим, список колонок есть:
# COLUMNS_PATH = '../models/model_columns.pkl'
# model_columns = joblib.load(COLUMNS_PATH)

# ВРЕМЕННОЕ РЕШЕНИЕ: Если нет файла с колонками, модель сама их знает,
# но нам нужно соблюсти порядок при создании DataFrame.
# Давай загрузим модель:
model = xgb.Booster()
model.load_model(MODEL_PATH)
# Получаем имена фичей из самой модели
model_columns = model.feature_names 

class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AGE_YEARS: float
    EMPLOYED_YEARS: float
    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    CODE_GENDER: int = 0
    NAME_EDUCATION_TYPE: int = 4

# --- ИНИЦИАЛИЗАЦИЯ SHAP (Один раз при старте) ---
explainer = shap.TreeExplainer(model)

def prepare_features(data_dict):
    df = pd.DataFrame([data_dict])
    
    # 1. Трансформация времени (годы -> дни)
    df['DAYS_BIRTH'] = (df['AGE_YEARS'] * -365).astype(float)
    df['DAYS_EMPLOYED'] = (df['EMPLOYED_YEARS'] * -365).astype(float)
    
    # 2. Логика Goods Price (предполагаем, что кредит берется на полную стоимость товара)
    # Это важный признак в топе SHAP!
    if 'AMT_GOODS_PRICE' not in df.columns:
         df['AMT_GOODS_PRICE'] = df['AMT_CREDIT']
    
    # 3. Feature Engineering (как при обучении)
    df['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['GOODS_DIFF'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT'] # Будет 0, но это корректно
    
    # Защита от деления на 0 для стажа
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'].abs() / (df['DAYS_BIRTH'].abs() + 1)

    # 4. Умное заполнение пропусков (Smart Imputation)
    # Мы создаем полный DataFrame со всеми колонками модели
    final_df = pd.DataFrame(index=[0], columns=model_columns)
    
    # Заполняем тем, что пришло от пользователя
    for col in df.columns:
        if col in final_df.columns:
            final_df[col] = df[col]
            
    # ЗАПОЛНЕНИЕ ОСТАЛЬНЫХ:
    # Вместо 0 заполняем "нейтральными" значениями (медианами трейна), 
    # чтобы не пугать модель аномалиями.
    defaults = {
        'DAYS_REGISTRATION': -4900,   # Среднее значение (~13 лет)
        'DAYS_ID_PUBLISH': -3000,     # Среднее значение (~8 лет)
        'DAYS_LAST_PHONE_CHANGE': -1000,
        'REGION_POPULATION_RELATIVE': 0.02,
        'FLAG_OWN_CAR': 0,            # Нет машины (безопасный дефолт)
        'FLAG_OWN_REALTY': 1,         # Есть жилье (чаще встречается)
    }
    
    for col in final_df.columns:
        if pd.isna(final_df.iloc[0][col]):
            if col in defaults:
                final_df[col] = defaults[col]
            else:
                final_df[col] = 0 # Остальные (флаги документов и т.д.) можно 0

    return final_df.astype('float32')

def generate_shap_plot(df):
    """Генерирует график SHAP Waterfall и возвращает его как base64 строку"""
    # Вычисляем SHAP значения для одной строки
    shap_values = explainer(df)
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    # waterfall отображает вклад каждого признака
    # shap_values[0] - берем первый (и единственный) элемент, так как у нас одна заявка
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close() # Обязательно закрываем, чтобы не забивать память
    buf.seek(0)
    
    # Кодируем в base64
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: ClientData):
    # model_dump() работает в Pydantic v2. Если v1, используй data.dict()
    try:
        input_data = data.model_dump()
    except AttributeError:
        input_data = data.dict()
    
    final_df = prepare_features(input_data)
    
    # Создаем DMatrix. Важно передать feature_names, чтобы порядок совпал с JSON
    dmatrix = xgb.DMatrix(final_df, feature_names=model_columns)
    
    # Предсказание
    probability = float(model.predict(dmatrix)[0])
    
    shap_image_base64 = generate_shap_plot(final_df)

    # Логика решения (порог можно подкрутить)
    risk_level = "High" if probability > 0.5 else ("Medium" if probability > 0.2 else "Low")
    decision = "Reject" if probability > 0.5 else "Approve"

    return {
        "probability": round(probability, 4),
        "decision": decision,
        "risk_level": risk_level,
        "shap_plot": shap_image_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)