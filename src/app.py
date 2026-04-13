import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # Если понадобятся стили отдельно
from pydantic import BaseModel
from pricing_engine import engine as pricing_engine
from rule_based import get_rule_based_explanation
# from llm_explainer import get_ai_explanation

load_dotenv()
matplotlib.use('Agg')

app = FastAPI(title="CreditAI Scoring System")
templates = Jinja2Templates(directory="templates")

# Пути к файлам
MODEL_PATH = '../models/xgboost_credit.json' # Проверь имя файла! Ты писал xgboost_credit.json в прошлом шаге

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

def extract_top_shap_features(shap_explanation, top_n=3):
    """
    Извлекает топ-N признаков с самым сильным влиянием.
    Возвращает список строк, понятных для LLM.
    """
    # shap_explanation — это shap_values[0]
    values = shap_explanation.values
    names = shap_explanation.feature_names
    
    # Объединяем названия и значения, сортируем по абсолютному значению (силе влияния)
    feature_impacts = list(zip(names, values))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_features = []
    for name, impact in feature_impacts[:top_n]:
        # Переводим в понятный язык для ИИ
        direction = "повышает риск" if impact > 0 else "снижает риск"
        top_features.append(f"{name} ({direction})")
        
    return top_features


def generate_shap_plot(shap_explanation): # <--- Изменили аргумент
    """Генерирует график SHAP Waterfall и возвращает его как base64 строку"""
    plt.figure(figsize=(10, 6))
    
    # Передаем готовый объект, max_display=10
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close() 
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Не забудь импортировать функцию для LLM, если будешь ее использовать
# from llm_explainer import get_ai_explanation 

@app.post("/predict")
async def predict(data: ClientData):
    try:
        input_data = data.model_dump()
    except AttributeError:
        input_data = data.dict()
    
    final_df = prepare_features(input_data)
    features_dict = final_df.iloc[0].to_dict()

    dmatrix = xgb.DMatrix(final_df, feature_names=model_columns)
    
    # Предсказание
    probability = float(model.predict(dmatrix)[0])

    # Считаем финансовые условия (Risk-Based Pricing)
    pricing = pricing_engine.calculate_rate(probability, features=features_dict)
    shap_values = explainer(final_df)
    single_explanation = shap_values[0]
    # Генерация графика
    shap_image_base64 = generate_shap_plot(single_explanation)
    explanation = get_rule_based_explanation(features_dict, probability)
    
    top_features_list = extract_top_shap_features(single_explanation, top_n=3)

    # Логика уровня риска (важно для UI прогресс-бара)
    # Пороги лучше ставить логичные: < 0.2 (Low), 0.2 - 0.5 (Medium), > 0.5 (High)
    if probability > 0.5:
        risk_level = "High"
    elif probability > 0.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # Базовое решение модели (без учета прайсинга, чисто ML)
    classic_decision = "Reject" if probability > 0.5 else "Approve"

    response = {
        "probability": round(probability, 4),
        "risk_level": risk_level,                 # ИСПРАВЛЕНО: Вернули для фронтенда
        "classic_decision": classic_decision,     # Базовый ML вердикт
        "decision": pricing["decision"],          # Итоговый вердикт от бизнес-логики (Pricing)
        "custom_rate": pricing["rate"],           # Ставка
        "risk_premium": pricing.get("premium", 0),
        "loyalty_discount": pricing.get("loyalty_discount", 0),
        "optimization_discount": pricing.get("optimization_discount", 0),
        "market_diff": pricing.get("market_comparison", 0),
        "shap_plot": shap_image_base64,
        "explanation": explanation,               
    }

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)