# src/rules_explainer.py

def get_rule_based_explanation(features: dict, probability: float):
    """
    Генерирует детерминированное объяснение решения на основе пороговых значений признаков.
    """
    reasons = []
    
    # 1. Анализ внешних источников (самый важный фактор в модели)
    ext_score = features.get('EXT_SOURCES_MEAN', 0)
    if ext_score < 0.3:
        reasons.append("Критически низкий рейтинг во внешних кредитных бюро.")
    elif ext_score < 0.5:
        reasons.append("Рейтинг во внешних источниках ниже среднего.")

    # 2. Долговая нагрузка (Debt-to-Income)
    annuity_perc = features.get('ANNUITY_INCOME_PERC', 0)
    if annuity_perc > 0.4:
        reasons.append("Слишком высокая доля ежемесячного платежа относительно дохода (более 40%).")
    elif annuity_perc > 0.3:
        reasons.append("Повышенная долговая нагрузка на бюджет.")

    # 3. Трудовой стаж
    employed_perc = features.get('DAYS_EMPLOYED_PERC', 0)
    if employed_perc < 0.05:
        reasons.append("Недостаточный стаж работы относительно возраста заемщика.")

    # 4. Интенсивность платежа (Payment Rate)
    payment_rate = features.get('PAYMENT_RATE', 0)
    if payment_rate > 0.12:
        reasons.append("Слишком короткий срок кредита при запрашиваемой сумме (высокая нагрузка).")

    # Формирование итогового сообщения
    if probability > 0.5:
        header = "Отказ обусловлен следующими факторами: "
        main_text = " ".join(reasons) if reasons else "Совокупность факторов риска превышает допустимый порог."
    elif probability > 0.2:
        header = "Одобрено с ограничениями: "
        main_text = " ".join(reasons) if reasons else "Рекомендуем не увеличивать кредитную нагрузку."
    else:
        header = "Высокая надежность: "
        main_text = "Ваш финансовый профиль полностью соответствует требованиям банка."

    return f"{header}{main_text}"