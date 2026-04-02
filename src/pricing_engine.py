import math

class PricingEngine:
    def __init__(self):
        # Исторические константы Чехии (2018 г.)
        self.CNB_KEY_RATE = 2.0      # Ключевая ставка Чешского нацбанка в 2018
        self.HC_MIN_RATE = 8.9       # Минимальная ставка Home Credit (рекламная)
        self.HC_MAX_RATE = 24.9      # Потолок для беззалоговых потребительских кредитов
        self.MARKET_AVG_BANK = 6.5   # Средняя ставка классических банков (CS, CSOB)
        
    def calculate_rate(self, probability):
        """
        Расчет индивидуальной ставки на основе вероятности дефолта.
        Используется логика: Базовая ставка + Премия за риск + Операционная маржа.
        """
        # 1. Если риск слишком высок (например, выше 50%), кредит не выдается
        if probability > 0.5:
            return {
                "decision": "Reject",
                "rate": None,
                "message": "Risk level exceeds bank tolerance."
            }

        # 2. Расчет риск-премии (Risk Premium)
        # Используем нелинейную зависимость: при росте вероятности ставка растет быстрее.
        # Коэффициент 35 подобран так, чтобы при prob=0.5 ставка достигала максимума.
        risk_premium = math.pow(probability, 1.2) * 38 
        
        # 3. Итоговая ставка
        # Добавляем небольшую маржу за операционные расходы (Admin cost)
        admin_margin = 1.5
        calculated_rate = self.HC_MIN_RATE + risk_premium + admin_margin
        
        # Ограничиваем историческим максимумом
        final_rate = min(calculated_rate, self.HC_MAX_RATE)
        
        # 4. Расчет выгоды по сравнению с рынком (для маркетинга в UI)
        # Home Credit дороже банков, но доступнее микрозаймов
        is_competitive = final_rate < 15.0 

        return {
            "decision": "Approve",
            "rate": round(final_rate, 2),
            "premium": round(risk_premium, 2),
            "key_rate_ref": self.CNB_KEY_RATE,
            "is_competitive": is_competitive,
            "market_comparison": round(final_rate - self.MARKET_AVG_BANK, 2)
        }

# Инстанс для импорта
engine = PricingEngine()