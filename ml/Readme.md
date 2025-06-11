## Скрипты
- `fetch_gold_ohlcv.py`: Скрипт для сбора данных OHLCV (Open, High, Low, Close, Volume)
- `calculate_indicators.py`: Скрипт для вычисления технических индикаторов (EMA20, RSI14, ATR14, MACD) на основе собранных данных.

## Инструкция по установке модулей
1. Убедитесь, что у вас установлен Python 3.7 или выше.
2. Установите необходимые библиотеки с помощью pip: 
`pip install pandas yfinance ta`
3. Проверьте установку, запустив Python и импортировав библиотеки: 
`import pandas`
`import yfinance`
`import ta`
Если ошибок нет, установка прошла успешно.

## Использование
1. Запустите `fetch_gold_ohlcv.py` для сбора данных в файл `gold_futures_yahoo_1h.csv`.
2. Запустите `calculate_indicators.py` для добавления индикаторов в файл `gold_futures_with_indicators.csv`.
