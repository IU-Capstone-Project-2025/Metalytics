import pandas as pd
import psycopg2
from psycopg2 import sql
from forecasting_framework import ForecastFramework
import numpy as np
import os
from dotenv import load_dotenv


def update_predicted_prices(data, metal_id, db_params):
    """
    Обновляет прогнозируемые цены для указанного металла:
    1. validation to ensure the metal_id exists:
    2. Удаляет все старые прогнозы для этого металла
    3. Вставляет новые прогнозы

    :param data: pandas Series с временными метками в индексе
    и ценами в значениях
    :param metal_id: ID металла из таблицы metals
    :param db_params: параметры подключения к БД
    """
    conn = None
    cursor = None
    try:
        # Подключаемся к БД
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # 1. Validation to ensure the metal_id exists:
        cursor.execute("SELECT id FROM metals WHERE id = %s", (metal_id,))
        if not cursor.fetchone():
            raise ValueError(f"Invalid metal_id: {metal_id}. \
Valid IDs are 1 (gold), 2 (silver), 3 (platinum)")

        # 2. Удаляем все старые прогнозы для этого металла
        delete_query = sql.SQL("""
            DELETE FROM predicted_prices
            WHERE metal_id = %s
        """)

        cursor.execute(
            delete_query,
            (int(metal_id),)
        )  # Явное преобразование в int
        print(f"Удалены старые прогнозы для metal_id={metal_id}")

        # 3. Вставляем новые прогнозы
        if not data.empty:
            # Преобразуем Series в список кортежей, явно преобразовывая типы
            records = []
            for timestamp, price in data.items():
                # Преобразуем numpy-типы в стандартные Python-типы
                if isinstance(price, (np.floating, np.integer)):
                    price = float(price)
                elif isinstance(price, np.ndarray):
                    price = float(price[0]) if len(price) > 0 else 0.0

                records.append((
                    int(metal_id),  # metal_id как int
                    pd.to_datetime(
                        timestamp
                    ).to_pydatetime(),  # timestamp как datetime
                    float(price)  # price как float
                ))

            # SQL-запрос для вставки данных
            insert_query = sql.SQL("""
                INSERT INTO predicted_prices (metal_id, timestamp, price)
                VALUES (%s, %s, %s)
            """)

            # Выполняем массовую вставку
            cursor.executemany(insert_query, records)
            print(f"Вставлено {len(records)} новых записей \
для metal_id={metal_id}")

        # Фиксируем изменения
        conn.commit()
        print("Обновление завершено успешно")

    except Exception as e:
        print(f"Ошибка при обновлении данных: {e}")
        if conn:
            conn.rollback()
        raise  # Повторно поднимаем исключение для отладки
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_forecast_to_db():
    path: str = "baseline_model"
    fm = ForecastFramework.load_from_file(
        path,
        # dataframe,
        # forecast_model=LSTMCloseFM(),
        # name="lstm_model",
    )

    # Create forecast
    unit = "h"  # units of time
    value = 24 * 10  # value of units

    # Obtain pandas series with forecasted data
    forecast = fm.create_forecast(value=value, unit=unit)

    load_dotenv()
    # Параметры подключения к БД (замените на свои)
    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

    # ID металла (из вашей таблицы metals: 1-Gold, 2-Silver, 3-Platinum)
    metal_id = 1  # Предполагаем, что это прогноз для Gold

    # Вызываем функцию для обновления данных
    update_predicted_prices(forecast, metal_id, db_params)
