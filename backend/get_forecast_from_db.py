import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv


def get_prices_from_db(
        metal_id,
        db_params,
        table_name="predicted_prices",
        limit=10):
    """
    Извлекает данные о ценах из базы данных и возвращает в формате JSON

    :param metal_id: ID металла (1-Gold, 2-Silver, 3-Platinum)
    :param db_params: параметры подключения к БД
    :param table_name: имя таблицы (по умолчанию 'predicted_prices')
    :param limit: ограничение количества записей (None - без ограничений)
    :return: JSON-строка с данными в указанном формате
    """
    conn = None
    cursor = None
    try:
        # Подключаемся к БД
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Формируем SQL-запрос
        query = sql.SQL(
            """
            SELECT timestamp, price
            FROM {}
            WHERE metal_id = %s
            ORDER BY timestamp
            {}
        """
        ).format(
            sql.Identifier(table_name),
            sql.SQL("LIMIT %s" if limit else ""),
        )

        # Параметры запроса
        params = (metal_id,)
        if limit:
            params = (metal_id, limit)

        # Выполняем запрос
        cursor.execute(query, params)

        # Получаем данные и преобразуем в нужный формат
        result = []
        for timestamp, price in cursor.fetchall():
            # Форматируем timestamp в ISO-формат с 'Z' на конце
            iso_timestamp = timestamp.isoformat()
            if not iso_timestamp.endswith("Z"):
                iso_timestamp = iso_timestamp.replace("+00:00", "Z")

            result.append({"timestamp": iso_timestamp, "price": float(price)})

        return result

    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# Пример использования
if __name__ == "__main__":
    load_dotenv()
    # Параметры подключения к БД (замените на свои)
    db_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }

    # Получаем данные для Gold (metal_id=1)
    json_data = get_prices_from_db(metal_id=1, db_params=db_params)
    print(json_data)
