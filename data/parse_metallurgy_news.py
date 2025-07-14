import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import os


def parse_metalinfo_news(page_url, metal_list):
    try:
        news_list = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' +
            ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.447' +
            '2.124 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        news_blocks = soup.find_all('div', 'news-block clearfix')
        for keyword in metal_list:
            for block in news_blocks:
                if keyword.lower() in block.get_text().lower():
                    news_item = {
                        'title': block.find(
                            'h2',
                            class_='news-title'
                        ).get_text(strip=True) if block.find(
                            'h2',
                            class_='news-title'
                        ) else None,
                        'url': "https://www.metalinfo.ru/" + block.find(
                            'a',
                            href=True
                        )['href'] if block.find(
                            'a',
                            href=True
                        )['href'] else None,
                        'date': block.find(
                            'small',
                            class_='news-date'
                        ).get_text(strip=True) if block.find(
                            'small',
                            class_='news-date'
                        ) else None,
                        'preview': block.find(
                            'div',
                            class_='news-annotation clearfix'
                        ).get_text(strip=True) if block.find(
                            'div',
                            class_='news-annotation clearfix'
                        ) else None,
                        'keyword': keyword,
                        'source': page_url,
                        'scraped_at': datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }
                    news_list.append(news_item)

        return news_list

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return []


def save_to_json(data, filename='metalinfo_news.json'):
    try:
        directory = "data"
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Данные успешно сохранены в файл {filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении в JSON: {e}")


# URL для парсинга
url = "https://www.metalinfo.ru/en/news/list.html?pn="

to_json = []

metal_list = ['gold', 'silver', 'zinc']

for page_number in range(1, 13):
    page_url = url + str(page_number)
    print(f"page_number:{page_number}")

    time.sleep(5)
    gold_news = parse_metalinfo_news(page_url, metal_list)
    to_json += gold_news

# Выводим результаты в консоль
print(f"Найдено {len(to_json)} новостей о золоте:")

# Сохраняем данные в JSON-файл
save_to_json(to_json)
