#!/bin/bash

# Путь к файлу
FILE="frontend/js/script.js"

# Проверяем, существует ли файл
if [ ! -f "$FILE" ]; then
    echo "Ошибка: файл $FILE не найден!"
    exit 1
fi

# Заменяем localhost на 89.223.121.67
sed -i 's/localhost/89.223.121.67/g' "$FILE"

echo "Замена выполнена успешно!"