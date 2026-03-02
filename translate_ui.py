#!/usr/bin/env python3
"""
Script to translate Russian UI strings to English in llama_gui.py
"""

translations = {
    # UI Elements
    "Режим работы сервера": "Server Mode",
    "Веб-интерфейс (для чата в браузере)": "Web Interface (for browser chat)",
    "API режим (для VSCode/агентов)": "API Mode (for VSCode/agents)",
    "В веб-режиме откроется браузер для чата.": "In web mode, browser will open for chat.",
    "В API режиме сервер будет доступен для интеграции с VSCode и другими инструментами.": "In API mode, server will be available for integration with VSCode and other tools.",
    "Параметры сервера": "Server Parameters",
    "Порт:": "Port:",
    "Размер контекста:": "Context Size:",
    "Потоков:": "Threads:",
    "Размер батча:": "Batch Size:",
    "Использовать GPU": "Use GPU",
    "GPU слоёв:": "GPU Layers:",
    "Включить CORS (для веб-доступа)": "Enable CORS (for web access)",
    "API ключ (опционально):": "API Key (optional):",
    "Оставьте пустым для открытого доступа": "Leave empty for open access",
    "Запустить сервер": "Start Server",
    "Остановить сервер": "Stop Server",
    "Открыть веб-интерфейс": "Open Web Interface",
    "Очистить лог": "Clear Log",
    "Сервер остановлен": "Server Stopped",
    "Лог сервера:": "Server Log:",
    
    # Inference tab
    "Параметры генерации": "Generation Parameters",
    "Промпт:": "Prompt:",
    "Введите ваш промпт здесь...": "Enter your prompt here...",
    "Токенов:": "Tokens:",
    "Дополнительные опции": "Additional Options",
    "Запустить": "Run",
    "Остановить": "Stop",
    "Очистить вывод": "Clear Output",
    "Вывод модели:": "Model Output:",
    
    # Download tab
    "Поиск и загрузка моделей из HuggingFace": "Search and Download Models from HuggingFace",
    "Поиск моделей на HuggingFace": "Search Models on HuggingFace",
    "Введите запрос (например: 'llama', 'mistral', 'codellama')...": "Enter query (e.g.: 'llama', 'mistral', 'codellama')...",
    "Поиск": "Search",
    "Популярные": "Popular",
    "Сортировка:": "Sort:",
    "По загрузкам": "By Downloads",
    "По лайкам": "By Likes",
    "По дате обновления": "By Date Updated",
    "По названию": "By Name",
    "Результаты поиска": "Search Results",
    "Название модели": "Model Name",
    "Автор": "Author",
    "Загрузки": "Downloads",
    "Лайки": "Likes",
    "Обновлено": "Updated",
    "Выбранная модель": "Selected Model",
    "Не выбрано": "Not Selected",
    "Файл:": "File:",
    "Прогресс загрузки": "Download Progress",
    "Готов к загрузке": "Ready to Download",
    "Скачать модель": "Download Model",
    "Отменить": "Cancel",
    
    # Build tab
    "Сборка llama.cpp с поддержкой выбранного железа": "Build llama.cpp with Selected Hardware Support",
    "Выбор backend": "Backend Selection",
    "CPU (только процессор)": "CPU (only processor)",
    "CUDA (NVIDIA GPU)": "CUDA (NVIDIA GPU)",
    "Metal (macOS GPU)": "Metal (macOS GPU)",
    "Vulkan (AMD/Intel/NVIDIA)": "Vulkan (AMD/Intel/NVIDIA)",
    "SYCL (Intel GPU)": "SYCL (Intel GPU)",
    "ROCm (AMD GPU)": "ROCm (AMD GPU)",
    "Дополнительные опции": "Additional Options",
    "Собрать llama-server": "Build llama-server",
    "Собрать тесты": "Build tests",
    "Использовать ccache (ускорение сборки)": "Use ccache (faster build)",
    "Конфигурировать": "Configure",
    "Собрать": "Build",
    "Установить зависимости": "Install Dependencies",
    "Лог сборки:": "Build Log:",
    
    # Hardware tab
    "Информация о вашей системе": "Your System Information",
    "Обновить информацию": "Refresh Information",
    
    # Messages
    "Ошибка": "Error",
    "Пожалуйста, выберите существующий файл модели": "Please select an existing model file",
    "Пожалуйста, введите промпт": "Please enter a prompt",
    "Не найден llama-cli.exe. Пожалуйста, сначала соберите проект.": "llama-cli.exe not found. Please build the project first.",
    "Ищу в:": "Looking in:",
    "Выполняется инференс...": "Running inference...",
    "Остановлено пользователем": "Stopped by user",
    "Инференс завершён": "Inference completed",
    "Завершено": "Completed",
    "Ошибка при выполнении": "Error during execution",
    "Введите URL или HuggingFace ID": "Enter URL or HuggingFace ID",
    "В разработке": "Under Development",
    "Функция загрузки моделей будет реализована в следующей версии": "Model download feature will be implemented in next version",
    "Подтверждение": "Confirmation",
    "Вы уверены, что хотите выйти?": "Are you sure you want to exit?",
    
    # Server messages
    "Не найден llama-server.exe. Пожалуйста, сначала соберите проект.": "llama-server.exe not found. Please build the project first.",
    "Запуск сервера:": "Starting server:",
    "Запуск сервера...": "Starting server...",
    "Сервер запускается...": "Server starting...",
    "Сервер запущен:": "Server running:",
    "Сервер запущен на": "Server running on",
    "Сервер остановлен": "Server stopped",
    "Ошибка сервера": "Server error",
    "Ошибка при запуске сервера": "Error starting server",
    "Открыт веб-интерфейс:": "Opened web interface:",
    "Сервер ещё работает": "Server is still running",
    "Инференс ещё выполняется": "Inference is still running",
    
    # Search and download
    "Поиск моделей:": "Searching models:",
    "Найдено": "Found",
    "моделей": "models",
    "Ошибка поиска": "Search error",
    "Не удалось выполнить поиск:": "Failed to search:",
    "Загрузка популярных моделей...": "Loading popular models...",
    "Загружено": "Loaded",
    "популярных моделей": "popular models",
    "Не удалось загрузить популярные модели:": "Failed to load popular models:",
    "Ошибка загрузки": "Download error",
    "Сначала выберите модель из списка": "First select a model from the list",
    "Выберите файл для загрузки": "Select file to download",
    "Файл существует": "File exists",
    "уже существует. Скачать заново?": "already exists. Download again?",
    "Загрузка": "Downloading",
    "Загрузка завершена": "Download completed",
    "Загружено:": "Downloaded:",
    "Успех": "Success",
    "Модель успешно загружена:": "Model downloaded successfully:",
    "Не удалось загрузить модель:": "Failed to download model:",
    "Загрузка списка файлов для": "Loading file list for",
    "найдено": "found",
    "файлов": "files",
    "В репозитории": "In repository",
    "не найдено .gguf файлов": "no .gguf files found",
    "Файлы не найдены": "Files not found",
    "Не удалось загрузить список файлов:": "Failed to load file list:",
    "Ошибка загрузки файлов": "File load error",
    
    # Build messages
    "Конфигурирование для backend:": "Configuring for backend:",
    "Начинается конфигурирование...": "Starting configuration...",
    "Начинается сборка...": "Starting build...",
    "Проверка и установка зависимостей...": "Checking and installing dependencies...",
}

import re

def translate_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for russian, english in translations.items():
        # Escape special regex characters
        pattern = re.escape(russian)
        content = re.sub(pattern, english, content)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Translated {filepath}")
        return True
    else:
        print(f"- No changes needed for {filepath}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "gui/llama_gui.py"
    
    translate_file(filepath)
