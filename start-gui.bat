@echo off
REM Запуск GUI для llama.cpp
echo ========================================
echo    LLaMA.cpp GUI Launcher
echo ========================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден! Установите Python 3.9+
    pause
    exit /b 1
)

echo [OK] Python найден
echo.

REM Проверка зависимостей
echo Проверка зависимостей...
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [INFO] Устанавливаю зависимости...
    pip install -r gui\requirements-gui.txt
    if errorlevel 1 (
        echo [ERROR] Не удалось установить зависимости
        pause
        exit /b 1
    )
)

echo [OK] Зависимости установлены
echo.
echo Запуск GUI...
echo.

REM Запуск GUI
python gui\llama_gui.py

if errorlevel 1 (
    echo.
    echo [ERROR] Ошибка при запуске GUI
    pause
)
