# 🔧 Исправление: Правильное определение MSVC

## Проблема
Программа писала "MSVC Build Tools не установлены" хотя они были установлены в системе:
```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64
```

## Причина
Функция `_check_msvc()` проверяла только наличие `cl.exe` в PATH, но после установки MSVC часто нужна инициализация переменных окружения.

## Решение

### Улучшена функция проверки MSVC

Теперь `_check_msvc()` использует 3 метода проверки:

1. **Проверка PATH** - ищет `cl.exe` в PATH
   ```cmd
   where cl.exe
   ```

2. **Прямой поиск по файловой системе** - ищет папки MSVC
   ```
   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\
   C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\
   ```
   
3. **Проверка в реестре Windows** - ищет ключи реестра Visual Studio
   ```
   HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\SxS\VS7
   ```

### Добавлена инициализация MSVC

Новый метод `initialize_msvc_env()`:
- Находит установленный MSVC
- Добавляет папку `bin` в PATH
- Добавляет Windows SDK в PATH
- Работает с VS 2019 и 2022

### Автоматическая инициализация при запуске

При запуске приложения на Windows:
```python
if self.os_type == "Windows":
    self.dependency_manager.initialize_msvc_env()
```

## Что изменилось

### `gui/dependency_installer.py`

**Метод `_check_msvc()` теперь проверяет:**
✅ Наличие `cl.exe` в PATH
✅ Папки MSVC 2022/2019 на диске
✅ Реестр Windows Visual Studio
✅ Поддерживает BuildTools, Community, Professional, Enterprise версии

**Новый метод `initialize_msvc_env()`:**
✅ Находит MSVC автоматически
✅ Инициализирует PATH для `cl.exe`
✅ Добавляет Windows SDK
✅ Работает при каждом запуске

### `gui/llama_gui.py`

**При инициализации:**
```python
self.dependency_manager = DependencyManager()
self.os_type = platform.system()

# Initialize MSVC environment if on Windows
if self.os_type == "Windows":
    self.dependency_manager.initialize_msvc_env()
```

## Как работает теперь

1. **При запуске приложения:**
   - Проверяется OS (Windows, Linux, macOS)
   - На Windows вызывается `initialize_msvc_env()`
   - MSVC автоматически добавляется в PATH если найден

2. **При проверке зависимостей:**
   - Функция `_check_msvc()` проверяет все 3 метода
   - Если MSVC найден хотя бы одним способом → "установлен"
   - Если не найден → "требуется установка"

3. **При запуске CMake/Build:**
   - CMake может найти `cl.exe` в PATH
   - Нет ошибок "cl.exe not found"
   - Сборка проходит успешно

## Поддерживаемые версии

✅ Visual Studio 2022 BuildTools
✅ Visual Studio 2022 Community
✅ Visual Studio 2022 Professional
✅ Visual Studio 2022 Enterprise
✅ Visual Studio 2019 BuildTools
✅ Visual Studio 2019 Community
✅ Visual Studio 2019 Professional
✅ Visual Studio 2019 Enterprise

## Ручная проверка

Если хотите проверить вручную:

```python
from gui.dependency_installer import DependencyManager

manager = DependencyManager()

# Проверить наличие MSVC
is_installed = manager._check_msvc()
print(f"MSVC installed: {is_installed}")

# Инициализировать PATH
manager.initialize_msvc_env()
print("MSVC PATH initialized")

# Проверить снова
is_installed = manager._check_msvc()
print(f"After init: {is_installed}")
```

## Результат

Теперь программа должна:
✅ Правильно определять MSVC если он установлен
✅ Автоматически инициализировать переменные окружения
✅ Позволять CMake найти компилятор
✅ Успешно собирать проект

## Если все еще не работает

Если программа все еще говорит что MSVC не установлен:

1. **Убедитесь что MSVC установлен:**
   ```cmd
   dir "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
   ```

2. **Убедитесь что папка VC\Tools\MSVC существует:**
   ```cmd
   dir "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
   ```

3. **Проверьте реестр:**
   ```cmd
   reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\SxS\VS7"
   ```

4. **Перезагрузите приложение** - нужно заново запустить чтобы инициализировалась переменная MSVC_INSTALL_PATH

5. **Если ничего не помогает**, напишите в консоль:
   ```cmd
   pip list | findstr pyqt
   python -c "from gui.dependency_installer import DependencyManager; m=DependencyManager(); print(f'MSVC check: {m._check_msvc()}')"
   ```

## Файлы которые изменились

- ✅ `gui/dependency_installer.py` - улучшена проверка MSVC
- ✅ `gui/llama_gui.py` - добавлена инициализация при запуске

Оба файла протестированы и скомпилированы успешно! ✅
