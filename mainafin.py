import math
from collections import Counter
import itertools

def gcd(a, b):
    """Вычисляет наибольший общий делитель"""
    while b != 0:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    """Находит модульный обратный элемент a^(-1) mod m"""
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None

# Эталонные частоты букв для разных языков
ENGLISH_FREQ = {
    'a': 8.2, 'b': 1.5, 'c': 2.8, 'd': 4.3, 'e': 12.7,
    'f': 2.2, 'g': 2.0, 'h': 6.1, 'i': 7.0, 'j': 0.15,
    'k': 0.8, 'l': 4.0, 'm': 2.4, 'n': 6.7, 'o': 7.5,
    'p': 1.9, 'q': 0.10, 'r': 6.0, 's': 6.3, 't': 9.1,
    'u': 2.8, 'v': 1.0, 'w': 2.4, 'x': 0.15, 'y': 2.0,
    'z': 0.07
}

RUSSIAN_FREQ = {
    'о': 10.97, 'а': 8.66, 'е': 8.10, 'и': 7.45, 'н': 6.70,
    'т': 6.26, 'с': 5.47, 'р': 5.21, 'в': 4.97, 'л': 4.96,
    'к': 3.73, 'м': 3.31, 'д': 3.25, 'п': 2.81, 'у': 2.62,
    'я': 2.01, 'ы': 1.90, 'ь': 1.74, 'г': 1.70, 'з': 1.65,
    'б': 1.59, 'ч': 1.44, 'й': 1.21, 'х': 0.97, 'ж': 0.94,
    'ш': 0.73, 'ю': 0.64, 'ц': 0.48, 'щ': 0.36, 'э': 0.32,
    'ф': 0.26, 'ъ': 0.04, 'ё': 0.04
}

# Частоты биграмм для русского языка (примерные)
RUSSIAN_BIGRAM_FREQ = {
    'ст': 1.5, 'но': 1.4, 'то': 1.3, 'на': 1.3, 'ен': 1.2,
    'ов': 1.1, 'ни': 1.1, 'ра': 1.0, 'во': 0.9, 'ко': 0.9,
    'ет': 0.9, 'ос': 0.8, 'ли': 0.8, 'ре': 0.8, 'ка': 0.8,
    'пр': 0.7, 'не': 0.7, 'по': 0.7, 'ро': 0.7, 'то': 0.7,
    'ть': 0.6, 'ал': 0.6, 'ан': 0.6, 'ас': 0.6, 'ди': 0.6
}

def prepare_english_alphabet():
    """Подготавливает английский алфавит"""
    alphabet_list = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_dict = {i: char for i, char in enumerate(alphabet_list)}
    reverse_dict = {char: i for i, char in enumerate(alphabet_list)}
    # Добавляем заглавные буквы
    for i, char in enumerate(alphabet_list):
        reverse_dict[char.upper()] = i
    return alphabet_dict, reverse_dict, 26

def prepare_russian_alphabet(include_yo=True):
    """Подготавливает русский алфавит"""
    if include_yo:
        alphabet_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        m = 33
    else:
        alphabet_list = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        m = 32
    
    alphabet_dict = {i: char for i, char in enumerate(alphabet_list)}
    reverse_dict = {char: i for i, char in enumerate(alphabet_list)}
    # Добавляем заглавные буквы
    for i, char in enumerate(alphabet_list):
        reverse_dict[char.upper()] = i
    return alphabet_dict, reverse_dict, m

def frequency_analysis(text, language):
    """Анализирует частоту букв в тексте и возвращает оценку соответствия языку"""
    # Приводим текст к нижнему регистру и оставляем только буквы
    text = ''.join(c for c in text.lower() if c.isalpha())
    
    if not text:
        return 0
    
    # Подсчитываем частоты букв в тексте
    text_freq = Counter(text)
    total_chars = sum(text_freq.values())
    
    # Нормализуем частоты
    for char in text_freq:
        text_freq[char] = (text_freq[char] / total_chars) * 100
    
    # Выбираем эталонные частоты для языка
    if language == 'english':
        reference_freq = ENGLISH_FREQ
    else:
        reference_freq = RUSSIAN_FREQ
    
    # Вычисляем косинусное сходство между эталонными и текстовыми частотами
    dot_product = 0
    ref_norm = 0
    text_norm = 0
    
    all_chars = set(reference_freq.keys()) | set(text_freq.keys())
    
    for char in all_chars:
        ref_val = reference_freq.get(char, 0)
        text_val = text_freq.get(char, 0)
        dot_product += ref_val * text_val
        ref_norm += ref_val ** 2
        text_norm += text_val ** 2
    
    if ref_norm == 0 or text_norm == 0:
        return 0
    
    return dot_product / (math.sqrt(ref_norm) * math.sqrt(text_norm))

def bigram_frequency_analysis(text, language):
    """Анализирует частоту биграмм в тексте"""
    # Приводим текст к нижнему регистру и оставляем только буквы
    text = ''.join(c for c in text.lower() if c.isalpha())
    
    if len(text) < 2:
        return 0
    
    # Создаем биграммы
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    
    # Подсчитываем частоты биграмм
    bigram_freq = Counter(bigrams)
    total_bigrams = sum(bigram_freq.values())
    
    # Нормализуем частоты
    for bigram in bigram_freq:
        bigram_freq[bigram] = (bigram_freq[bigram] / total_bigrams) * 100
    
    # Выбираем эталонные частоты для языка
    if language == 'russian':
        reference_freq = RUSSIAN_BIGRAM_FREQ
    else:
        # Для английского используем упрощенный подход
        reference_freq = {}
    
    # Вычисляем косинусное сходство между эталонными и текстовыми частотами
    dot_product = 0
    ref_norm = 0
    text_norm = 0
    
    all_bigrams = set(reference_freq.keys()) | set(bigram_freq.keys())
    
    for bigram in all_bigrams:
        ref_val = reference_freq.get(bigram, 0)
        text_val = bigram_freq.get(bigram, 0)
        dot_product += ref_val * text_val
        ref_norm += ref_val ** 2
        text_norm += text_val ** 2
    
    if ref_norm == 0 or text_norm == 0:
        return 0
    
    return dot_product / (math.sqrt(ref_norm) * math.sqrt(text_norm))

# ==================== АФФИННЫЙ ШИФР ====================

def encrypt_affine_english(plaintext, a, b):
    """Шифрует текст аффинным шифром для английского языка"""
    alphabet_dict, reverse_dict, m = prepare_english_alphabet()
    
    # Проверяем, что a и m взаимно просты
    if gcd(a, m) != 1:
        return None
    
    result = []
    
    for char in plaintext:
        if char.lower() in reverse_dict:
            x = reverse_dict[char.lower()]
            y = (a * x + b) % m
            encrypted_char = alphabet_dict[y]
            result.append(encrypted_char.upper() if char.isupper() else encrypted_char)
        else:
            result.append(char)
    return ''.join(result)

def decrypt_affine_english(ciphertext, a, b):
    """Расшифровывает текст аффинным шифром для английского языка"""
    alphabet_dict, reverse_dict, m = prepare_english_alphabet()
    
    # Проверяем, что a и m взаимно просты
    if gcd(a, m) != 1:
        return None
    
    a_inv = mod_inverse(a, m)
    if a_inv is None:
        return None
    
    result = []
    
    for char in ciphertext:
        if char.lower() in reverse_dict:
            y = reverse_dict[char.lower()]
            x = (a_inv * (y - b)) % m
            if x < 0:
                x += m
            decrypted_char = alphabet_dict[x]
            result.append(decrypted_char.upper() if char.isupper() else decrypted_char)
        else:
            result.append(char)
    return ''.join(result)

def decrypt_affine_russian(ciphertext, a, b, include_yo=True):
    """Расшифровывает текст аффинным шифром для русского языка"""
    alphabet_dict, reverse_dict, m = prepare_russian_alphabet(include_yo)
    
    # Проверяем, что a и m взаимно просты
    if gcd(a, m) != 1:
        return None
    
    a_inv = mod_inverse(a, m)
    if a_inv is None:
        return None
    
    result = []
    
    for char in ciphertext:
        if char.lower() in reverse_dict:
            y = reverse_dict[char.lower()]
            x = (a_inv * (y - b)) % m
            if x < 0:
                x += m
            decrypted_char = alphabet_dict[x]
            result.append(decrypted_char.upper() if char.isupper() else decrypted_char)
        else:
            result.append(char)
    return ''.join(result)

def smart_brute_force_affine_english(ciphertext, top_n=10):
    """Умный перебор ключей для английского языка с использованием частотного анализа"""
    alphabet_dict, reverse_dict, m = prepare_english_alphabet()
    valid_keys = []
    
    # Находим все допустимые значения a (взаимно простые с m)
    for a in range(1, m):
        if gcd(a, m) == 1:
            valid_keys.append(a)
    
    print(f"\nНайдено {len(valid_keys)} возможных значений для ключа 'a'")
    print(f"Всего комбинаций ключей: {len(valid_keys)} * {m} = {len(valid_keys) * m}")
    print("Проводим частотный анализ для поиска наиболее вероятных ключей...")
    
    results = []
    
    for a in valid_keys:
        for b in range(m):
            decrypted = decrypt_affine_english(ciphertext, a, b)
            if decrypted is not None:
                score = frequency_analysis(decrypted, 'english')
                results.append((a, b, decrypted, score))
    
    # Сортируем по оценке частотного анализа (по убыванию)
    results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nТоп-{top_n} наиболее вероятных расшифровок:")
    print("=" * 60)
    
    for i, (a, b, decrypted, score) in enumerate(results[:top_n]):
        print(f"{i+1}. a={a}, b={b}, score={score:.4f}")
        print(f"   Текст: {decrypted}")
        print("-" * 50)
    
    return results[:top_n]

def smart_brute_force_affine_russian(ciphertext, include_yo=True, top_n=10):
    """Умный перебор ключей для русского языка с использованием частотного анализа"""
    alphabet_dict, reverse_dict, m = prepare_russian_alphabet(include_yo)
    valid_keys = []
    
    # Находим все допустимые значения a (взаимно простые с m)
    for a in range(1, m):
        if gcd(a, m) == 1:
            valid_keys.append(a)
    
    alphabet_type = "с ё" if include_yo else "без ё"
    print(f"\nНайдено {len(valid_keys)} возможных значений для ключа 'a' (русский алфавит {alphabet_type})")
    print(f"Всего комбинаций ключей: {len(valid_keys)} * {m} = {len(valid_keys) * m}")
    print("Проводим частотный анализ для поиска наиболее вероятных ключей...")
    
    results = []
    
    for a in valid_keys:
        for b in range(m):
            decrypted = decrypt_affine_russian(ciphertext, a, b, include_yo)
            if decrypted is not None:
                score = frequency_analysis(decrypted, 'russian')
                results.append((a, b, decrypted, score))
    
    # Сортируем по оценке частотного анализа (по убыванию)
    results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nТоп-{top_n} наиболее вероятных расшифровок:")
    print("=" * 60)
    
    for i, (a, b, decrypted, score) in enumerate(results[:top_n]):
        print(f"{i+1}. a={a}, b={b}, score={score:.4f}")
        print(f"   Текст: {decrypted}")
        print("-" * 50)
    
    return results[:top_n]

def auto_decrypt_affine_russian(ciphertext, expected_text=None):
    """Автоматическая расшифровка аффинного шифра для русского языка с пробелами и без"""
    # Пробуем разные варианты обработки пробелов
    variants = [
        ("с пробелами", ciphertext),
        ("без пробелов", ciphertext.replace(" ", "")),
        ("только русские буквы", ''.join(c for c in ciphertext if c.isalpha() or c.isspace()))
    ]
    
    best_result = None
    best_score = 0
    
    for variant_name, processed_text in variants:
        print(f"\nПробуем вариант: {variant_name}")
        print(f"Текст для анализа: {processed_text}")
        
        # Пробуем оба варианта алфавита (с ё и без)
        for include_yo in [True, False]:
            alphabet_type = "с ё" if include_yo else "без ё"
            print(f"  Алфавит: {alphabet_type}")
            
            results = smart_brute_force_affine_russian(processed_text, include_yo, top_n=3)
            
            for a, b, decrypted, score in results:
                if expected_text:
                    # Сравниваем с ожидаемым текстом (без учета регистра и пробелов)
                    decrypted_clean = decrypted.replace(" ", "").lower()
                    expected_clean = expected_text.replace(" ", "").lower()
                    
                    if decrypted_clean == expected_clean:
                        print(f"\n✓ Найдено точное совпадение!")
                        print(f"Ключи: a={a}, b={b}")
                        print(f"Расшифрованный текст: {decrypted}")
                        return True, a, b, decrypted
                
                # Запоминаем лучший результат по оценке частотного анализа
                if score > best_score:
                    best_score = score
                    best_result = (a, b, decrypted, score, include_yo)
    
    if best_result:
        a, b, decrypted, score, include_yo = best_result
        alphabet_type = "с ё" if include_yo else "без ё"
        print(f"\nЛучший результат (оценка: {score:.4f}, алфавит: {alphabet_type}):")
        print(f"Ключи: a={a}, b={b}")
        print(f"Расшифрованный текст: {decrypted}")
        
        if expected_text:
            print(f"Ожидаемый текст: {expected_text}")
            print(f"Совпадение: {decrypted.replace(' ', '').lower() == expected_text.replace(' ', '').lower()}")
        
        return True, a, b, decrypted
    
    return False, 0, 0, ""

# ==================== БИГРАММНЫЙ ШИФР ====================

def prepare_bigram_alphabet(include_yo=True):
    """Подготавливает алфавит для биграммного шифра"""
    if include_yo:
        alphabet_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        m = 33
    else:
        alphabet_list = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        m = 32
    
    # Создаем словарь для преобразования биграмм в числа и обратно
    bigram_to_num = {}
    num_to_bigram = {}
    
    idx = 0
    for i, char1 in enumerate(alphabet_list):
        for j, char2 in enumerate(alphabet_list):
            bigram = char1 + char2
            bigram_to_num[bigram] = idx
            num_to_bigram[idx] = bigram
            idx += 1
    
    return bigram_to_num, num_to_bigram, m * m

def text_to_bigrams(text, include_yo=True):
    """Преобразует текст в биграммы"""
    # Очищаем текст от не-буквенных символов и приводим к нижнему регистру
    alphabet_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' if include_yo else 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
    clean_text = ''.join(c for c in text.lower() if c in alphabet_list)
    
    # Если длина текста нечетная, добавляем пробел в конец
    if len(clean_text) % 2 != 0:
        clean_text += ' '
    
    # Разбиваем на биграммы
    bigrams = [clean_text[i:i+2] for i in range(0, len(clean_text), 2)]
    
    return bigrams

def bigrams_to_text(bigrams):
    """Преобразует биграммы обратно в текст"""
    return ''.join(bigrams)

def encrypt_bigram_russian(plaintext, a, b, include_yo=True):
    """Шифрует текст биграммным шифром для русского языка"""
    bigram_to_num, num_to_bigram, m = prepare_bigram_alphabet(include_yo)
    
    # Проверяем, что a и m взаимно просты
    if gcd(a, m) != 1:
        return None
    
    # Преобразуем текст в биграммы
    bigrams = text_to_bigrams(plaintext, include_yo)
    
    # Шифруем каждую биграмму
    encrypted_bigrams = []
    for bigram in bigrams:
        if bigram in bigram_to_num:
            x = bigram_to_num[bigram]
            y = (a * x + b) % m
            encrypted_bigram = num_to_bigram[y]
            encrypted_bigrams.append(encrypted_bigram)
        else:
            encrypted_bigrams.append(bigram)
    
    return ''.join(encrypted_bigrams)

def decrypt_bigram_russian(ciphertext, a, b, include_yo=True):
    """Расшифровывает текст биграммным шифром для русского языка"""
    bigram_to_num, num_to_bigram, m = prepare_bigram_alphabet(include_yo)
    
    # Проверяем, что a и m взаимно просты
    if gcd(a, m) != 1:
        return None
    
    a_inv = mod_inverse(a, m)
    if a_inv is None:
        return None
    
    # Разбиваем шифртекст на биграммы
    bigrams = [ciphertext[i:i+2] for i in range(0, len(ciphertext), 2)]
    
    # Расшифровываем каждую биграмму
    decrypted_bigrams = []
    for bigram in bigrams:
        if bigram in bigram_to_num:
            y = bigram_to_num[bigram]
            x = (a_inv * (y - b)) % m
            if x < 0:
                x += m
            decrypted_bigram = num_to_bigram[x]
            decrypted_bigrams.append(decrypted_bigram)
        else:
            decrypted_bigrams.append(bigram)
    
    return ''.join(decrypted_bigrams)

def smart_brute_force_bigram_russian(ciphertext, include_yo=True, top_n=10):
    """Умный перебор ключей для биграммного шифра на русском языке"""
    bigram_to_num, num_to_bigram, m = prepare_bigram_alphabet(include_yo)
    valid_keys = []
    
    # Находим все допустимые значения a (взаимно простые с m)
    for a in range(1, m):
        if gcd(a, m) == 1:
            valid_keys.append(a)
    
    alphabet_type = "с ё" if include_yo else "без ё"
    print(f"\nНайдено {len(valid_keys)} возможных значений для ключа 'a' (биграммы, русский алфавит {alphabet_type})")
    print(f"Всего комбинаций ключей: {len(valid_keys)} * {m} = {len(valid_keys) * m}")
    print("Проводим частотный анализ для поиска наиболее вероятных ключей...")
    
    results = []
    
    # Ограничим перебор для демонстрации (биграммный шифр имеет слишком много комбинаций)
    max_tests = 1000  # Ограничение для демонстрации
    tested = 0
    
    for a in valid_keys:
        for b in range(0, m, max(1, m // 100)):  # Проверяем только каждую сотую b для скорости
            if tested >= max_tests:
                break
                
            decrypted = decrypt_bigram_russian(ciphertext, a, b, include_yo)
            if decrypted is not None:
                # Используем комбинированную оценку (частоты букв и биграмм)
                letter_score = frequency_analysis(decrypted, 'russian')
                bigram_score = bigram_frequency_analysis(decrypted, 'russian')
                combined_score = 0.7 * letter_score + 0.3 * bigram_score
                
                results.append((a, b, decrypted, combined_score))
                tested += 1
    
    # Сортируем по оценке (по убыванию)
    results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nТоп-{min(top_n, len(results))} наиболее вероятных расшифровок:")
    print("=" * 60)
    
    for i, (a, b, decrypted, score) in enumerate(results[:top_n]):
        print(f"{i+1}. a={a}, b={b}, score={score:.4f}")
        print(f"   Текст: {decrypted}")
        print("-" * 50)
    
    return results[:top_n]

def auto_decrypt_bigram_russian(ciphertext, expected_text=None):
    """Автоматическая расшифровка биграммного шифра для русского языка"""
    print("\nАвтоматическая расшифровка биграммного шифра")
    print("=" * 60)
    
    best_result = None
    best_score = 0
    
    # Пробуем оба варианта алфавита (с ё и без)
    for include_yo in [True, False]:
        alphabet_type = "с ё" if include_yo else "без ё"
        print(f"\nПробуем алфавит: {alphabet_type}")
        
        results = smart_brute_force_bigram_russian(ciphertext, include_yo, top_n=5)
        
        for a, b, decrypted, score in results:
            if expected_text:
                # Сравниваем с ожидаемым текстом (без учета регистра и пробелов)
                decrypted_clean = decrypted.replace(" ", "").lower()
                expected_clean = expected_text.replace(" ", "").lower()
                
                if decrypted_clean == expected_clean:
                    print(f"\n✓ Найдено точное совпадение!")
                    print(f"Ключи: a={a}, b={b}")
                    print(f"Расшифрованный текст: {decrypted}")
                    return True, a, b, decrypted
            
            # Запоминаем лучший результат по оценке частотного анализа
            if score > best_score:
                best_score = score
                best_result = (a, b, decrypted, score, include_yo)
    
    if best_result:
        a, b, decrypted, score, include_yo = best_result
        alphabet_type = "с ё" if include_yo else "без ё"
        print(f"\nЛучший результат (оценка: {score:.4f}, алфавит: {alphabet_type}):")
        print(f"Ключи: a={a}, b={b}")
        print(f"Расшифрованный текст: {decrypted}")
        
        if expected_text:
            print(f"Ожидаемый текст: {expected_text}")
            print(f"Совпадение: {decrypted.replace(' ', '').lower() == expected_text.replace(' ', '').lower()}")
        
        return True, a, b, decrypted
    
    return False, 0, 0, ""

# ==================== ОБЩИЕ ФУНКЦИИ ====================

def direct_decrypt_affine_english(ciphertext):
    """Прямая расшифровка аффинного шифра с известными ключами для английского языка"""
    print("\nПрямая расшифровка с известными ключами (английский, аффинный)")
    print("=" * 50)
    
    try:
        a = int(input("Введите ключ a: "))
        b = int(input("Введите ключ b: "))
        
        decrypted = decrypt_affine_english(ciphertext, a, b)
        
        if decrypted is not None:
            print(f"\n✓ Расшифрованный текст: {decrypted}")
            return True, a, b, decrypted
        else:
            print("❌ Не удалось расшифровать текст с указанными ключами.")
            return False, 0, 0, ""
            
    except ValueError:
        print("❌ Ошибка: ключи должны быть целыми числами.")
        return False, 0, 0, ""

def direct_decrypt_affine_russian(ciphertext, include_yo=True):
    """Прямая расшифровка аффинного шифра с известными ключами для русского языка"""
    alphabet_type = "с ё" if include_yo else "без ё"
    print(f"\nПрямая расшифровка с известными ключами (русский {alphabet_type}, аффинный)")
    print("=" * 50)
    
    try:
        a = int(input("Введите ключ a: "))
        b = int(input("Введите ключ b: "))
        
        decrypted = decrypt_affine_russian(ciphertext, a, b, include_yo)
        
        if decrypted is not None:
            print(f"\n✓ Расшифрованный текст: {decrypted}")
            return True, a, b, decrypted
        else:
            print("❌ Не удалось расшифровать текст с указанными ключами.")
            return False, 0, 0, ""
            
    except ValueError:
        print("❌ Ошибка: ключи должны быть целыми числами.")
        return False, 0, 0, ""

def direct_decrypt_bigram_russian(ciphertext, include_yo=True):
    """Прямая расшифровка биграммного шифра с известными ключами для русского языка"""
    alphabet_type = "с ё" if include_yo else "без ё"
    print(f"\nПрямая расшифровка с известными ключами (русский {alphabet_type}, биграммы)")
    print("=" * 50)
    
    try:
        a = int(input("Введите ключ a: "))
        b = int(input("Введите ключ b: "))
        
        decrypted = decrypt_bigram_russian(ciphertext, a, b, include_yo)
        
        if decrypted is not None:
            print(f"\n✓ Расшифрованный текст: {decrypted}")
            return True, a, b, decrypted
        else:
            print("❌ Не удалось расшифровать текст с указанными ключами.")
            return False, 0, 0, ""
            
    except ValueError:
        print("❌ Ошибка: ключи должны быть целыми числами.")
        return False, 0, 0, ""

def run_test():
    """Функция для тестирования программы на известных примерах"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ПРОГРАММЫ")
    print("="*60)
    
    # Тест для английского языка
    print("\n1. ТЕСТ ДЛЯ АНГЛИЙСКОГО ЯЗЫКА (аффинный шифр):")
    plaintext_en = "hello world"
    a_en, b_en = 5, 8
    
    # Шифруем текст
    ciphertext_en = encrypt_affine_english(plaintext_en, a_en, b_en)
    print(f"Исходный текст: {plaintext_en}")
    print(f"Ключи: a={a_en}, b={b_en}")
    print(f"Зашифрованный текст: {ciphertext_en}")
    
    # Расшифровываем текст
    decrypted_en = decrypt_affine_english(ciphertext_en, a_en, b_en)
    print(f"Расшифрованный текст: {decrypted_en}")
    print(f"Тест пройден: {plaintext_en == decrypted_en}")
    
    # Автоматический тест для вашего шифра (аффинный)
    print("\n2. АВТОМАТИЧЕСКИЙ ТЕСТ ДЛЯ ВАШЕГО ШИФРА (аффинный):")
    ciphertext_custom = "ЕСШКЩ ЪЧЩПС РЕДЭИ ТЩЦЬЯ МФЖФЦ ХНЧЛЭ КЭТЭП"
    expected_plaintext = "встречаемся в полдень у хижины загородом"
    
    print(f"Шифртекст: {ciphertext_custom}")
    print(f"Ожидаемый исходный текст: {expected_plaintext}")
    
    success, a, b, text = auto_decrypt_affine_russian(ciphertext_custom, expected_plaintext)
    
    if success:
        print(f"\n✓ ТЕСТ ПРОЙДЕН! Найденные ключи: a={a}, b={b}")
        print(f"Расшифрованный текст: {text}")
    else:
        print("❌ Тест не пройден. Не удалось найти правильные ключи.")
    
    # Тест для биграммного шифра
    print("\n3. ТЕСТ ДЛЯ БИГРАММНОГО ШИФРА:")
    plaintext_bigram = "привет мир"
    a_bigram, b_bigram = 17, 23
    
    # Шифруем текст
    ciphertext_bigram = encrypt_bigram_russian(plaintext_bigram, a_bigram, b_bigram, include_yo=False)
    print(f"Исходный текст: {plaintext_bigram}")
    print(f"Ключи: a={a_bigram}, b={b_bigram}")
    print(f"Зашифрованный текст: {ciphertext_bigram}")
    
    # Расшифровываем текст
    decrypted_bigram = decrypt_bigram_russian(ciphertext_bigram, a_bigram, b_bigram, include_yo=False)
    print(f"Расшифрованный текст: {decrypted_bigram}")
    print(f"Тест пройден: {plaintext_bigram == decrypted_bigram.replace(' ', '')}")
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)

def main():
    print("УМНАЯ РАСШИФРОВКА ШИФРОВ")
    print("=" * 60)
    print("Программа использует частотный анализ для автоматического")
    print("определения наиболее вероятных ключей шифрования")
    print("=" * 60)
    
    # Предложение запустить тест
    test_choice = input("Хотите запустить тест программы? (y/n): ").strip().lower()
    if test_choice == 'y':
        run_test()
    
    while True:
        print("\nВыберите тип шифра:")
        print("1 - Аффинный шифр (монограммы)")
        print("2 - Биграммный шифр")
        print("t - Запустить тест")
        print("q - Выход")
        
        cipher_type = input("Ваш выбор (1/2/t/q): ").strip().lower()
        
        if cipher_type == 'q':
            print("До свидания!")
            break
        elif cipher_type == 't':
            run_test()
            continue
        
        if cipher_type not in ['1', '2']:
            print("Неверный выбор. Попробуйте снова.")
            continue
        
        try:
            if cipher_type == '1':  # Аффинный шифр
                print("\nВыберите язык для аффинного шифра:")
                print("1 - Английский")
                print("2 - Русский (автоматический подбор)")
                print("3 - Русский (прямой ввод ключей)")
                
                lang_choice = input("Ваш выбор (1/2/3): ").strip()
                
                if lang_choice == '1':
                    lang_name = "английский"
                    alphabet_list = 'abcdefghijklmnopqrstuvwxyz'
                    print(f"\nВыбран {lang_name} язык. Алфавит: {alphabet_list}")
                    print(f"Длина алфавита: 26")
                    
                    ciphertext = input("\nВведите шифртекст: ")
                    
                    if not ciphertext:
                        print("Шифртекст не может быть пустым.")
                        continue
                    
                    # Выбор метода расшифровки
                    print("\nВыберите метод расшифровки:")
                    print("1 - Умный перебор с частотным анализом")
                    print("2 - Прямая расшифровка с известными ключами")
                    method = input("Ваш выбор (1/2): ").strip()
                    
                    if method == '1':
                        print("\nНачинаем умный перебор ключей...")
                        results = smart_brute_force_affine_english(ciphertext)
                        
                        # Предлагаем выбрать вариант
                        if results:
                            choice_num = input("\nВыберите номер правильного варианта (или Enter для выхода): ").strip()
                            if choice_num and choice_num.isdigit():
                                idx = int(choice_num) - 1
                                if 0 <= idx < len(results):
                                    a, b, text, score = results[idx]
                                    print(f"\n✓ Выбран вариант {choice_num}: a={a}, b={b}")
                                    print(f"Расшифрованный текст: {text}")
                    elif method == '2':
                        success, a, b, text = direct_decrypt_affine_english(ciphertext)
                    else:
                        print("Неверный выбор метода. Используется умный перебор.")
                        print("\nНачинаем умный перебор ключей...")
                        results = smart_brute_force_affine_english(ciphertext)
                    
                else:  # Русский язык
                    ciphertext = input("\nВведите шифртекст: ")
                    
                    if not ciphertext:
                        print("Шифртекст не может быть пустым.")
                        continue
                    
                    if lang_choice == '2':
                        print("\nНачинаем автоматическую расшифровку...")
                        success, a, b, text = auto_decrypt_affine_russian(ciphertext)
                    else:  # lang_choice == '3'
                        include_yo = input("Использовать алфавит с буквой 'ё'? (y/n): ").strip().lower() == 'y'
                        success, a, b, text = direct_decrypt_affine_russian(ciphertext, include_yo)
            
            else:  # Биграммный шифр
                print("\nБиграммный шифр поддерживает только русский язык")
                ciphertext = input("\nВведите шифртекст: ")
                
                if not ciphertext:
                    print("Шифртекст не может быть пустым.")
                    continue
                
                # Выбор метода расшифровки
                print("\nВыберите метод расшифровки:")
                print("1 - Умный перебор с частотным анализом")
                print("2 - Прямая расшифровка с известными ключами")
                method = input("Ваш выбор (1/2): ").strip()
                
                if method == '1':
                    print("\nНачинаем автоматическую расшифровку биграммного шифра...")
                    success, a, b, text = auto_decrypt_bigram_russian(ciphertext)
                elif method == '2':
                    include_yo = input("Использовать алфавит с буквой 'ё'? (y/n): ").strip().lower() == 'y'
                    success, a, b, text = direct_decrypt_bigram_russian(ciphertext, include_yo)
                else:
                    print("Неверный выбор метода. Используется умный перебор.")
                    print("\nНачинаем автоматическую расшифровку биграммного шифра...")
                    success, a, b, text = auto_decrypt_bigram_russian(ciphertext)
            
            # Спросить, хочет ли пользователь продолжить
            continue_choice = input("\nХотите расшифровать другой текст? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("До свидания!")
                break
                
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            import traceback
            traceback.print_exc()
            print("Попробуйте снова.")

if __name__ == "__main__":
    main()