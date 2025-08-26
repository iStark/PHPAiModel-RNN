# PHPAiModel-RNN

A toy **Recurrent Neural Network (RNN)** implementation written in pure **PHP**.  
The project demonstrates how simple matrix-based RNNs can be used for chat-style text generation on shared hosting without external dependencies.

⚠️ This is a **learning project**, not a production-ready ML framework.

---

## Features
- Pure PHP runtime (no external libraries required).
- RNN architecture with matrix multiplications for hidden state updates.
- Weight initialization and training from custom datasets.
- Support for bilingual (RU/EN) datasets.
- Simple web-based chat interface (`index.php`).
- JSON-based weight files for portability.
- Easy to deploy on shared hosting.

---

## Tools included
- `aicore.php` — Core runtime (matrix math, forward/backward pass, text generation).
- `generator_weights.php` — RNN weight generator, builds initial matrices from datasets.
- `index.php` — Minimal web chat interface.
- `Datasets/` — Datasets where each dialog is marked with:
    - `<Q>` — Question
    - `<A>` — Answer
    - `<NL>` — New line

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/iStark/PHPAiModel-RNN
   cd PHPAiModel-RNN
2. Place it on any PHP-capable server (Apache, Nginx, or shared hosting).
3. Open index.php in the browser to start chatting with the model.

---

## License

MIT License © 2025  
Developed by Artur Strazewicz

Links:
- [GitHub](https://github.com/iStark/PHPAiModel-RNN)
- [X (Twitter)](https://x.com/strazewicz)
- [TruthSocial](https://truthsocial.com/@strazewicz)

---

# PHPAiModel-RNN

Простейшая реализация **Recurrent Neural Network (RNN)** на чистом **PHP**.  
Проект демонстрирует, как матричные RNN могут использоваться для генерации текста в формате чата даже на обычном shared-хостинге без внешних библиотек.

⚠️ Это **учебный проект**, а не готовый ML-фреймворк.

---

## Возможности
- Чистый PHP (без внешних зависимостей).
- Архитектура RNN с матричными операциями для обновления скрытого состояния.
- Инициализация весов и обучение на кастомных датасетах.
- Поддержка русско-английских диалогов.
- Простой веб-интерфейс чата (`index.php`).
- Веса в формате JSON для удобства.
- Лёгкий деплой на shared-хостинге.

---

## Инструменты
- `aicore.php` — ядро (матричные операции, forward/backward pass, генерация текста).
- `generator_weights.php` — генератор весов RNN, создаёт матрицы из датасетов.
- `index.php` — минимальный веб-интерфейс чата.
- `Datasets/` — датасеты, где каждый диалог размечен:
    - `<Q>` — Вопрос
    - `<A>` — Ответ
    - `<NL>` — Перенос строки  

---

## Установка

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/iStark/PHPAiModel-RNN
   cd PHPAiModel-RNN

2. Разместить на любом сервере с поддержкой PHP (Apache, Nginx или shared-хостинг).
3. Открыть `index.php` в браузере, чтобы начать чат с моделью.

---

## Лицензия

MIT License © 2025  
Разработал Artur Strazewicz

Ссылки:
- [GitHub](https://github.com/iStark/PHPAiModel-RNN)
- [X (Twitter)](https://x.com/strazewicz)
- [TruthSocial](https://truthsocial.com/@strazewicz) 