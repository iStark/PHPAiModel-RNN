<?php
declare(strict_types=1);

// Папка с моделями
$MODELS_DIR = __DIR__ . DIRECTORY_SEPARATOR . 'Models';
@mkdir($MODELS_DIR, 0777, true);

// PHP 7.4 safe ends_with
function ends_with(string $haystack, string $needle): bool {
    $len = strlen($needle);
    if ($len === 0) return true;
    return substr($haystack, -$len) === $needle;
}
function list_files(string $dir, string $ext): array {
    if (!is_dir($dir)) return [];
    $files = scandir($dir) ?: [];
    $out = [];
    foreach ($files as $f) {
        if ($f === '.' || $f === '..') continue;
        $lf = strtolower($f);
        if (ends_with($lf, '.' . strtolower($ext))) $out[] = $f;
    }
    sort($out);
    return $out;
}

$models = list_files($MODELS_DIR, 'json');
?>
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PHP RNN Chat — Chat UI</title>
    <style>
        :root{ --bg:#f7f8fb; --card:#ffffff; --text:#0f172a; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }
        *{box-sizing:border-box}
        body{margin:0;background:var(--bg);color:var(--text);font-family:System-ui,-apple-system,Segoe UI,Roboto,Inter,Arial}
        header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center}
        header h1{font-size:16px;margin:0}
        header .sp{flex:1}
        select,button,textarea{border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;background:#fff;color:var(--text)}
        button{background:var(--accent);color:#fff;border-color:transparent;cursor:pointer}
        main{max-width:980px;margin:0 auto;padding:16px;}
        .chatwrap{background:var(--card);border:1px solid var(--line);border-radius:16px;display:flex;flex-direction:column;min-height:70vh}
        .chat{flex:1;overflow:auto;padding:16px}
        .bubble{max-width:80%;padding:10px 12px;margin:8px 0;border-radius:12px;box-shadow:0 1px 0 rgba(0,0,0,.03)}
        .me{align-self:flex-end;background:#eef2ff}
        .bot{align-self:flex-start;background:#f9fafb}
        .composer{display:flex;gap:8px;border-top:1px solid var(--line);padding:12px}
        .composer textarea{flex:1;resize:vertical;min-height:54px}
        .row{display:flex;gap:8px;align-items:center}
        .hint{font-size:12px;color:var(--muted)}
    </style>
</head>
<body>
<header>
    <h1>PHP RNN Chat</h1>
    <div class="sp"></div>
    <label for="model">Модель:</label>
    <select id="model" title="Выберите модель из /Models">
        <option value="">— выберите —</option>
        <?php foreach($models as $m): ?>
            <option value="<?= htmlspecialchars($m, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?>"><?= htmlspecialchars($m, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?></option>
        <?php endforeach; ?>
    </select>
    <button id="clear">Сбросить диалог</button>
</header>
<main>
    <p class="hint">Совет: Сначала обучите модель в <code>generator_weights.php</code>, файл появится в папке <code>/Models</code>, затем выберите её здесь.</p>
    <div class="chatwrap">
        <div id="chat" class="chat"></div>
        <div class="composer">
            <textarea id="prompt" placeholder="Напишите сообщение…"></textarea>
            <button id="send">Отправить</button>
        </div>
    </div>
</main>
<hr style="margin-top:40px; border:0; border-top:1px solid #ccc;">

<footer style="background:#222; color:#eee; text-align:center; padding:20px; font-family:Arial, sans-serif; font-size:14px;">
    <div style="margin-bottom:10px;">
        <strong>PHPAiModel-RNN</strong> © 2025 — MIT License
    </div>
    <div style="margin-bottom:10px;">
        Developed by <a href="https://www.linkedin.com/in/arthur-stark/" style="color:#4ea3ff; text-decoration:none;">Artur Strazewicz</a>
    </div>
    <div>
        <a href="https://github.com/iStark/PHPAiModel-RNN" style="color:#aaa; margin:0 8px; text-decoration:none;">GitHub</a> |
        <a href="https://x.com/strazewicz" style="color:#aaa; margin:0 8px; text-decoration:none;">X (Twitter)</a> |
        <a href="https://truthsocial.com/@strazewicz" style="color:#aaa; margin:0 8px; text-decoration:none;">TruthSocial</a>
    </div>
</footer>

<script>
    const elChat   = document.getElementById('chat');
    const elSend   = document.getElementById('send');
    const elClear  = document.getElementById('clear');
    const elPrompt = document.getElementById('prompt');
    const elModel  = document.getElementById('model');

    function addBubble(text, who){
        const div = document.createElement('div');
        div.className = 'bubble ' + (who==='me' ? 'me' : 'bot');
        div.textContent = text;
        elChat.appendChild(div);
        elChat.scrollTop = elChat.scrollHeight;
    }

    async function sendMessage(){
        const model = (elModel.value || '').trim();
        const prompt = (elPrompt.value || '').trim();
        if (!model){ addBubble('Ошибка: выберите модель.', 'bot'); return; }
        if (!prompt){ return; }

        addBubble(prompt, 'me');
        elPrompt.value = '';
        elSend.disabled = true;

        try{
            const body = {
                model,
                prompt,
                temperature: 0.9,
                top_k: 50,
                max_tokens: 300
            };
            const res = await fetch('aicore.php', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify(body)
            });
            let js;
            try { js = await res.json(); } catch(e){ throw new Error('Некорректный ответ от aicore.php'); }
            if (!js.ok) throw new Error(js.error || 'Ошибка инференса');
            addBubble(js.reply || '(пусто)', 'bot');
        } catch(err){
            addBubble('Ошибка: ' + (err.message || String(err)), 'bot');
        } finally {
            elSend.disabled = false;
            elPrompt.focus();
        }
    }

    elSend.addEventListener('click', sendMessage);
    elPrompt.addEventListener('keydown', (e)=>{
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });

    elClear.addEventListener('click', async ()=>{
        try { await fetch('aicore.php?reset=1'); } catch(_){}
        elChat.innerHTML = '';
        addBubble('Диалог сброшен.', 'bot');
    });
</script>
</body>
</html>
