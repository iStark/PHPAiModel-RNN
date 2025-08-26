<?php
/*
 * PHPAiModel-RNN — index.php
 * Simple web UI for chat with the PHP RNN model: message form, AJAX, and response rendering.
 *
 * Developed by: Artur Strazewicz — concept, architecture, PHP RNN runtime, UI.
 * Year: 2025. License: MIT.
 *
 * Links:
 *   GitHub:      https://github.com/iStark/PHPAiModel-RNN
 *   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
 *   TruthSocial: https://truthsocial.com/@strazewicz
 *   X (Twitter): https://x.com/strazewicz
 */
$modelsDir = __DIR__ . '/Models';
if (!is_dir($modelsDir)) { @mkdir($modelsDir, 0777, true); }
$models = array_values(array_filter(is_dir($modelsDir) ? scandir($modelsDir) : [], fn($f)=>preg_match('/\.json$/i',$f)));
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
            <option><?= htmlspecialchars($m, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?></option>
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
    function addMsg(text, who){
        const el = document.createElement('div');
        el.className = 'bubble ' + (who==='user'?'me':'bot');
        el.textContent = text;
        const chat = document.getElementById('chat');
        chat.appendChild(el);
        chat.scrollTop = chat.scrollHeight;
    }
    async function send(){
        const select = document.getElementById('model');
        const model = select.value;
        if(!model){ alert('Выберите модель из /Models'); return; }
        const prompt = document.getElementById('prompt');
        const text = prompt.value.trim();
        if(!text) return;
        prompt.value='';
        addMsg(text,'user');
        try{
            const res = await fetch('aicore.php', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ model: model, user: text, temperature: 1.0, top_k: 20, max_tokens: 60 })
            });
            const data = await res.json();
            addMsg(data.reply || ('[error] '+(data.error||'unknown')), 'bot');
        }catch(e){ addMsg('[network error] '+e, 'bot'); }
    }
    document.getElementById('send').onclick = send;
    document.getElementById('clear').onclick = function(){ document.getElementById('chat').innerHTML=''; };
    document.getElementById('prompt').addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
</script>
</body>
</html>