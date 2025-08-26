<?php
/*
 * PHPAiModel-RNN — generator_weights.php
 * Weights generator for RNN: builds initial matrices and training data from datasets (RU/EN).
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
@ini_set('memory_limit','1024M');
@set_time_limit(0);

function format_hms(float $sec): string {
    $sec = max(0, (int)round($sec));
    $h = intdiv($sec, 3600);
    $m = intdiv($sec % 3600, 60);
    $s = $sec % 60;
    if ($h > 0) return sprintf('%02d:%02d:%02d', $h, $m, $s);
    return sprintf('%02d:%02d', $m, $s);
}

$action = $_GET['action'] ?? '';
if ($action === 'run') {
    // Настраиваем заголовки и отключаем буферизацию для потокового вывода
    header('Content-Type: text/plain; charset=utf-8');
    header('Cache-Control: no-cache, no-transform');
    header('X-Accel-Buffering: no'); // для nginx
    if (function_exists('apache_setenv')) @apache_setenv('no-gzip', '1');
    @ini_set('output_buffering', 'off');
    @ini_set('zlib.output_compression', '0');
    while (ob_get_level() > 0) { @ob_end_flush(); }
    @ob_implicit_flush(true);

    $dataset = basename((string)($_GET['dataset'] ?? 'greetings_ru_en.txt'));
    $H       = max(8,  (int)($_GET['hidden'] ?? 64));
    $SEQ     = max(4,  (int)($_GET['seq'] ?? 16));
    $EPOCHS  = max(1,  (int)($_GET['epochs'] ?? 5));
    $LR      = max(0.0001,(float)($_GET['lr'] ?? 0.05));

    $dsPath = __DIR__ . '/Datasets/' . $dataset;
    if (!is_file($dsPath)) { echo "Dataset not found: $dataset\n"; flush(); exit; }

    $t_start = microtime(true);

    $text = file_get_contents($dsPath);
    $tokens = build_tokens($text);
    [$vocab,$ivocab] = build_vocab($tokens);
    $V = count($vocab);
    $Ntokens = count($tokens);

    // Подготовим id-шки и матрицы
    $ids = array_map(fn($t)=>$vocab[$t], $tokens);
    $Wxh = rand_matrix($H,$V, 0.05);
    $Whh = rand_matrix($H,$H, 0.05);
    $Why = rand_matrix($V,$H, 0.05);
    $bh  = array_fill(0,$H,0.0);
    $by  = array_fill(0,$V,0.0);
    $mWxh = zeros_mat($H,$V); $mWhh=zeros_mat($H,$H); $mWhy=zeros_mat($V,$H);
    $mbh = zeros_vec($H); $mby = zeros_vec($V);

    // Оценим общее количество шагов для % и ETA
    $stepsPerEpoch = (int) floor(max(0, count($ids) - $SEQ) / max(1, $SEQ));
    $totalSteps = max(1, $stepsPerEpoch * $EPOCHS);
    $doneSteps  = 0;

    // Шапка
    echo "Dataset: $dataset\nTokens: $Ntokens\nVocab: $V\nH: $H  SEQ: $SEQ  Epochs: $EPOCHS  LR: $LR\n";
    echo "Planned steps: $totalSteps (≈ $stepsPerEpoch / epoch)\n\n";
    flush();

    // Обучение
    for($epoch=1;$epoch<=$EPOCHS;$epoch++){
        $loss_sum=0.0; $nsteps=0; $hprev=array_fill(0,$H,0.0);

        for($pos=0; $pos+$SEQ < count($ids); $pos += $SEQ){
            $inputs  = array_slice($ids,$pos,$SEQ);
            $targets = array_slice($ids,$pos+1,$SEQ);

            [$loss,$grads,$hprev] = bptt($inputs,$targets,$hprev,$Wxh,$Whh,$Why,$bh,$by,$V);
            $loss_sum += $loss; $nsteps++; $doneSteps++;

            // Adagrad
            for($i=0;$i<$H;$i++){
                for($j=0;$j<$V;$j++){
                    $mWxh[$i][$j] += $grads['dWxh'][$i][$j]**2;
                    $Wxh[$i][$j]  -= $LR*$grads['dWxh'][$i][$j]/(1e-8+sqrt($mWxh[$i][$j]));
                }
                for($j=0;$j<$H;$j++){
                    $mWhh[$i][$j] += $grads['dWhh'][$i][$j]**2;
                    $Whh[$i][$j]  -= $LR*$grads['dWhh'][$i][$j]/(1e-8+sqrt($mWhh[$i][$j]));
                }
            }
            for($i=0;$i<$V;$i++){
                for($j=0;$j<$H;$j++){
                    $mWhy[$i][$j] += $grads['dWhy'][$i][$j]**2;
                    $Why[$i][$j]  -= $LR*$grads['dWhy'][$i][$j]/(1e-8+sqrt($mWhy[$i][$j]));
                }
                $mby[$i] += $grads['dby'][$i]**2;
                $by[$i]  -= $LR*$grads['dby'][$i]/(1e-8+sqrt($mby[$i]));
            }
            for($i=0;$i<$H;$i++){
                $mbh[$i] += $grads['dbh'][$i]**2;
                $bh[$i]  -= $LR*$grads['dbh'][$i]/(1e-8+sqrt($mbh[$i]));
            }

            // Каждые ~50 шагов — обновляем прогресс (проценты + ETA)
            if ($doneSteps % 50 === 0 || $doneSteps === $totalSteps) {
                $elapsed = microtime(true) - $t_start;
                $rate    = $doneSteps > 0 ? ($elapsed / $doneSteps) : 0.0;
                $remainS = max(0.0, ($totalSteps - $doneSteps) * $rate);
                $percent = round(($doneSteps / $totalSteps) * 100, 2);
                $eta     = format_hms($remainS);
                $tspent  = format_hms($elapsed);
                $avgLoss = $nsteps ? $loss_sum / $nsteps : 0.0;

                echo sprintf(
                    "Progress: %6.2f%% | ETA %s | Spent %s | epoch %d/%d | step %d/%d | avg loss %.5f\n",
                    $percent, $eta, $tspent, $epoch, $EPOCHS, $nsteps, $stepsPerEpoch, $avgLoss
                );
                flush();
            }
        }

        $avg = $nsteps? $loss_sum/$nsteps : 0.0;
        echo "Epoch $epoch/$EPOCHS done | avg loss=".round($avg,5)." | steps=$nsteps\n\n";
        flush();
    }

    // Сохранение модели
    if(!is_dir(__DIR__.'/Models')) @mkdir(__DIR__.'/Models',0777,true);
    $fname = 'rnn_'.pathinfo($dataset,PATHINFO_FILENAME).'_H'.$H.'_'.date('Ymd_His').'.json';
    $out = [
        'V'=>$V,'H'=>$H,'vocab'=>$vocab,'ivocab'=>$ivocab,
        'Wxh'=>$Wxh,'Whh'=>$Whh,'Why'=>$Why,'bh'=>$bh,'by'=>$by,
        'meta'=>[
            'dataset'=>$dataset,'epochs'=>$EPOCHS,'seq'=>$SEQ,'lr'=>$LR,'time'=>date('c'),
            'train_seconds'=> round(microtime(true)-$t_start,3),
            'tokens'=>$Ntokens
        ]
    ];
    file_put_contents(__DIR__.'/Models/'.$fname, json_encode($out, JSON_UNESCAPED_UNICODE));

    $totalSpent = format_hms(microtime(true)-$t_start);
    echo "DONE 100.00% | Total time $totalSpent\n";
    echo "Saved: Models/$fname\n";
    flush();
    exit;
}

// ---------- UI ----------
$datasetsDir = __DIR__ . '/Datasets';
if (!is_dir($datasetsDir)) { @mkdir($datasetsDir, 0777, true); }
$datasets = array_values(array_filter(is_dir($datasetsDir) ? scandir($datasetsDir) : [], fn($f)=>preg_match('/\.txt$/i',$f)));
$modelsDir = __DIR__ . '/Models';
if (!is_dir($modelsDir)) { @mkdir($modelsDir, 0777, true); }
$models = array_values(array_filter(is_dir($modelsDir) ? scandir($modelsDir) : [], fn($f)=>preg_match('/\.json$/i',$f)));
?>
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>RNN Trainer — generator_weights.php</title>
        <style>
            :root{ --bg:#f7f8fb; --card:#ffffff; --text:#0f172a; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }
            *{box-sizing:border-box}
            body{margin:0;background:var(--bg);color:var(--text);font-family:System-ui,-apple-system,Segoe UI,Roboto,Inter,Arial}
            header{background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px}
            main{max-width:980px;margin:0 auto;padding:16px}
            .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
            select,input,button{border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;background:#fff;color:var(--text)}
            button{background:var(--accent);color:#fff;border-color:transparent;cursor:pointer}
            pre{white-space:pre-wrap;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px;min-height:200px;max-height:60vh;overflow:auto}
            table{width:100%;border-collapse:collapse;margin-top:12px}
            th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;font-size:14px}
            .hint{font-size:12px;color:var(--muted)}
        </style>
    </head>
    <body>
    <header>
        <b>RNN Trainer</b> — интерфейс обучения. После сохранения модели откройте <code>index.php</code> и выберите её в чате.
    </header>
    <main>
        <section>
            <div class="row">
                <label>Dataset</label>
                <select id="dataset">
                    <?php foreach($datasets as $f): ?>
                        <option><?= htmlspecialchars($f, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?></option>
                    <?php endforeach; ?>
                </select>
                <label>Hidden</label><input id="hidden" type="number" value="64" min="8" max="256" style="width:90px">
                <label>SeqLen</label><input id="seq" type="number" value="16" min="4" max="64" style="width:90px">
                <label>Epochs</label><input id="epochs" type="number" value="5" min="1" max="100" style="width:90px">
                <label>LR</label><input id="lr" type="number" step="0.001" value="0.05" style="width:90px">
                <button id="run">Training</button>
            </div>
            <p class="hint">Тренировка запускается сервером (PHP). Логи (проценты, ETA, потеря) выводятся ниже в реальном времени. Файл модели сохраняется в <code>/Models</code>.</p>
            <pre id="log">Готов к обучению…</pre>
        </section>

        <section>
            <h3>Доступные модели</h3>
            <table>
                <thead><tr><th>Файл</th><th>Размер</th><th>Дата</th></tr></thead>
                <tbody>
                <?php foreach($models as $m): $p=$modelsDir.'/'.$m; ?>
                    <tr>
                        <td><?= htmlspecialchars($m, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?></td>
                        <td><?= number_format(filesize($p) ?: 0, 0, '.', ' ') ?> B</td>
                        <td><?= date('Y-m-d H:i:s', filemtime($p) ?: time()) ?></td>
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        </section>
    </main>
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
        async function run(){
            const params = new URLSearchParams({
                action:'run',
                dataset: document.getElementById('dataset').value,
                hidden:  document.getElementById('hidden').value,
                seq:     document.getElementById('seq').value,
                epochs:  document.getElementById('epochs').value,
                lr:      document.getElementById('lr').value
            });
            const url = 'generator_weights.php?' + params.toString();

            const logEl = document.getElementById('log');
            logEl.textContent = 'Запуск обучения…\n';
            try {
                const res = await fetch(url, {cache:'no-store'});
                if (!res.body) {
                    const txt = await res.text();
                    logEl.textContent += txt + '\n';
                    return;
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let done, value;
                while (true) {
                    ({done, value} = await reader.read());
                    if (done) break;
                    logEl.textContent += decoder.decode(value, {stream:true});
                    logEl.scrollTop = logEl.scrollHeight;
                }
            } catch (e) {
                logEl.textContent += '\\nОшибка: ' + (e && e.message ? e.message : e) + '\\n';
            }
            // Обновить список моделей после окончания
            setTimeout(()=>location.reload(), 600);
        }
        document.getElementById('run').onclick = run;
    </script>
    </body>
    </html>

<?php
// ------- Trainer helpers below --------
function build_tokens($text) {
    $text = str_replace(["\r"], '', $text); // Удаляем \r
    $text = preg_replace('/([.,!?;:\\-])/u', ' $1 ', $text); // Пунктуация как токены
    $text = str_replace("\n", ' <NL> ', trim($text)); // Сохраняем \n как <NL>
    $text = "<BOS> $text <EOS>"; // Добавляем BOS/EOS
    $text = preg_replace('/\s+/u', ' ', $text); // Нормализация пробелов
    return explode(' ', $text);
}
function build_vocab($tokens){ $tok2id=[]; $id2tok=[]; $i=0; foreach($tokens as $t){ if(!isset($tok2id[$t])){ $tok2id[$t]=$i; $id2tok[$i]=$t; $i++; } } return [$tok2id,$id2tok]; }
function rand_matrix($r,$c,$scale){ $M=[]; for($i=0;$i<$r;$i++){ $row=[]; for($j=0;$j<$c;$j++){ $row[]=(mt_rand()/mt_getrandmax()*2-1)*$scale; } $M[]=$row; } return $M; }
function zeros_mat($r,$c){ $M=[]; for($i=0;$i<$r;$i++){ $M[] = array_fill(0,$c,0.0); } return $M; }
function zeros_vec($n){ return array_fill(0,$n,0.0); }
function tanh_arr($v){ $o=[]; foreach($v as $x){ $o[] = tanh($x); } return $o; }
function add_vec($a,$b){ $n=count($a); $o=[]; for($i=0;$i<$n;$i++) $o[$i]=$a[$i]+$b[$i]; return $o; }
function matvec($M,$v){ $r=count($M); $c=count($v); $o=[]; for($i=0;$i<$r;$i++){ $s=0.0; for($j=0;$j<$c;$j++){ $s += $M[$i][$j]*$v[$j]; } $o[$i]=$s; } return $o; }
function softmax($v){ $mx=max($v); $ex=[]; $sum=0.0; foreach($v as $x){ $e=exp($x-$mx); $ex[]=$e; $sum+=$e; } if($sum<=0)$sum=1.0; foreach($ex as $i=>$e){ $ex[$i]=$e/$sum; } return $ex; }
function onehot_idx($idx,$V){ $v=array_fill(0,$V,0.0); $v[$idx]=1.0; return $v; }
function bptt($inputs,$targets,$hprev,$Wxh,$Whh,$Why,$bh,$by,$V){
    $H = count($bh);
    $xs=$hs=$ys=$ps=[]; $hs[-1]=$hprev; $loss=0.0;
    for($t=0;$t<count($inputs);$t++){
        $xs[$t] = onehot_idx($inputs[$t],$V);
        $hs[$t] = tanh_arr( add_vec( matvec($Wxh,$xs[$t]), add_vec(matvec($Whh,$hs[$t-1]), $bh) ) );
        $ys[$t] = add_vec(matvec($Why,$hs[$t]), $by);
        $ps[$t] = softmax($ys[$t]);
        $loss  += -log(max($ps[$t][$targets[$t]], 1e-12));
    }
    $dWxh=zeros_mat($H,$V); $dWhh=zeros_mat($H,$H); $dWhy=zeros_mat($V,$H); $dbh=zeros_vec($H); $dby=zeros_vec($V);
    $dh_next = zeros_vec($H);
    for($t=count($inputs)-1;$t>=0;$t--){
        $dy = $ps[$t]; $dy[$targets[$t]] -= 1.0;
        for($i=0;$i<$V;$i++){
            for($j=0;$j<$H;$j++){ $dWhy[$i][$j] += $dy[$i]*$hs[$t][$j]; }
            $dby[$i] += $dy[$i];
        }
        $dh = zeros_vec($H);
        for($j=0;$j<$H;$j++){ $s=0.0; for($i=0;$i<$V;$i++){ $s += $Why[$i][$j]*$dy[$i]; } $dh[$j] = $s + $dh_next[$j]; }
        $dhraw=$dh; for($j=0;$j<$H;$j++){ $dhraw[$j] = (1 - $hs[$t][$j]*$hs[$t][$j]) * $dh[$j]; }
        for($j=0;$j<$H;$j++){ $dbh[$j] += $dhraw[$j]; }
        for($i=0;$i<$H;$i++){
            for($j=0;$j<$V;$j++){ $dWxh[$i][$j] += $dhraw[$i]*$xs[$t][$j]; }
            for($j=0;$j<$H;$j++){ $dWhh[$i][$j] += $dhraw[$i]*$hs[$t-1][$j]; }
        }
        $dh_next = zeros_vec($H);
        for($j=0;$j<$H;$j++){ $s=0.0; for($i=0;$i<$H;$i++){ $s += $Whh[$i][$j]*$dhraw[$i]; } $dh_next[$j] = $s; }
    }
    $clip=5.0; $dbh=array_map(fn($x)=>max(-$clip,min($clip,$x)),$dbh); $dby=array_map(fn($x)=>max(-$clip,min($clip,$x)),$dby);
    for($i=0;$i<$H;$i++){
        for($j=0;$j<$V;$j++){ $dWxh[$i][$j]=max(-$clip,min($clip,$dWxh[$i][$j])); }
        for($j=0;$j<$H;$j++){ $dWhh[$i][$j]=max(-$clip,min($clip,$dWhh[$i][$j])); }
    }
    for($i=0;$i<$V;$i++){ for($j=0;$j<$H;$j++){ $dWhy[$i][$j]=max(-$clip,min($clip,$dWhy[$i][$j])); } }
    return [$loss, ['dWxh'=>$dWxh,'dWhh'=>$dWhh,'dWhy'=>$dWhy,'dbh'=>$dbh,'dby'=>$dby], $hs[count($inputs)-1]];
}
?>