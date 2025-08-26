<?php
/*
 * PHPAiModel-RNN — aicore.php
 * Core runtime of simple RNN in PHP: matrix multiplications, forward/backward pass, chat inference.
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


declare(strict_types=1);
@session_start();

// ---- Error policy: логируем, но не отображаем ----
@ini_set('display_errors','0');
@ini_set('log_errors','1');
@ini_set('error_log', __DIR__ . '/php_errors.log');
error_reporting(E_ALL);

header('Content-Type: application/json; charset=utf-8');

function json_out(array $data, int $code=200): void {
    http_response_code($code);
    echo json_encode($data, JSON_UNESCAPED_UNICODE);
    exit;
}

try {
    $raw  = file_get_contents('php://input') ?: '';
    $body = json_decode($raw, true);
    if (!is_array($body)) {
        json_out(['ok'=>false,'error'=>'bad_json','hint'=>'Request body must be JSON','raw'=>$raw], 400);
    }

    $model_file = (string)($body['model'] ?? '');
    if ($model_file === '') json_out(['ok'=>false,'error'=>'model is required'], 400);

    $model_path = __DIR__ . '/Models/' . basename($model_file);
    if (!is_file($model_path)) json_out(['ok'=>false,'error'=>'model_not_found','path'=>basename($model_path)], 404);

    $payload_str = file_get_contents($model_path);
    if ($payload_str === false) json_out(['ok'=>false,'error'=>'read_failed'], 500);
    $payload = json_decode($payload_str, true);
    if (!is_array($payload)) json_out(['ok'=>false,'error'=>'model_json_invalid','msg'=>json_last_error_msg()], 500);

    $V  = (int)($payload['V'] ?? 0);
    $H  = (int)($payload['H'] ?? 0);
    $Wxh= $payload['Wxh'] ?? null; // [H][V]
    $Whh= $payload['Whh'] ?? null; // [H][H]
    $Why= $payload['Why'] ?? null; // [V][H]
    $bh = $payload['bh']  ?? null; // [H]
    $by = $payload['by']  ?? null; // [V]
    $vocab = $payload['vocab'] ?? null; // token=>id
    $ivocab= $payload['ivocab'] ?? null; // id=>token

    if ($V<=0 || $H<=0 || !$Wxh || !$Whh || !$Why || !$bh || !$by || !$vocab || !$ivocab) {
        json_out(['ok'=>false,'error'=>'model_fields_missing'], 500);
    }

    $temperature = max(0.1, (float)($body['temperature'] ?? 0.7)); // Уменьшаем до 0.7
    $top_k       = max(1,   (int)($body['top_k'] ?? 5)); // Уменьшаем до 5
    $max_tokens  = max(1,   (int)($body['max_tokens'] ?? 20)); // Уменьшаем до 20 для коротких ответов
    $user_msg    = (string)($body['user'] ?? '');

    $_SESSION['history'] = $_SESSION['history'] ?? [];
    if ($user_msg !== '') {
        $_SESSION['history'][] = $user_msg;
        if (count($_SESSION['history'])>20) array_shift($_SESSION['history']);
    }
    $context_text = trim(implode(" <NL> ", $_SESSION['history'])); // Используем <NL> для истории
    $tokens = tokenize($context_text);
    $ids = [];
    foreach ($tokens as $t) { if (isset($vocab[$t])) $ids[] = $vocab[$t]; }
    $h = zeros($H);
    foreach ($ids as $id) { [$h, $_] = rnn_step($id, $h, $Wxh, $Whh, $bh, $Why, $by, $V); }

// Начинаем генерацию с <A>
    $gen = ['<A>'];
    $last_id = $vocab['<A>'] ?? $vocab['<BOS>']; // Стартуем с <A> или <BOS>
    for ($i = 0; $i < $max_tokens; $i++) {
        [$h, $probs] = rnn_step($last_id, $h, $Wxh, $Whh, $bh, $Why, $by, $V, $temperature);
        $last_id = sample_topk($probs, $top_k);
    $token = $ivocab[$last_id] ?? '';
    $gen[] = $token;
    if ($token === '<NL>' || $token === '<EOS>') break; // Останавливаемся на <NL> или <EOS>
}
    $reply = detokenize($gen);
    $_SESSION['history'][] = $reply;

    json_out(['ok' => true, 'reply' => $reply, 'tokens_generated' => count($gen)]);

} catch (Throwable $e) {
    // В случае любой непойманной ошибки — вернуть JSON и написать в лог
    error_log('[aicore.php] '.$e->getMessage().' @ '.$e->getFile().':'.$e->getLine());
    json_out(['ok'=>false,'error'=>'server_exception','msg'=>$e->getMessage()], 500);
}

// ---- helpers ----
function zeros(int $n): array { return array_fill(0,$n,0.0); }
function tanh_arr(array $v): array { $o=[]; foreach($v as $x){ $o[] = tanh((float)$x); } return $o; }
function add_vec(array $a,array $b): array { $o=[]; $n=count($a); for($i=0;$i<$n;$i++) $o[$i]=((float)$a[$i])+((float)$b[$i]); return $o; }
function matvec(array $M,array $v): array { $r=[]; $rows=count($M); $cols=count($v); for($i=0;$i<$rows;$i++){ $sum=0.0; for($j=0;$j<$cols;$j++){ $sum += ((float)$M[$i][$j])*((float)$v[$j]); } $r[$i]=$sum; } return $r; }
function vec_softmax(array $v, float $temperature=1.0): array {
    $mx = max($v);
    $ex=[]; $sum=0.0;
    foreach($v as $x){ $e = exp(((float)$x - (float)$mx)/$temperature); $ex[]=$e; $sum+=$e; }
    if($sum<=0) $sum=1.0;
    foreach($ex as $i=>$e){ $ex[$i]=$e/$sum; }
    return $ex;
}
function onehot(int $idx,int $V): array { $v=array_fill(0,$V,0.0); if($idx>=0&&$idx<$V)$v[$idx]=1.0; return $v; }
function rnn_step(int $x_id,array $h_prev,array $Wxh,array $Whh,array $bh,array $Why,array $by,int $V,float $temperature=1.0): array {
    $x = onehot($x_id,$V);
    $hh = add_vec( matvec($Wxh,$x), add_vec(matvec($Whh,$h_prev), $bh) );
    $h = tanh_arr($hh);
    $y = add_vec(matvec($Why,$h), $by);
    $p = vec_softmax($y, $temperature);
    return [$h, $p];
}
function sample_topk(array $probs,int $k): int {
    $pairs=[]; foreach($probs as $i=>$p){ $pairs[]=['i'=>(int)$i,'p'=>(float)$p]; }
    usort($pairs, fn($A,$B)=> $B['p']<=>$A['p']);
    $pairs = array_slice($pairs,0,max(1,$k));
    $sum=0.0; foreach($pairs as $pp){ $sum+=$pp['p']; }
    if($sum<=0) return $pairs[0]['i'];
    $r = mt_rand()/mt_getrandmax();
    $acc=0.0;
    foreach($pairs as $pp){ $acc += $pp['p']/$sum; if($r <= $acc) return $pp['i']; }
    return $pairs[count($pairs)-1]['i'];
}
function tokenize(string $text): array {
    $text = str_replace(["\r"], '', $text);
    $text = preg_replace('/([.,!?;:\\-])/u', ' $1 ', $text);
    $text = str_replace("\n", ' <NL> ', trim($text)); // Сохраняем \n как <NL>
    $text = preg_replace('/\s+/u', ' ', $text);
    if ($text === '') return ['<BOS>'];
    return explode(' ', $text);
}
function detokenize(array $tokens): string {
    $s = trim(implode(' ', $tokens));
    $s = preg_replace('/\s+([.,!?;:])/u', '$1', $s); // Убираем пробелы перед пунктуацией
    $s = str_replace(['<BOS>', '<EOS>', '<NL>', '<Q>', '<A>'], '', $s); // Убираем все служебные токены
    return trim($s);
}
?>