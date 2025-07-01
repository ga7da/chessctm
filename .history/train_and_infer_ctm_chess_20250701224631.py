# train_and_infer_ctm_chess.py
#
# Self-Play RL fine-tuning of a CTM chess engine on CUDA,
# с учётом трёхкратного правила, insufficient-material draw,
# полного словаря ходов и возможностью возобновить с последнего чекпоинта.
#
# Требования:
#   pip install torch python-chess tqdm

import os
import re
import time
from collections import defaultdict

import chess
import chess.pgn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from models.ctm import ContinuousThoughtMachine

# ========== 1) Гиперпараметры ==========
BATCH_SIZE   = 4      # параллельных self-play партий
RL_ITERS     = 5000   # число итераций RL
SAVE_EVERY   = 10     # чекпоинт каждые N итераций
EVAL_EVERY   = 10     # PGN каждые N итераций
LR           = 1e-4   # learning rate
TEMPERATURE  = 1.0    # температура для стохастического выбора
MAX_PLY      = 50     # ограничение длины партии (в полу-ходах)
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 1.a) Информация о девайсе ==========
print(f"CUDA available: {torch.cuda.is_available()} – Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB total")

# ========== 2) Построение словаря всех возможных UCI-ходов ==========
def build_move_vocab():
    files = 'abcdefgh'
    ranks = '12345678'
    m2i, i2m = {}, {}
    idx = 0
    # тихие и capture-ходы без промоции
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    u = f1 + r1 + f2 + r2
                    m2i[u] = idx
                    i2m[idx] = u
                    idx += 1
    # промоции белых (включая capture-промоции)
    for f in files:
        for promo in ('q','r','b','n'):
            for df in (-1, 0, +1):
                j = files.index(f) + df
                if 0 <= j < 8:
                    dest = files[j]
                    u = f + '7' + dest + '8' + promo
                    m2i[u] = idx
                    i2m[idx] = u
                    idx += 1
    # промоции чёрных
    for f in files:
        for promo in ('q','r','b','n'):
            for df in (-1, 0, +1):
                j = files.index(f) + df
                if 0 <= j < 8:
                    dest = files[j]
                    u = f + '2' + dest + '1' + promo
                    m2i[u] = idx
                    i2m[idx] = u
                    idx += 1
    return m2i, i2m

move2idx, idx2move = build_move_vocab()
C = len(move2idx)

# ========== 3) Кодирование позиции ==========
def encode_board(board: chess.Board) -> torch.Tensor:
    M = torch.zeros(12, 8, 8, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        base = 0 if piece.color else 6
        ch   = base + (piece.piece_type - 1)
        r    = chess.square_rank(sq)
        f    = chess.square_file(sq)
        M[ch, r, f] = 1.0
    return M

# ========== 4) Настройка CTM и Optimizer ==========
T, S = 16, 1
model = ContinuousThoughtMachine(
    iterations=T,
    d_model=512,
    d_input=12*8*8,
    heads=8,
    n_synch_out=256,
    n_synch_action=256,
    synapse_depth=4,
    memory_length=8,
    deep_nlms=True,
    memory_hidden_dims=64,
    do_layernorm_nlm=False,
    backbone_type='none',
    positional_embedding_type='none',
    out_dims=S*C,
    prediction_reshaper=[S, C],
    dropout=0.1,
    neuron_select_type='random-pairing',
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ========== 5) Resume from last checkpoint ==========
def get_latest_checkpoint():
    files = [f for f in os.listdir('.') if re.match(r'ctm_rl_ckpt_(\d+)\.pt', f)]
    if not files:
        return None, 1
    iters = [int(re.match(r'ctm_rl_ckpt_(\d+)\.pt', f).group(1)) for f in files]
    last = max(iters)
    return f'ctm_rl_ckpt_{last}.pt', last + 1

start_iter = 1
ckpt_file, start_iter = get_latest_checkpoint()
if ckpt_file:
    print(f"Loading checkpoint {ckpt_file}, resuming from iter {start_iter}")
    ckpt = torch.load(ckpt_file, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        print("  (no optimizer state found — optimizer starts fresh)")
model.train()

# ========== 6) Self-play with threefold & insufficient-material ==========
def self_play_batch(bs: int):
    games, lengths = [], []
    for _ in range(bs):
        board = chess.Board()
        traj = []
        pos_counts = defaultdict(int)
        repeat_penalty = False

        while (not board.is_game_over(claim_draw=True)
               and len(traj) < MAX_PLY):
            # insufficient material draw
            if board.is_insufficient_material():
                tqdm.write("Insufficient material: draw")
                res = '1/2-1/2'
                break

            # threefold repetition
            fen = board.fen()
            pos_counts[fen] += 1
            if pos_counts[fen] >= 3:
                repeat_penalty = True
                tqdm.write("Repetition penalty applied")
                res = '1/2-1/2'
                break

            # forward
            X = encode_board(board).unsqueeze(0).to(DEVICE)
            logits_raw, cert, _ = model(X)
            _, _, TT = logits_raw.shape
            logits = logits_raw.view(1, S, C, TT)

            tick = cert[0,1].argmax().item()
            step_logits = logits[0,0,:,tick]

            # mask illegal
            mask = torch.full((C,), float('-inf'), device=DEVICE)
            for mv in board.legal_moves:
                mask[move2idx[mv.uci()]] = 0.0

            # sample
            probs = F.softmax((step_logits + mask) / TEMPERATURE, dim=-1)
            dist  = Categorical(probs)
            a     = dist.sample()
            logp  = dist.log_prob(a)

            traj.append((logp, board.turn))
            board.push(chess.Move.from_uci(idx2move[a.item()]))

        else:
            # loop ended normally
            res = board.result(claim_draw=True)

        # assign reward
        if repeat_penalty:
            Rw = Rb = -0.5
        else:
            if   res == '1-0': Rw, Rb = +1.0, -1.0
            elif res == '0-1': Rw, Rb = -1.0, +1.0
            else:              Rw, Rb = 0.0, 0.0

        games.append([(lp, Rw if pl else Rb) for lp, pl in traj])
        lengths.append(len(traj))

    return games, sum(lengths) / len(lengths)

# ========== 7) REINFORCE update ==========
def reinforce_update(games):
    logps, returns = [], []
    for g in games:
        for lp, r in g:
            logps.append(lp)
            returns.append(r)
    R = torch.tensor(returns, device=DEVICE)
    baseline = R.mean().detach()

    loss = 0.0
    for lp, r in zip(logps, R):
        loss += -lp * (r - baseline)
    loss = loss / len(games)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), baseline.item()

# ========== 8) Inference → PGN ==========
def play_and_return_pgn():
    model.eval()
    board = chess.Board()
    game  = chess.pgn.Game()
    node  = game
    pos_counts = defaultdict(int)

    while not board.is_game_over(claim_draw=True):
        if board.is_insufficient_material():
            tqdm.write("Insufficient material in inference: draw")
            break
        fen = board.fen()
        pos_counts[fen] += 1
        if pos_counts[fen] >= 3:
            tqdm.write("Repetition in inference: draw")
            break

        X = encode_board(board).unsqueeze(0).to(DEVICE)
        logits_raw, cert, _ = model(X)
        _, _, TT = logits_raw.shape
        logits = logits_raw.view(1, S, C, TT)

        tick = cert[0,1].argmax().item()
        step_logits = logits[0,0,:,tick]

        mask = torch.full((C,), float('-inf'), device=DEVICE)
        for mv in board.legal_moves:
            mask[move2idx[mv.uci()]] = 0.0

        probs = F.softmax(step_logits + mask, dim=-1)
        idx = Categorical(probs).sample().item()
        mv  = idx2move[idx]

        board.push(chess.Move.from_uci(mv))
        node = node.add_main_variation(chess.Move.from_uci(mv))

    game.headers["Result"] = board.result(claim_draw=True)
    model.train()
    return game

# ========== 9) Главный цикл RL ==========
if __name__ == '__main__':
    pbar = tqdm(total=RL_ITERS, initial=start_iter-1,
                desc='RL Training', unit='iter')
    for it in range(start_iter, RL_ITERS+1):
        t0 = time.time()
        games, avg_len = self_play_batch(BATCH_SIZE)
        loss, baseline = reinforce_update(games)
        dt = time.time() - t0

        if DEVICE.type == 'cuda':
            alloc    = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved()  / 1e9
            pbar.set_postfix(loss=f'{loss:.3f}',
                             baseline=f'{baseline:.3f}',
                             avg_len=f'{avg_len:.1f}',
                             t=f'{dt:.2f}s',
                             alloc=f'{alloc:.2f}G',
                             res=f'{reserved:.2f}G')
        else:
            pbar.set_postfix(loss=f'{loss:.3f}',
                             baseline=f'{baseline:.3f}',
                             avg_len=f'{avg_len:.1f}',
                             t=f'{dt:.2f}s')

        if it % SAVE_EVERY == 0:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            fname = f'ctm_rl_ckpt_{it}.pt'
            torch.save(ckpt, fname)
            tqdm.write(f'[Iter {it}] Model & Optimizer saved → {fname}')

        if it == 1 or it % EVAL_EVERY == 0:
            pgn = play_and_return_pgn()
            fname = f'ctm_game_{it}.pgn'
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(str(pgn))
            tqdm.write(f'[Iter {it}] PGN saved → {fname}')

        pbar.update(1)

    pbar.close()
    print('RL fine-tuning complete.')