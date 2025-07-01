#!/usr/bin/env python3
# train_and_infer_ctm_chess_tpu.py
#
# Self-Play RL fine-tuning of CTM Chess Engine on Google Cloud TPU (4 cores),
# always from scratch, без загрузки старых чекпоинтов.

import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_NUM_DEVICES", "4")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "1")

import re
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import chess
import chess.pgn

from models.ctm import ContinuousThoughtMachine

# Гиперпараметры
BATCH_SIZE  = 8
RL_ITERS    = 5000
SAVE_EVERY  = 2500
EVAL_EVERY  = 2500
LR          = 1e-4
TEMPERATURE = 1.0
MAX_PLY     = 50
TICKS, S    = 64, 1

def build_move_vocab():
    files, ranks = "abcdefgh", "12345678"
    m2i, i2m, idx = {}, {}, 0
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    u = f1 + r1 + f2 + r2
                    m2i[u], i2m[idx] = idx, u; idx += 1
    for f in files:
        for promo in ("q","r","b","n"):
            for df in (-1,0,1):
                j = files.index(f) + df
                if 0 <= j < 8:
                    dest = files[j]
                    u = f + "7" + dest + "8" + promo
                    m2i[u], i2m[idx] = idx, u; idx += 1
    for f in files:
        for promo in ("q","r","b","n"):
            for df in (-1,0,1):
                j = files.index(f) + df
                if 0 <= j < 8:
                    dest = files[j]
                    u = f + "2" + dest + "1" + promo
                    m2i[u], i2m[idx] = idx, u; idx += 1
    return m2i, i2m

move2idx, idx2move = build_move_vocab()
C = len(move2idx)

def encode_board(board: chess.Board) -> torch.Tensor:
    M = torch.zeros(12, 8, 8, dtype=torch.float32)
    for sq, p in board.piece_map().items():
        base = 0 if p.color else 6
        ch   = base + (p.piece_type - 1)
        r, f = chess.square_rank(sq), chess.square_file(sq)
        M[ch, r, f] = 1.0
    return M

def train_fn(index, flags=None):
    device = xm.xla_device()
    print(f"[core {index}] Using device: {device}", flush=True)

    model = ContinuousThoughtMachine(
        iterations=TICKS,
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
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=S*C,
        prediction_reshaper=[S, C],
        dropout=0.1,
        neuron_select_type="random-pairing",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # всегда стартуем с итерации 1
    start_iter = 1
    model.train()

    for it in range(start_iter, RL_ITERS + 1):
        # 1) Self-play
        games = []
        for _ in range(BATCH_SIZE):
            board = chess.Board()
            traj = []
            pos_counts = defaultdict(int)
            repeat_penalty = False

            while not board.is_game_over(claim_draw=True) \
                  and len(traj) < MAX_PLY:
                if board.is_insufficient_material():
                    res = "1/2-1/2"; break
                fen = board.fen()
                pos_counts[fen] += 1
                if pos_counts[fen] >= 3:
                    repeat_penalty = True; res = "1/2-1/2"; break

                X = encode_board(board).unsqueeze(0).to(device)
                logits_raw, cert, _ = model(X)
                logits = logits_raw.view(1, S, C, TICKS)[0, 0]
                tick = cert[0,1].argmax().item()
                step_logits = logits[:, tick]

                mask = torch.full((C,), -1e9, device=device)
                for mv in board.legal_moves:
                    mask[move2idx[mv.uci()]] = 0.0
                probs = F.softmax((step_logits + mask)/TEMPERATURE, dim=-1)
                dist  = Categorical(probs)
                a     = dist.sample()
                logp  = dist.log_prob(a)

                traj.append((logp, board.turn))
                board.push(chess.Move.from_uci(idx2move[a.item()]))
            else:
                res = board.result(claim_draw=True)

            if repeat_penalty:
                Rw = Rb = -0.5
            else:
                if   res=="1-0": Rw,Rb = +1, -1
                elif res=="0-1": Rw,Rb = -1, +1
                else:            Rw,Rb = 0, 0

            games.append([(lp, Rw if pl else Rb) for lp,pl in traj])

        # 2) REINFORCE loss
        logps, returns = [], []
        for g in games:
            for lp, r in g:
                logps.append(lp); returns.append(r)
        R = torch.tensor(returns, device=device)
        baseline = R.mean().detach()
        loss = sum(-lp*(r-baseline) for lp,r in zip(logps, R)) / len(games)

        # 3) Backward + step
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)

        # 4) Сохраняем чекпоинт
        if it % SAVE_EVERY == 0 or it == RL_ITERS:
            ck = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            fname = f"ctm_rl_ckpt_{it}.pt"
            torch.save(ck, fname)
            print(f"[core {index}] Saved checkpoint → {fname}", flush=True)

        # 5) Eval → PGN
        if it % EVAL_EVERY == 0 or it == 1:
            model.eval()
            b = chess.Board()
            game = chess.pgn.Game(); node = game
            pos_counts = defaultdict(int)
            while not b.is_game_over(claim_draw=True):
                X = encode_board(b).unsqueeze(0).to(device)
                logits_raw, cert, _ = model(X)
                logits = logits_raw.view(1,S,C,TICKS)[0,0]
                tick = cert[0,1].argmax().item()
                step_logits = logits[:,tick]

                mask = torch.full((C,), -1e9, device=device)
                for mv in b.legal_moves:
                    mask[move2idx[mv.uci()]] = 0.0
                idxm = int((step_logits+mask).argmax().item())
                mv   = idx2move[idxm]
                b.push(chess.Move.from_uci(mv))
                node = node.add_main_variation(chess.Move.from_uci(mv))

            game.headers["Result"] = b.result(claim_draw=True)
            pgn = f"ctm_game_{it}.pgn"
            with open(pgn,"w") as f:
                f.write(str(game))
            print(f"[core {index}] Saved PGN → {pgn}", flush=True)
            model.train()

        if it % 10 == 0:
            print(f"[core {index}] iter {it}/{RL_ITERS} "
                  f"loss={loss:.3f} baseline={baseline:.3f}",
                  flush=True)

    print(f"[core {index}] Training complete.", flush=True)


if __name__ == "__main__":
    xmp.spawn(train_fn, args=(), nprocs=None)