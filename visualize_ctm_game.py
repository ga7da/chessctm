# analyze_ctm_chess.py
#
# Self-play анализ CTM Chess Engine:
#  – attention.gif  — оверлей внимания по ходам
#  – neural.gif     — heatmap K нейронов по ходам
#  – game.pgn       — сыгранная партия
#
# Требования:
#   pip install torch python-chess tqdm matplotlib seaborn imageio opencv-python

import os, re, math
import numpy as np
import torch
import chess
import chess.pgn
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import cv2
from models.ctm import ContinuousThoughtMachine

sns.set_style('darkgrid')

# ------------ параметры ------------
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TICKS, S   = 16, 1    # internal ticks, moves per forward
N_MOVES    = 16       # сколько ходов проиграть
K_NEURONS  = 64       # сколько нейронов для heatmap
OUT_DIR    = 'analysis_chess'
CKPT_PATTERN = r'ctm_rl_ckpt_(\d+)\.pt'

# ------------ словарь ходов ------------
def build_move_vocab():
    files, ranks = 'abcdefgh', '12345678'
    m2i, i2m, idx = {}, {}, 0
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    u = f1+r1+f2+r2
                    m2i[u], i2m[idx] = idx, u
                    idx += 1
    for f in files:
        for promo in ('q','r','b','n'):
            for df in (-1,0,1):
                j = files.index(f)+df
                if 0<=j<8:
                    u = f+'7'+files[j]+'8'+promo
                    m2i[u], i2m[idx] = idx, u
                    idx += 1
    for f in files:
        for promo in ('q','r','b','n'):
            for df in (-1,0,1):
                j = files.index(f)+df
                if 0<=j<8:
                    u = f+'2'+files[j]+'1'+promo
                    m2i[u], i2m[idx] = idx, u
                    idx += 1
    return m2i, i2m

move2idx, idx2move = build_move_vocab()
C = len(move2idx)

# ------------ загрузка CTM ------------
def get_latest_ckpt():
    files = [f for f in os.listdir('.') if re.match(CKPT_PATTERN, f)]
    if not files:
        raise FileNotFoundError("No checkpoint found")
    iters = [int(re.search(r'(\d+)',f).group(1)) for f in files]
    return f'ctm_rl_ckpt_{max(iters)}.pt'

ckpt = get_latest_ckpt()
print("Loading", ckpt)
data = torch.load(ckpt, map_location=DEVICE)
ctm = ContinuousThoughtMachine(
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
    backbone_type='none',
    positional_embedding_type='none',
    out_dims=S*C,
    prediction_reshaper=[S, C],
    dropout=0.1,
    neuron_select_type='random-pairing',
).to(DEVICE)
ctm.load_state_dict(data['model_state_dict'], strict=False)
ctm.eval()

# ------------ вспомогательные ------------
def board_to_rgb(board):
    arr = np.ones((8,8,3), dtype=np.uint8)*255
    cmap = {
        chess.PAWN:   (200,200,200),
        chess.KNIGHT: (150,150,150),
        chess.BISHOP: (100,100,100),
        chess.ROOK:   ( 50, 50, 50),
        chess.QUEEN:  (255,  0,  0),
        chess.KING:   (  0,  0,255),
    }
    for sq,p in board.piece_map().items():
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        col = cmap[p.piece_type]
        if not p.color:
            col = tuple(255-x for x in col)
        arr[r,f] = col
    return arr

def encode_board(board):
    M = np.zeros((12,8,8), dtype=np.float32)
    for sq,p in board.piece_map().items():
        base = 0 if p.color else 6
        ch   = base + (p.piece_type - 1)
        r,f  = chess.square_rank(sq), chess.square_file(sq)
        M[ch,r,f] = 1.0
    return M

# ------------ self-play + сбор ------------
def run_self_play(n_moves):
    board = chess.Board()
    attn_maps = []
    post_acts  = []
    moves      = []
    for _ in range(n_moves):
        X = torch.tensor(encode_board(board)[None]).to(DEVICE)
        with torch.no_grad():
            preds, cert, _, _, post, attn = ctm(X, track=True)
        # attn: np.array (T, B, heads,1,seq_len)
        att = attn[:,0,:,0,:]        # (T, heads, seq_len)
        seq_len = att.shape[-1]
        side = int(math.sqrt(seq_len))
        tick = int(cert[0,1].argmax().item())
        # среднее по голов, reshape в карту
        amap = att[tick].mean(0).reshape(side, side)
        attn_maps.append(amap)
        # пост-активации
        post_np = post[:,0,:]        # (T, d_model)
        post_acts.append(post_np[tick])
        # greedy move
        logits = preds.view(1,S,C,TICKS)[0,0,:,tick]
        mask = torch.full((C,), -1e9, device=DEVICE)
        for mv in board.legal_moves:
            mask[move2idx[mv.uci()]] = 0.0
        idxm = int((logits+mask).argmax().item())
        mv = idx2move[idxm]
        moves.append(mv)
        board.push(chess.Move.from_uci(mv))
        if board.is_game_over(): break
    return moves, np.stack(attn_maps), np.stack(post_acts)

# ------------ визуализация ------------
def visualize():
    os.makedirs(OUT_DIR, exist_ok=True)
    moves, attn_maps, post_acts = run_self_play(N_MOVES)

    # attention.gif
    frames = []
    board = chess.Board()
    for t, amap in enumerate(attn_maps):
        img = board_to_rgb(board).astype(np.float32)/255
        att = cv2.resize(amap, (8,8), interpolation=cv2.INTER_LINEAR)
        att = (att-att.min())/(att.max()-att.min()+1e-9)
        vis = img*(1-att[:,:,None]) + att[:,:,None]*np.array([1,0,0])
        up = cv2.resize((vis*255).astype(np.uint8), (256,256), interpolation=cv2.INTER_NEAREST)
        frames.append(up)
        board.push(chess.Move.from_uci(moves[t]))
    imageio.mimsave(f'{OUT_DIR}/attention.gif', frames, fps=2)
    print("Saved", OUT_DIR+"/attention.gif")

    # neural.gif
    frames = []
    for t in range(len(post_acts)):
        vec = post_acts[t,:K_NEURONS]
        fig, ax = plt.subplots(figsize=(6,1))
        ax.imshow(vec[None,:], aspect='auto', cmap='magma')
        ax.axis('off')
        fig.canvas.draw()
        buf, (w,h) = fig.canvas.print_to_buffer()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h,w,4)[...,1:4]
        frames.append(arr)
        plt.close(fig)
    imageio.mimsave(f'{OUT_DIR}/neural.gif', frames, fps=2)
    print("Saved", OUT_DIR+"/neural.gif")

    # PGN
    game = chess.pgn.Game(); node=game; board=chess.Board()
    for mv in moves:
        node = node.add_main_variation(chess.Move.from_uci(mv))
    game.headers["Result"] = board.result()
    with open(f'{OUT_DIR}/game.pgn','w') as f:
        f.write(str(game))
    print("Saved", OUT_DIR+"/game.pgn")

if __name__=='__main__':
    visualize()