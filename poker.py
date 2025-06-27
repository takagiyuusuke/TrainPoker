#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced DQN agent for Limit Texas Hold'em (RLCard + PyTorch)
──────────────────────────────────────────────────────────────────
* Behavior Cloning (BC) bootstrap from rule‑based agent
* Curriculum learning with a **population** of past agents
* Risk‑sensitive reward: bankroll delta − λ·(bet/stack)
* Double‑DQN, LayerNorm MLP, gradient clipping
* Long, slow ε‑schedule (2 M steps)
* Full CLI: train / eval / demo / human play
* All original visualisation utilities preserved
"""
from __future__ import annotations
import argparse, random, math, copy
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Any

import numpy as np
import torch; import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rlcard
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1
from tqdm import trange

# ───────────────────────── Config ──────────────────────────
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# Hyper‑parameters
REPLAY_CAPACITY       =   200_000
BATCH_SIZE            =       512
GAMMA                 =       0.99          # longer horizon
LR                    =       1e-4
TARGET_UPDATE         =       5_000         # gradient steps
SAVE_INTERVAL         =      20_000         # environment steps

# 2‑M step ε‑schedule (1.0 → 0.05)
EPS_START             =       1.00
EPS_END               =       0.05
EPS_DECAY_STEPS       =   2_000_000

GRAD_CLIP             =       5.0

# risk penalty λ schedule (0 → 0.05 over first 100k episodes)
LAMBDA_MAX            =       0.01
LAMBDA_FULL_EP        =   10_000

# Behavior Cloning
BC_HANDS              =   200_000
BC_EPOCHS             =        10
BC_BATCH              =     1_024

CHECKPOINT_DIR = Path("checkpoints"); CHECKPOINT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curriculum settings
TEACHER_PHASE_EP      =    30_000   # only rule‑based opponent
MIXED_PHASE_EP        =    60_000   # rule + weak DQN snapshots
SNAPSHOT_INTERVAL_EP  =    10_000   # add snapshot to pool

START_STACK           =      100.0
USE_PENALTY           = True        # bankroll delta − λ·bet

# Card encoding
SUITS       = "CDHS"
RANKS       = "23456789TJQKA"
SUIT_SYMBOL = {'C':'♦','D':'♥','H':'♣','S':'♠'}
SUIT2IDX    = {s:i for i,s in enumerate(SUITS)}
RANK2IDX    = {r:i for i,r in enumerate(RANKS)}
CARD2IDX    = {f"{r}{s}": (SUIT2IDX[s], RANK2IDX[r]) for s in SUITS for r in RANKS}
STAGE2IDX   = {"PREFLOP":0,"FLOP":1,"TURN":2,"RIVER":3}
STATE_DIM   = 4*13*2 + 3 + 4   # 2×(4×13) cards + 3 scalars + 4 stage‑one‑hot

# ───────────────────────── Utilities ─────────────────────────

def _cards_map(cards: List[str]) -> np.ndarray:
    mat = np.zeros((4,13), dtype=np.float32)
    for c in cards:
        if c in CARD2IDX:
            s_i, r_i = CARD2IDX[c]
            mat[s_i, r_i] = 1.0
    return mat.reshape(-1)

def extract_state(raw_state: dict) -> torch.Tensor:
    hole  = raw_state.get("hand", [])
    board = raw_state.get("public_cards", [])
    cards_enc = np.concatenate([_cards_map(hole), _cards_map(board)], dtype=np.float32)
    my_stack  = raw_state.get("my_chips",0) / START_STACK
    pot       = raw_state.get("pot",0) / START_STACK
    pot_ratio = pot / (my_stack + 1e-8)
    stage_idx = STAGE2IDX.get(raw_state.get("stage","PREFLOP").upper(), 0)
    stage_oh  = np.eye(4, dtype=np.float32)[stage_idx]
    vec       = np.concatenate([cards_enc, np.array([my_stack, pot, pot_ratio], dtype=np.float32), stage_oh])
    return torch.from_numpy(vec)

# map teacher action name/id → env action id

def map_rule_action(raw_state: dict, teacher_act: Any) -> int:
    leg_ids   = list(raw_state.get('legal_actions', {}).keys())
    raw_names = raw_state.get('raw_legal_actions', [])
    # integer id
    if isinstance(teacher_act, int) and teacher_act in leg_ids:
        return teacher_act
    act_str = teacher_act.name.lower() if hasattr(teacher_act,'name') else str(teacher_act).lower()
    for idx,name in zip(leg_ids,raw_names):
        name_str = name.name.lower() if hasattr(name,'name') else str(name).lower()
        if name_str == act_str:
            return idx
    return random.choice(leg_ids)

# ────────────────────────── Replay Buffer ──────────────────────────
class ReplayBuffer:
    def __init__(self, cap:int): self.buf:Deque = deque(maxlen=cap)
    def push(self,s,a:int,r:float,sn): self.buf.append((s,a,r,sn))
    def sample(self,batch:int):
        data = random.sample(self.buf, batch)
        s,a,r,sn = map(list, zip(*data))
        s  = torch.stack(s).to(DEVICE)
        a  = torch.tensor(a, dtype=torch.int64, device=DEVICE)
        r  = torch.tensor(r, dtype=torch.float32, device=DEVICE)
        # すべての sn の要素を DEVICE 上に揃える
        sn = torch.stack([
            t.to(DEVICE) if t is not None else torch.zeros_like(s[0], device=DEVICE)
            for t in sn
        ])
        done = torch.tensor([t is None for t in sn], dtype=torch.bool, device=DEVICE)
        return s,a,r,sn,done
    def __len__(self): return len(self.buf)

# ────────────────────────── Q Network & Agent ──────────────────────────
class QNet(nn.Module):
    def __init__(self, sd:int, ad:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd,512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512,512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512,ad)
        )
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self, action_dim:int):
        self.action_dim = action_dim
        self.policy  = QNet(STATE_DIM, action_dim).to(DEVICE)
        self.target  = QNet(STATE_DIM, action_dim).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict()); self.target.eval()
        self.opt     = optim.Adam(self.policy.parameters(), lr=LR)
        self.buf     = ReplayBuffer(REPLAY_CAPACITY)
        self.steps_done = 0; self.updates = 0
    # ε following 2‑M step linear decay
    def epsilon(self):
        return max(EPS_END, EPS_START - (EPS_START-EPS_END)*self.steps_done/EPS_DECAY_STEPS)
    @torch.no_grad()
    def select_action(self, state:torch.Tensor, legal:List[int], greedy=False):
        if (not greedy):
            self.steps_done += 1
            if random.random() < self.epsilon():
                return random.choice(legal)
        q = self.policy(state.unsqueeze(0).to(DEVICE)).squeeze(0)
        q_legal = q[torch.tensor(legal, device=DEVICE)]
        return legal[int(torch.argmax(q_legal).item())]
    def optimise(self):
        if len(self.buf) < BATCH_SIZE: return
        s,a,r,sn,done = self.buf.sample(BATCH_SIZE)
        # Double‑DQN: argmax w/ policy, value w/ target
        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_act = self.policy(sn).argmax(1, keepdim=True)
            max_next = self.target(sn).gather(1, next_act).squeeze(1)
            q_tgt = r + (~done).float()*GAMMA*max_next
        loss = F.smooth_l1_loss(q_pred, q_tgt)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP); self.opt.step()
        self.updates += 1
        if self.updates % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())
    # persistence helpers
    def save(self, path:Path, ep:int):
        torch.save({
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'opt':    self.opt.state_dict(),
            'steps':  self.steps_done,
            'updates': self.updates,
            'ep':     ep
        }, path)
    def load(self, path:Path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(ckpt['policy'])
        self.target.load_state_dict(ckpt['target'])
        self.opt.load_state_dict(ckpt['opt'])
        self.steps_done = ckpt['steps']; self.updates = ckpt['updates']
        return ckpt.get('ep', 0)

# ─────────────────── Opponent Population ───────────────────
class Population:
    """Pool of frozen policy networks for diverse opponents."""
    def __init__(self):
        self.models:List[QNet] = []
    def add(self, agent:DQNAgent):
        model = copy.deepcopy(agent.policy).to('cpu').eval()
        for p in model.parameters(): p.requires_grad=False
        self.models.append(model)
    def random_agent(self, action_dim:int):
        if not self.models:
            return LimitholdemRuleAgentV1()
        model = random.choice(self.models)
        class _StaticAgent:
            def __init__(self, m, ad): self.m,self.ad=m,ad
            @torch.no_grad()
            def step(self, raw_state):
                legal=list(raw_state['legal_actions'].keys())
                s=extract_state(raw_state).unsqueeze(0)
                q=self.m(s).squeeze(0)
                q_legal=q[torch.tensor(legal)]
                return legal[int(torch.argmax(q_legal))]
        return _StaticAgent(model, action_dim)

# ─────────────────── Behavior Cloning Data & Training ───────────────────

def collect_bc_data(hands:int) -> List[Tuple[dict,int]]:
    env     = rlcard.make("limit-holdem", config={'seed': SEED})
    teacher = LimitholdemRuleAgentV1(); env.set_agents([teacher,teacher])
    data=[]
    for _ in trange(hands, desc="BC collect"):
        traj,_ = env.run(is_training=False)
        for player in traj:
            for i in range(0, len(player)-1, 2):
                s_dict, a_raw = player[i], player[i+1]
                legal_ids = list(s_dict['legal_actions'].keys())
                raw_names = s_dict['raw_legal_actions']
                try:
                    a_id = raw_names.index(a_raw)
                    data.append((s_dict['raw_obs'], a_id))
                except ValueError:
                    continue
    return data

def train_bc(net:QNet, data:list):
    loader = torch.utils.data.DataLoader(data, batch_size=BC_BATCH, shuffle=True, collate_fn=lambda b:b)
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(net.parameters(), lr=LR)
    for _ in range(BC_EPOCHS):
        for batch in loader:
            states, acts = zip(*batch)
            s = torch.stack([extract_state({**{'hand':None,'public_cards':[]}, **st}) for st in states]).to(DEVICE)
            a = torch.tensor(acts, device=DEVICE)
            loss = crit(net(s), a)
            opt.zero_grad(); loss.backward(); opt.step()

# ───────────────────────────── Training ────────────────────────────

def train(total_ep:int, resume:str|None=None, use_bc:bool=False):
    env    = rlcard.make("limit-holdem", config={'seed': SEED})
    action_dim = env.num_actions
    agent  = DQNAgent(action_dim)
    teacher= LimitholdemRuleAgentV1()
    pop    = Population()

    # resume
    start_ep = 0
    if resume and Path(resume).is_file():
        start_ep = agent.load(Path(resume)) + 1
        print(f"Resumed from ep={start_ep}")

    # behavior cloning
    if use_bc and start_ep == 0:
        print("Collecting BC data…")
        bc_data = collect_bc_data(BC_HANDS)
        print(f"Training BC with {len(bc_data)} samples")
        train_bc(agent.policy, bc_data)
        agent.target.load_state_dict(agent.policy.state_dict())
        print("BC bootstrap done → RL training")
        agent.save(CHECKPOINT_DIR / "bc_bootstrap_step0.pth", 0)  # Save checkpoint after BC

    step_cnt = 0
    for ep in trange(start_ep, total_ep, desc="Train RL"):
        raw, cur = env.reset()
        state    = extract_state(raw)
        prev_st  = [p.in_chips for p in env.game.players]

        # determine opponent for this episode
        if ep < TEACHER_PHASE_EP:
            opponent = teacher
        elif ep < MIXED_PHASE_EP:
            # 50% rule, 50% snapshot
            opponent = teacher if random.random() < 0.5 else pop.random_agent(action_dim)
        else:
            opponent = pop.random_agent(action_dim)
        env.set_agents([agent, opponent])

        # λ penalty schedule
        lam = LAMBDA_MAX * min(1.0, ep / LAMBDA_FULL_EP) if USE_PENALTY else 0.0

        while not env.is_over():
            legal_ids = list(raw['legal_actions'].keys())
            if cur == 0:   # learning agent
                action = agent.select_action(state, legal_ids)
                action = int(action)  # Ensure action is int
            else:          # opponent (teacher/pop)
                opp_act = opponent.step(raw)
                action  = map_rule_action(raw, opp_act) if cur == 1 and not isinstance(opponent, DQNAgent) else opp_act
                action = int(action)  # Ensure action is int

            raw_next, next_p = env.step(action)
            new_st = [p.in_chips for p in env.game.players]
            delta  = (new_st[cur] - prev_st[cur]) / START_STACK
            bet_amt= getattr(env.game.round, 'bet', [0])[cur] / START_STACK if hasattr(env.game.round,'bet') else 0.0
            reward = delta - lam * bet_amt if cur == 0 else 0.0

            prev_st = new_st
            done = env.is_over()
            nxt_state = None if done else extract_state(raw_next)
            if cur == 0:  # only learner stores
                agent.buf.push(state, action, reward, nxt_state)
            state = nxt_state if nxt_state is not None else state
            cur   = next_p
            agent.optimise(); step_cnt += 1

            if step_cnt % SAVE_INTERVAL == 0:
                path = CHECKPOINT_DIR / f"ep{ep}_step{step_cnt}.pth"
                agent.save(path, ep)
                pop.add(agent)   # snapshot to population

    # final save
    agent.save(CHECKPOINT_DIR / f"dqn_final_ep{total_ep-1}.pth", total_ep-1)

# ───────── ASCII Rendering / Eval / Demo / Play ─────────
# (unchanged utilities copied from original script)

def ascii_card(c: str) -> str:
    if len(c)==2 and c[0] in RANKS and c[1] in SUITS:
        return f"{c[0]}{SUIT_SYMBOL[c[1]]}"
    return f"[{c}]"

def ascii_cards(cards: List[str]) -> str:
    return "".join(f"[{ascii_card(c)}]" for c in cards)

def ascii_render_state(raw_state: dict, cur_player: int, env):
    states=[env.game.get_state(i) for i in range(len(env.game.players))]
    chips=[st['my_chips'] for st in states]
    comm = states[0]['public_cards']; W=50
    print("┌"+"─"*W+"┐")
    print(f"│ Community: {ascii_cards(comm):<{W-12}} │")
    print("├"+"─"*W+"┤")
    for pid,st in enumerate(states):
        m='←' if pid==cur_player else ' '
        print(f"│ P{pid}{m} Hole: {ascii_cards(st['hand']):<{W-12}} │")
    print("├"+"─"*W+"┤")
    print(f"│ Stack:    {chips[0]:<5}   Pot: {raw_state.get('pot',0):<5} {'':<{W-22}}│")
    print("└"+"─"*W+"┘")

# Evaluation

def evaluate(num_ep:int, ckpt:Path):
    env = rlcard.make('limit-holdem', config={'seed': SEED+42})
    agent = DQNAgent(env.num_actions); agent.load(ckpt)
    teacher = LimitholdemRuleAgentV1(); env.set_agents([agent,teacher])
    total = 0.0
    for _ in trange(num_ep, desc="Evaluating"):
        raw,cur = env.reset()
        while not env.is_over():
            legal = list(raw['legal_actions'].keys())
            action = agent.select_action(extract_state(raw), legal, greedy=True)
            raw,cur = env.step(action)
        total += env.get_payoffs()[0]
    print(f"Average payoff vs rule over {num_ep} hands: {total/num_ep:.3f}")

# Demo

def demo_render(ckpt:Path):
    env = rlcard.make('limit-holdem')
    agent = DQNAgent(env.num_actions); agent.load(ckpt)
    teacher= LimitholdemRuleAgentV1(); env.set_agents([agent,teacher])
    raw,cur = env.reset(); ascii_render_state(raw,cur,env)
    while not env.is_over():
        legal = list(raw['legal_actions'].keys())
        aid = agent.select_action(extract_state(raw), legal, greedy=True)
        act_name = dict(zip(raw['legal_actions'], raw['raw_legal_actions'])).get(aid, str(aid))
        print(f"\n>> Player {cur} action: {act_name} (id={aid})\n")
        raw,cur = env.step(aid); ascii_render_state(raw,cur,env)
    print(f"\n** Hand Over ** Payoffs: {env.get_payoffs()}\n")

# Human play vs trained agent
class UserAgent:
    def step(self, raw_state):
        for a,n in zip(raw_state['legal_actions'], raw_state['raw_legal_actions']):
            print(f" {a:2d}: {n}")
        while True:
            choice = input("Select action ID > ")
            if choice.isdigit() and int(choice) in raw_state['legal_actions']:
                return int(choice)
            print("! invalid")

def play_vs_agent(ckpt:Path):
    env = rlcard.make('limit-holdem', config={'seed': SEED+999})
    agent = DQNAgent(env.num_actions); agent.load(ckpt)
    user  = UserAgent()
    env.set_agents([agent,user])
    raw,cur = env.reset()
    while not env.is_over():
        if cur == 1:
            action = user.step(raw)
        else:
            legal = list(raw['legal_actions'].keys())
            action = agent.select_action(extract_state(raw), legal, greedy=True)
            print(f"\n>> DQN action id: {action}\n")
        raw,cur = env.step(action)
    print(f"\n=== Game Over === Payoffs: {env.get_payoffs()}\n")

# Checkpoint helper & Main

def latest_checkpoint() -> Path|None:
    files = sorted(CHECKPOINT_DIR.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=300_000)
    p.add_argument('--resume',   type=str)
    p.add_argument('--bc',       action='store_true')
    p.add_argument('--eval',     action='store_true')
    p.add_argument('--eval-episodes', type=int, default=10_000)
    p.add_argument('--demo',     action='store_true')
    p.add_argument('--play',     action='store_true')
    args = p.parse_args()

    if args.demo:
        ck = Path(args.resume) if args.resume else latest_checkpoint()
        if ck is None:
            print("No checkpoint found."); return
        demo_render(ck)
    elif args.eval:
        ck = Path(args.resume) if args.resume else latest_checkpoint()
        if ck is None:
            print("No checkpoint found."); return
        evaluate(args.eval_episodes, ck)
    elif args.play:
        ck = Path(args.resume) if args.resume else latest_checkpoint()
        if ck is None:
            print("No checkpoint found."); return
        play_vs_agent(ck)
    else:
        train(args.episodes, args.resume, use_bc=args.bc)

if __name__ == '__main__':
    main()
