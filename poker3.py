#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Survival‑oriented DQN agent for Limit Texas Hold'em (RLCard + PyTorch)
────────────────────────────────────────────────────────────────────────
* **Episode = 1 match**: hands are played back‑to‑back until one player is bankrupt or `MAX_HANDS` is reached.
* Per‑step reward = bankroll Δ − λ·bet + ϵ_survive  (+1/‑1 at match end)
* Behavior Cloning, population opponent, Double‑DQN ... existing features are maintained.
"""
from __future__ import annotations
import argparse, random, copy
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rlcard
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1
from tqdm import trange, tqdm

# ───────────────────────── Config ──────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hyper‑parameters
REPLAY_CAPACITY       =   400_000
BATCH_SIZE            =       512
GAMMA                 =       0.99
LR                    =       1e-4
TARGET_UPDATE         =       5_000
SAVE_INTERVAL         =      20_000

# Adjusted for shorter training run of ~10,000 episodes
EPS_START             =       1.00
EPS_END               =       0.05
EPS_DECAY_STEPS       =    70_000 
GRAD_CLIP             =       5.0

# λ penalty schedule
LAMBDA_MAX            =       0.1
LAMBDA_FULL_EP        =     0 

# survival reward
SURVIVE_BONUS         =       0.01
MAX_HANDS             =     10000
BC_HANDS              =   100_000
BC_EPOCHS             =        10
BC_BATCH              =     1_024

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curriculum / population adjusted for shorter training
TEACHER_PHASE_EP      =     10
MIXED_PHASE_EP        =     30
SNAPSHOT_INTERVAL_EP  =       500

START_BANKROLL        =      200.0
USE_PENALTY           = True

# Card encoding
SUITS       = "CDHS"
RANKS       = "23456789TJQKA"
SUIT_SYMBOL = {'C':'♦', 'D':'♥', 'H':'♣', 'S':'♠'}
SUIT2IDX    = {s: i for i, s in enumerate(SUITS)}
RANK2IDX    = {r: i for i, r in enumerate(RANKS)}
CARD2IDX    = {f"{r}{s}": (SUIT2IDX[s], RANK2IDX[r]) for s in SUITS for r in RANKS}
STAGE2IDX   = {"preflop":0, "flop":1, "turn":2, "river":3}
STATE_DIM   = 4*13*2 + 5 + 4

# ───────────────────────── Utilities ─────────────────────────

def _cards_map(cards: List[Any]) -> np.ndarray:
    mat = np.zeros((4, 13), dtype=np.float32)
    for c in cards:
        card_str = str(c)
        if card_str in CARD2IDX:
            s_i, r_i = CARD2IDX[card_str]
            mat[s_i, r_i] = 1.0
    return mat.flatten()

def extract_state(player_state: dict) -> torch.Tensor:
    hole  = _cards_map(player_state.get("hand", []))
    board = _cards_map(player_state.get("public_cards", []))
    cards = np.concatenate((hole, board)).astype(np.float32)

    my_stack    = player_state.get("my_chips", 0) / START_BANKROLL
    pot         = player_state.get("pot", 0) / START_BANKROLL
    pot_ratio   = pot / (my_stack + 1e-6)
    # 追加情報
    stakes      = player_state.get("in_chips", 0) / START_BANKROLL
    action_hist = player_state.get("action_record", [])
    num_raises  = sum(
        1 for a in action_hist
        if hasattr(a, "name") and a.name.lower().startswith("raise")
    )

    stage_str = player_state.get("stage", "preflop")
    stage_idx = STAGE2IDX.get(stage_str, 0)
    stage_oh  = np.eye(4, dtype=np.float32)[stage_idx]

    vec = np.concatenate((
        cards,
        [my_stack, pot, pot_ratio, stakes, num_raises],
        stage_oh
    ))
    
    # print("extract_state vec.shape:", vec.shape)  # ←追加
    return torch.from_numpy(vec).float()

def map_action_to_id(obs: dict, action_name_str: str) -> int | None:
    for action_id, raw_action in zip(obs['legal_actions'].keys(), obs['raw_legal_actions']):
        current_action_name = raw_action.name if hasattr(raw_action, 'name') else raw_action
        if current_action_name == action_name_str:
            return action_id
    return None

def map_id_to_name(obs: dict, action_id: int) -> str:
    try:
        idx = list(obs['legal_actions'].keys()).index(action_id)
        raw_action = obs['raw_legal_actions'][idx]
        return raw_action.name if hasattr(raw_action, 'name') else raw_action
    except (ValueError, IndexError):
        return "unknown_action"


# ────────────────────────── Replay Buffer ──────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor | None):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states_list = map(list, zip(*transitions))

        states = torch.stack(states).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        
        non_final_mask = torch.tensor([s is not None for s in next_states_list], dtype=torch.bool, device=DEVICE)
        if non_final_mask.any():
            non_final_next_states = torch.stack([s for s in next_states_list if s is not None]).to(DEVICE)
        else:
            non_final_next_states = torch.empty(0, states.size(1), device=DEVICE)

        return states, actions, rewards, non_final_next_states, non_final_mask

    def __len__(self) -> int:
        return len(self.buffer)

# ────────────────────────── Q‑net & Agent ──────────────────────────
class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.policy_net = QNet(STATE_DIM, action_dim).to(DEVICE)
        self.target_net = QNet(STATE_DIM, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(REPLAY_CAPACITY)
        self.steps_done = 0
        self.updates_done = 0

    def get_epsilon(self) -> float:
        return max(EPS_END, EPS_START - (EPS_START - EPS_END) * self.steps_done / EPS_DECAY_STEPS)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, legal_actions: List[int], is_greedy: bool = False) -> int:
        self.steps_done += 1
        if not is_greedy and random.random() < self.get_epsilon():
            return random.choice(legal_actions)
        
        q_values = self.policy_net(state.unsqueeze(0).to(DEVICE)).squeeze(0)
        mask = torch.full((self.action_dim,), -float('inf'), device=DEVICE)
        mask[legal_actions] = 0
        q_values_legal = q_values + mask
        return int(torch.argmax(q_values_legal))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE: return
        states, actions, rewards, non_final_next_states, non_final_mask = self.memory.sample(BATCH_SIZE)
        
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_actions = self.policy_net(non_final_next_states).argmax(1, keepdim=True)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

        expected_state_action_values = rewards + (GAMMA * next_state_values)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.updates_done += 1
        if self.updates_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path: Path, episode: int):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
            'episode': episode,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.updates_done = checkpoint['updates_done']
        return checkpoint.get('episode', 0)

# ───────── Population ─────────
class Population:
    def __init__(self):
        self.models: List[QNet] = []

    def add(self, agent: DQNAgent):
        model = copy.deepcopy(agent.policy_net).cpu().eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)

    def get_random_agent(self, action_dim: int) -> Any:
        if not self.models:
            return LimitholdemRuleAgentV1()
        
        model = random.choice(self.models)
        
        class _PopulationAgent:
            def __init__(self, model: QNet, action_dim: int):
                self.model = model.to(DEVICE)
                self.action_dim = action_dim

            @torch.no_grad()
            def step(self, obs: dict) -> int:
                legal_actions = list(obs['legal_actions'].keys())
                state_vec = extract_state(obs['raw_obs']).to(DEVICE)
                q_values = self.model(state_vec.unsqueeze(0)).squeeze(0)
                mask = torch.full((self.action_dim,), -float('inf'), device=DEVICE)
                mask[legal_actions] = 0
                return int(torch.argmax(q_values + mask))
        
        return _PopulationAgent(model, action_dim)

# ───────── BC ─────────
def collect_bc_data(num_hands: int) -> List[Tuple[dict, int]]:
    # 3人環境で全員ルールベース
    env = rlcard.make('limit-holdem', config={'seed': SEED, 'game_num_players': 3})
    teacher1 = LimitholdemRuleAgentV1()
    teacher2 = LimitholdemRuleAgentV1()
    teacher3 = LimitholdemRuleAgentV1()
    env.set_agents([teacher1, teacher2, teacher3])
    data = []
    for _ in trange(num_hands, desc='BC Data Collection'):
        trajectories, _ = env.run(is_training=False)
        for trajectory in trajectories:
            for i in range(0, len(trajectory) - 1, 2):
                obs, action_obj = trajectory[i], trajectory[i+1]
                if obs is None or action_obj is None or 'raw_obs' not in obs or obs['raw_obs'] is None:
                    continue
                try:
                    idx = obs['raw_legal_actions'].index(action_obj)
                    action_id = list(obs['legal_actions'].keys())[idx]
                    data.append((obs['raw_obs'], action_id))
                except (ValueError, IndexError):
                    continue
    return data

def train_bc(net: QNet, data: List[Tuple[dict, int]]):
    class BCDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            raw_obs, action_id = self.data[idx]
            return extract_state(raw_obs), action_id

    dataset = BCDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BC_BATCH, shuffle=True)
    criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in trange(BC_EPOCHS, desc="BC Training"):
        for states, actions in loader:
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            optimizer.zero_grad(); outputs = net(states)
            loss = criterion(outputs, actions); loss.backward(); optimizer.step()
    net.eval()

# ───────── Training loop (match‑based) ─────────

def train(total_episodes: int, resume_path: str | None = None, use_bc: bool = False):
    # 3人環境でDQN+2ルールベース
    env = rlcard.make('limit-holdem', config={'seed': SEED, 'game_num_players': 3})
    action_dim = env.num_actions
    agent = DQNAgent(action_dim)
    teacher1 = LimitholdemRuleAgentV1()
    teacher2 = LimitholdemRuleAgentV1()
    population = Population()

    start_episode = 0
    if resume_path and Path(resume_path).is_file():
        start_episode = agent.load_checkpoint(Path(resume_path)) + 1
        print(f"Resumed training from episode {start_episode}")

    if use_bc and start_episode == 0:
        bc_data = collect_bc_data(BC_HANDS)
        print(f"Collected {len(bc_data)} samples for BC.")
        train_bc(agent.policy_net, bc_data)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("Behavioral Cloning finished. Starting Reinforcement Learning.")

    total_steps = 0
    STAGE_SURVIVE_BONUS = {
        "preflop": 0.005,
        "flop":    0.01,
        "turn":    0.02,
        "river":   0.03,
    }
    
    λ = LAMBDA_MAX
    for ep in trange(start_episode, total_episodes, desc='Train RL'):
        if ep > TEACHER_PHASE_EP and ep % SNAPSHOT_INTERVAL_EP == 0:
            population.add(agent)
            print(f"\nEpisode {ep}: Added current agent to population. Size: {len(population.models)}")

        # 2人の相手を決定
        if ep < TEACHER_PHASE_EP:
            opponent1 = teacher1
            opponent2 = teacher2
        elif ep < MIXED_PHASE_EP:
            opponent1 = teacher1 if random.random() < 0.5 else population.get_random_agent(action_dim)
            opponent2 = teacher2 if random.random() < 0.5 else population.get_random_agent(action_dim)
        else:
            opponent1 = population.get_random_agent(action_dim)
            opponent2 = population.get_random_agent(action_dim)

        env.set_agents([agent, opponent1, opponent2])

        bankrolls = [START_BANKROLL] * 3
        num_hands = 0
        match_over = False
        
        while not match_over:
            obs, current_player_id = env.reset()
            for i in range(env.num_players):
                env.game.players[i].stack = bankrolls[i]
            obs = env.get_state(current_player_id)

            hand_over = False
            last_agent_experience = None

            while not hand_over:
                if current_player_id == 0:
                    # DQNエージェント
                    raw_obs = obs['raw_obs']
                    stage_str = raw_obs.get("stage", "preflop")
                    player = env.game.players[0]
                    stack_before = player.stack
                    state_vec = extract_state(raw_obs)
                    legal_actions = list(obs['legal_actions'].keys())
                    action = agent.select_action(state_vec, legal_actions)
                    action_name = map_id_to_name(obs, action)
                    next_obs, next_player_id = env.step(action)
                    stack_after = env.game.players[0].stack
                    bet_amount = max(stack_before - stack_after, 0.0)
                    norm_bet = bet_amount / START_BANKROLL
                    if action_name.lower().startswith("fold"):
                        survive_bonus = 0.0
                    else:
                        survive_bonus = STAGE_SURVIVE_BONUS.get(stage_str, STAGE_SURVIVE_BONUS["preflop"])
                    reward = survive_bonus - λ * norm_bet
                    reward = max(min(reward, 1.0), -1.0)
                    next_state_vec = None if env.is_over() else extract_state(next_obs['raw_obs'])
                    agent.memory.push(state_vec, action, reward, next_state_vec)
                    agent.optimize_model()
                    total_steps += 1
                    obs = next_obs
                    current_player_id = next_player_id
                    last_agent_experience = (state_vec, action, reward, next_state_vec)
                else:
                    # ルールベース or population agent
                    if current_player_id == 1:
                        opponent = opponent1
                    else:
                        opponent = opponent2
                    action_obj = opponent.step(obs)
                    if not isinstance(action_obj, int):
                        action_name_str = action_obj.name if hasattr(action_obj, 'name') else action_obj
                        action_id = map_action_to_id(obs, action_name_str)
                        if action_id is None:
                            action = random.choice(list(obs['legal_actions'].keys()))
                        else:
                            action = action_id
                    else:
                        action = action_obj
                    obs, current_player_id = env.step(action)
                hand_over = env.is_over()

            final_stacks = [p.stack for p in env.game.players]
            net_payoffs_chips = [final_stacks[i] - bankrolls[i] for i in range(env.num_players)]
            if agent.memory.buffer and last_agent_experience is not None:
                s, a, r, ns = agent.memory.buffer.pop()
                norm_payoff = net_payoffs_chips[0] / START_BANKROLL
                final_reward = r + norm_payoff
                agent.memory.push(s, a, final_reward, ns)
            for i in range(env.num_players):
                bankrolls[i] += net_payoffs_chips[i]
            num_hands += 1
            if any(b <= 0 for b in bankrolls) or num_hands >= MAX_HANDS:
                match_over = True
                # 3人勝敗ボーナス
                winner = np.argmax(bankrolls)
                win_reward = 1.0 if winner == 0 else -1.0
                if agent.memory.buffer:
                    s, a, r, ns = agent.memory.buffer.pop()
                    final_reward = r + win_reward
                    agent.memory.push(s, a, final_reward, ns)

    agent.save_checkpoint(CHECKPOINT_DIR / 'dqn_final.pth', total_episodes - 1)

# ───────── ASCII Rendering / Eval / Demo / Play ─────────

def ascii_card(c: Any) -> str:
    """Returns a formatted card string with suit symbols, handling Card objects."""
    if hasattr(c, 'suit') and hasattr(c, 'rank'):
        return f"{c.rank}{SUIT_SYMBOL[c.suit]}"
    elif isinstance(c, str) and len(c) == 2 and c[0] in RANKS and c[1] in SUITS:
        return f"{c[0]}{SUIT_SYMBOL[c[1]]}"
    return f"[{str(c)}]"

def ascii_cards(cards: List[Any]) -> str:
    """Returns a formatted string for a list of cards."""
    return "".join(f" [{ascii_card(c):<2}]" for c in cards)

def ascii_render_state(env, obs: dict, current_player_id: int):
    """Renders the current game state in ASCII art, ensuring stacks are read correctly."""
    # player_stacks = [p.stack for p in env.game.players]
    player_stacks = [p.stack - p.in_chips for p in env.game.players]
    player_hands = [p.hand for p in env.game.players]
    
    W = 55
    print("┌" + "─" * W + "┐")
    print(f"│ Community: {ascii_cards(env.game.public_cards):<{W-12}} │")
    print("├" + "─" * W + "┤")
    for pid, (hand, remaining) in enumerate(zip(player_hands, player_stacks)):
        marker = ' <<' if pid == current_player_id else ''
        print(f"│ P{pid} Hand: {ascii_cards(hand):<20} Stack: {remaining:<8.0f}{marker:<{W-43}} │")
    
    pot_amount = sum(p.in_chips for p in env.game.players)
    print(f"├{'─'*W}┤\n│ Pot: {pot_amount:<10.0f}{'':<{W-16}}│\n└{'─'*W}┘")


# --- Evaluation ---
def evaluate(num_matches: int, checkpoint_path: Path):
    """Evaluates the agent's performance over a number of full matches.
    追加でハンドあたりのペイオフ分布、平均・標準偏差・信頼区間などを算出します。
    """
    env = rlcard.make('limit-holdem', config={'seed': SEED + 42, 'game_num_players': 3})
    agent = DQNAgent(env.num_actions); agent.load_checkpoint(checkpoint_path)
    teacher = LimitholdemRuleAgentV1()
    env.set_agents([agent, teacher])
    
    matches_won = 0
    total_hands = 0
    # 各ハンドごとのエージェントペイオフ（Big Blind 単位）を蓄積するリスト
    per_hand_payoffs = []
    # matchごとの最終ペイオフも得たいなら別リストを作る（ここではハンド粒度を重視）
    match_payoffs = []
    
    BIG_BLIND = None  # 最初に environment から取得しておく
    # RLCard では env.game.small_blind が存在する場合が多いので、reset 前後で取得
    # ただし、evaluate の呼び出し前に env.make() だけで small_blind が確定していれば：
    try:
        temp_sb = env.game.small_blind
        BIG_BLIND = temp_sb * 2
    except Exception:
        BIG_BLIND = 1.0  # 環境に依存。必要に応じて修正。
    
    for _ in trange(num_matches, desc="Evaluating Matches"):
        bankrolls = [START_BANKROLL, START_BANKROLL]
        num_hands = 0
        match_over = False
        
        # マッチ開始時に、ハンドごとのエージ蓄積用のローカルリスト
        match_per_hand_payoffs = []
        
        while not match_over:
            # Reset hand
            obs, _ = env.reset()
            for i in range(env.num_players):
                env.game.players[i].stack = bankrolls[i]
            
            # ハンド内ループ
            while not env.is_over():
                current_player_id = env.get_player_id()
                obs = env.get_state(current_player_id)
                if current_player_id == 0:
                    state_vec = extract_state(obs['raw_obs'])
                    legal_actions = list(obs['legal_actions'].keys())
                    action = agent.select_action(state_vec, legal_actions, is_greedy=True)
                else:
                    action_obj = teacher.step(obs)
                    action_name_str = action_obj.name if hasattr(action_obj, 'name') else action_obj
                    action = map_action_to_id(obs, action_name_str)
                    if action is None:
                        action = random.choice(list(obs['legal_actions'].keys()))
                env.step(action)
            
            # ハンド終了：ペイオフ取得
            # RLCard の get_payoffs() は通常 Big Blind 単位のユニットを返す (例えば +1, -1 など) ことが多い
            payoffs = env.get_payoffs()  # 例: [u_agent, u_opponent]
            # Big Blind 単位ではなくチップ単位で返る場合は、BIG_BLIND で乗算する必要がある
            # ここでは、payoffs が「Big Blind 単位」と仮定して扱う
            # もし payoffs がチップ単位の場合は: payoffs_chips = [float(u) for u in payoffs]; 
            #   per_hand_payoff_bb = payoffs_chips[0] / BIG_BLIND
            per_hand_payoff_bb = float(payoffs[0])
            per_hand_payoffs.append(per_hand_payoff_bb)
            match_per_hand_payoffs.append(per_hand_payoff_bb)
            
            # 銀行ロール更新
            # RLCard の payoffs が BB単位なら、チップで扱う場合は * BIG_BLIND してスタックに加える設計 
            # しかしここでは銀行ロールを Big Blind 単位として追跡してもよい。
            bankrolls[0] += per_hand_payoff_bb * BIG_BLIND
            bankrolls[1] += float(payoffs[1]) * BIG_BLIND
            num_hands += 1
            total_hands += 1
            
            # マッチ終了判定
            if any(b <= 0 for b in bankrolls) or num_hands >= MAX_HANDS:
                match_over = True
                # 勝率カウント
                if bankrolls[0] > bankrolls[1]:
                    matches_won += 1
                # match-level payoffs（オプションで保存）
                net_match_payoff_bb = sum(match_per_hand_payoffs)  # Big Blind 単位での合計
                match_payoffs.append(net_match_payoff_bb)
        
        # 次のマッチへ
    
    # 結果集計
    win_rate = matches_won / num_matches

    # ハンドあたり平均 EV (BB 単位)
    per_hand_array = np.array(per_hand_payoffs, dtype=np.float64)  # shape=(total_hands,)
    mean_ev_bb = per_hand_array.mean()  # 1ハンドあたり平均 EV
    std_ev_bb = per_hand_array.std(ddof=1) if total_hands > 1 else 0.0  # 1ハンドあたり標本標準偏差

    # 1ハンドあたり平均 EV の 95% 信頼区間（正規近似）：mean ± 1.96 * (std / sqrt(N))
    ci_half_width = 1.96 * (std_ev_bb / np.sqrt(total_hands)) if total_hands > 0 else 0.0

    # --- 100ハンドあたりの期待値を「平均EVを100倍で表現」 ---
    mean_ev_per_100_bb = mean_ev_bb * 100
    # 95% CI も 100 倍して表現
    ci_half_width_per_100 = ci_half_width * 100

    # --- 100ハンドまとめたときの「合計リターン」分布の標準偏差 ---
    std_per_100 = std_ev_bb * np.sqrt(100)  # 1ハンドあたり std_ev_bb を sqrt(100) 倍

    # マッチごとの合計 payoffs 分布
    match_array = np.array(match_payoffs, dtype=np.float64)
    mean_match_bb = match_array.mean() if len(match_array) > 0 else 0.0
    std_match_bb = match_array.std(ddof=1) if len(match_array) > 1 else 0.0
    # マッチあたり平均 EV の 95% CI：1.96 * std_match_bb / sqrt(マッチ数)
    ci_match_half = 1.96 * (std_match_bb / np.sqrt(len(match_array))) if len(match_array) > 0 else 0.0

    # 標準出力またはログ出力
    print(f"=== Evaluation over {num_matches} matches ({total_hands} hands) ===")
    print(f"Match win rate vs rule-based agent: {win_rate:.2%} ({matches_won}/{num_matches})")
    # 1ハンドあたり
    print(f"Hand-level EV (BB単位): mean {mean_ev_bb:.4f} ± {ci_half_width:.4f} (95% CI), std {std_ev_bb:.4f}")
    # 100ハンドあたり平均EV表現と、そのCI
    print(f"Hand-level EV per 100 hands: mean {mean_ev_per_100_bb:.2f} BB ± {ci_half_width_per_100:.2f} BB (95% CI)")
    # 100ハンドまとめた合計リターンのばらつきイメージ
    print(f"  (100ハンド合計の標準偏差イメージ: {std_per_100:.2f} BB)")
    # マッチレベル
    print(f"Match-level total EV (BB単位): mean {mean_match_bb:.2f} ± {ci_match_half:.2f} (95% CI), std {std_match_bb:.2f}")

# --- Demo & Play (Match-based) ---
def run_match(env, agents: List, mode: str):
    bankrolls = [START_BANKROLL for _ in range(env.num_players)]
    num_hands = 0
    print(f"=== Num Players: {env.num_players} ===")
    print(f"Agents: {env.agents}")


    while True:
        if any(b <= 0 for b in bankrolls):
            break
        obs, current_player_id = env.reset()
        for i in range(env.num_players):
            env.game.players[i].stack = bankrolls[i]
        obs = env.get_state(current_player_id)
        ascii_render_state(env, obs, current_player_id)

        while not env.is_over():
            current_agent = agents[current_player_id]

            if isinstance(current_agent, UserAgent):
                action_id = current_agent.step(obs)
                action_name = map_id_to_name(obs, action_id)
                print(f"\n>> User (P{current_player_id}) plays: {action_name} (id={action_id})\n")

            elif isinstance(current_agent, DQNAgent):
                state_vec = extract_state(obs['raw_obs'])
                legal_actions = list(obs['legal_actions'].keys())
                action_id = current_agent.select_action(state_vec, legal_actions, is_greedy=True)
                action_name = map_id_to_name(obs, action_id)
                print(f"\n>> DQNAgent (P{current_player_id}) plays: {action_name} (id={action_id})\n")

            else:  # Rule-based or other type of agent
                if mode == 'play':
                    action_id = current_agent.step(obs)
                else:  # demo mode
                    action_obj = current_agent.step(obs)
                    action_name_str = action_obj.name if hasattr(action_obj, 'name') else action_obj
                    action_id = map_action_to_id(obs, action_name_str)
                    if action_id is None:
                        action_id = random.choice(list(obs['legal_actions'].keys()))
                action_name = map_id_to_name(obs, action_id)
                print(f"\n>> Agent (P{current_player_id}) plays: {action_name} (id={action_id})\n")

            obs, current_player_id = env.step(action_id)
            ascii_render_state(env, obs, current_player_id)

        # 勝敗計算とbankroll更新
        units = env.get_payoffs()
        BIG_BLIND = env.game.small_blind * 2
        payoffs_chips = [float(u) * BIG_BLIND for u in units]
        print(f"\n** Hand Over ** Payoffs (chips): {payoffs_chips}\n")

        for i in range(env.num_players):
            bankrolls[i] += payoffs_chips[i]
        num_hands += 1

        if any(b <= 0 for b in bankrolls) or num_hands >= MAX_HANDS:
            print("=== Match Over ===")
            for i, b in enumerate(bankrolls):
                print(f"Final Bankroll (P{i}): {b:.0f}")
            winners = [i for i, b in enumerate(bankrolls) if b == max(bankrolls)]
            if len(winners) == 1:
                print(f"Player {winners[0]} Won the Match!")
            else:
                print("The Match is a Draw!")
            break

        input("Press Enter to start next hand...")



class UserAgent:
    def step(self, state: dict) -> int:
        print("\nYour turn. Available actions:")
        # [FIX] Correctly iterate over legal_actions (list) and get names
        for action_id in state['legal_actions']:
            action_name = map_id_to_name(state, action_id)
            print(f" {action_id:2d}: {action_name}")
        while True:
            choice = input("Select action ID > ")
            if choice.isdigit() and int(choice) in state['legal_actions']:
                return int(choice)
            print("! Invalid choice. Please select a valid action ID.")

def main():
    p = argparse.ArgumentParser(description="DQN Agent for Limit Texas Hold'em")
    p.add_argument('--episodes', type=int, default=10_000, help="Total matches to train for.")
    p.add_argument('--resume', type=str, help="Path to a checkpoint to resume from.")
    p.add_argument('--bc', action='store_true', help="Perform behavioral cloning before RL.")
    p.add_argument('--eval', action='store_true', help="Evaluate a trained agent.")
    p.add_argument('--eval-matches', type=int, default=100, help="Number of matches to evaluate for.")
    p.add_argument('--demo', action='store_true', help="Run a demo of a trained agent.")
    p.add_argument('--play', action='store_true', help="Play against a trained agent.")
    args = p.parse_args()

    ck_path = Path(args.resume) if args.resume else find_latest_checkpoint()
    if any([args.eval, args.demo, args.play]) and (not ck_path or not ck_path.is_file()):
        print(f"!! Error: Checkpoint not found at '{ck_path}'. Please train first or provide a valid path via --resume.")
        return

    if args.demo:
        print(f"Running demo with {ck_path}")
        env = rlcard.make('limit-holdem', config={'seed': SEED, 'game_num_players': 3})
        agent = DQNAgent(env.num_actions); agent.load_checkpoint(ck_path)
        opponent1 = LimitholdemRuleAgentV1()
        opponent2 = LimitholdemRuleAgentV1()
        env.set_agents([agent, opponent1, opponent2])
        run_match(env, [agent, opponent1, opponent2], 'demo')
    elif args.eval:
        print(f"Evaluating {ck_path}")
        evaluate(args.eval_matches, ck_path)
    elif args.play:
        print(f"Playing against {ck_path}")
        env = rlcard.make('limit-holdem', config={'seed': SEED, 'game_num_players': 3})

        user = UserAgent()
        opponent1 = DQNAgent(env.num_actions); opponent1.load_checkpoint(ck_path)
        opponent2 = DQNAgent(env.num_actions); opponent2.load_checkpoint(ck_path)
        env.set_agents([user, opponent1, opponent2])
        run_match(env, [user, opponent1, opponent2], 'play')
    else:
        train(args.episodes, args.resume, use_bc=args.bc)

def find_latest_checkpoint() -> Path | None:
    files = sorted(CHECKPOINT_DIR.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

if __name__ == '__main__':
    main()
