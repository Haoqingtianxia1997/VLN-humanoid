import torch, torch.nn as nn
from typing import Tuple, NamedTuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


# ─────────────────── RecurrentState 容器 ───────────────────
class RecurrentStates(NamedTuple):
    pi: Tuple[torch.Tensor, torch.Tensor]   # actor (h,c)
    vf: Tuple[torch.Tensor, torch.Tensor]   # critic(h,c)


# ───────────────  无处理 extractor (直接 passthrough) ─────────
class PassThroughExtractor(BaseFeaturesExtractor):
    def __init__(self, space):          # features_dim = obs_dim
        super().__init__(space, features_dim=space.shape[0])
        self.identity = nn.Identity()
    def forward(self, x):               # x shape = (batch, obs_dim)
        return self.identity(x)


# ───────────────────   主策略网络   ──────────────────────────
class MotionLSTMPolicy(RecurrentActorCriticPolicy):
    """与 motion.pt 完全一致:  obs→LSTM64→MLP32→动作&值"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            net_arch=None,
            features_extractor_class = PassThroughExtractor,
            features_extractor_kwargs= {},
            **kwargs
        )

        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]

        # LSTM（actor / critic 各一套）
        self.lstm_actor  = nn.LSTM(obs_dim, 64, batch_first=True)
        self.lstm_critic = nn.LSTM(obs_dim, 64, batch_first=True)

        # MLP 头
        self.actor_net = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, act_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1)
        )

        # 替换 sb3 默认 mlp_extractor，防止维度冲突
        class _Bypass(nn.Module):
            def forward_actor (self, x): return x
            def forward_critic(self, x): return x
        self.mlp_extractor = _Bypass()

        # 连续动作高斯分布的 log_std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self._build(lr_schedule=lambda _: 0.0)   # 注册参数


    # ---------- helper ----------
    def _clip_state(self, state, batch):
        """保证 hidden state 的 batch 维与 obs 一致"""
        h, c = state
        if h.shape[1] != batch:
            h = h[:, :batch, :]
            c = c[:, :batch, :]
        return h, c

    # ---------- forward (rollout 用) ----------
    def forward(self, obs, lstm_states, episode_starts):
        (h_a, c_a), (h_v, c_v) = lstm_states          # unpack

        B = obs.shape[0]        # batch*time
        N = h_a.shape[1]        # n_env
        T = B // N              # time steps

        seq = obs.view(N, T, -1).transpose(0,1)       # [T,B,obs]

        a_out, state_a2 = self.lstm_actor (seq, (h_a, c_a))
        v_out, state_v2 = self.lstm_critic(seq, (h_v, c_v))

        a_out = a_out.transpose(0,1).reshape(B, -1)
        v_out = v_out.transpose(0,1).reshape(B, -1)

        mean     = self.actor_net(a_out)
        log_std  = self.log_std.expand_as(mean)
        dist     = self.action_dist.proba_distribution(mean, log_std)

        actions  = dist.get_actions()
        log_prob = dist.log_prob(actions)
        values   = self.value_net(v_out)

        return actions, values, log_prob, RecurrentStates(pi=state_a2, vf=state_v2)


    # ---------- predict_values (train 阶段 critic) ----------
    def predict_values(self, obs, lstm_states, episode_starts):
        latent_vf, _ = self._process_sequence(obs, lstm_states,
                                              episode_starts, self.lstm_critic)
        return self.value_net(latent_vf)


    # ---------- evaluate_actions (train 阶段 actor/critic) ----------
    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        # lstm_states 可能是 RecurrentStates / tuple / Tensor
        if hasattr(lstm_states, "pi"):                   # RecurrentStates
            actor_state  = lstm_states.pi
            critic_state = lstm_states.vf
        elif isinstance(lstm_states, (tuple, list)) and len(lstm_states) == 2:
            actor_state, critic_state = lstm_states      # ((h,c),(h,c))
        elif isinstance(lstm_states, torch.Tensor):      # 单块 Tensor
            h = lstm_states if lstm_states.dim() == 3 else lstm_states.unsqueeze(0)
            c = torch.zeros_like(h)
            actor_state  = (h, c)
            critic_state = (h.clone(), c.clone())
        else:
            raise ValueError(f"Unsupported lstm_states type: {type(lstm_states)}")

        h_a, c_a = self._clip_state(actor_state , obs.shape[0])
        h_v, c_v = self._clip_state(critic_state, obs.shape[0])

        latent_pi, _ = self._process_sequence(obs, (h_a, c_a),
                                            episode_starts, self.lstm_actor)
        latent_vf, _ = self._process_sequence(obs, (h_v, c_v),
                                            episode_starts, self.lstm_critic)

        mean     = self.actor_net(latent_pi)
        log_std  = self.log_std.expand_as(mean)
        dist     = self.action_dist.proba_distribution(mean, log_std)

        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self.value_net(latent_vf)
        return values, log_prob, entropy



    # ---------- 让 SB3 知道如何初始化 LSTM state ----------
    def get_init_lstm_state(self, batch_size: int = 1):
        device = self.device
        h0 = torch.zeros((1, batch_size, 64), device=device)
        c0 = torch.zeros((1, batch_size, 64), device=device)
        return RecurrentStates(pi=(h0.clone(), c0.clone()),
                               vf=(h0.clone(), c0.clone()))
