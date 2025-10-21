#!/usr/bin/env python
"""
Bi-family MPE Runner (Stage-1) with hybrid defender actions
===========================================================

Defenders share a single policy with a hybrid action head:
- Tuple(Box(2), Discrete(N+1)) => (velocity, assignment)
Intruders share a single policy with a Box(2) velocity action.

Assignments are rebalanced per step to enforce coverage:
- Each active intruder gets up to 2 defenders if capacity allows
- Extras are reassigned to the next most threatening intruders

Env receives only continuous actions; assignments are forwarded via
SubprocVecEnv.set_assignments() (Stage-1 scenario will ignore them, but
we keep the interface consistent for Stage-2 transfer).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from gymnasium import spaces

from runner.shared.base_runner import Runner
import torch
from utils.shared_buffer import SharedReplayBuffer


def _t2n(x: torch.Tensor):
    return x.detach().cpu().numpy()


class BiFamilyRunnerS1(Runner):
    def __init__(self, config):
        super().__init__(config)

        # Family sizes
        self.num_defenders = int(getattr(self.all_args, 'num_defenders', self.num_agents))
        self.num_intruders = int(getattr(self.all_args, 'num_intruders', max(0, self.num_agents - self.num_defenders)))
        assert self.num_defenders + self.num_intruders == self.num_agents

        # Build hybrid defender action space: Tuple(Box(2), Discrete(N+1))
        box_action_space = self.envs.action_space[0]
        assert box_action_space.__class__.__name__ == 'Box', "Env must expose Box actions for velocity"
        self.assign_action_space = spaces.Discrete(self.num_intruders + 1)
        self.hybrid_def_act_space = spaces.Tuple([box_action_space, self.assign_action_space])

        # Policies
        from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        obs_space = self.envs.observation_space[0]
        share_space = self.envs.share_observation_space[0] if self.use_centralized_V else obs_space

        self.def_policy = Policy(self.all_args, obs_space, share_space, self.hybrid_def_act_space, device=self.device)
        self.int_policy = Policy(self.all_args, obs_space, share_space, box_action_space, device=self.device)

        # Trainers
        self.def_trainer = TrainAlgo(self.all_args, self.def_policy, device=self.device)
        self.int_trainer = TrainAlgo(self.all_args, self.int_policy, device=self.device)

        # Buffers
        self.def_buffer = SharedReplayBuffer(self.all_args, self.num_defenders, obs_space, share_space, self.hybrid_def_act_space)
        self.int_buffer = SharedReplayBuffer(self.all_args, self.num_intruders, obs_space, share_space, box_action_space)

        # Stats (Stage-1 style)
        from collections import deque
        self.window_size = 100
        self.defense_success_window = deque(maxlen=self.window_size)
        self.attack_success_window = deque(maxlen=self.window_size)
        self.defender_reward_window = deque(maxlen=self.window_size)
        self.intruder_reward_window = deque(maxlen=self.window_size)
        self.episode_length_window = deque(maxlen=self.window_size)
        self.current_episode_rewards = {
            'defenders': np.zeros(self.n_rollout_threads),
            'intruders': np.zeros(self.n_rollout_threads),
        }
        self.current_episode_steps = np.zeros(self.n_rollout_threads)

        self.total_env_steps = 0
        # Track last-step per-intruder coverage counts for masking
        self.last_counts = np.zeros((self.n_rollout_threads, self.num_intruders), dtype=np.int32)

    def run(self):
        # Warmup
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self._buf_init(share_obs, obs)

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        from tqdm import trange
        pbar = trange(episodes, desc=f"S1-BiFamily-{self.algorithm_name}", leave=True)

        for ep in pbar:
            if self.use_linear_lr_decay:
                self.def_trainer.policy.lr_decay(ep, episodes)
                self.int_trainer.policy.lr_decay(ep, episodes)

            for step in range(self.episode_length):
                d_obs, i_obs = self._split_obs(self.def_buffer.obs[step], self.int_buffer.obs[step])

                # Collect actions
                self.def_trainer.prep_rollout()
                self.int_trainer.prep_rollout()

                dV, dA, dL, dRS, dRSC, dAct_env, dAssign = self._act_def(self.def_policy, self.def_buffer, step)
                iV, iA, iL, iRS, iRSC, iAct_env = self._act_int(self.int_policy, self.int_buffer, step)

                # Build and rebalance assignments
                assignments = self._build_assignments_from_actions(dAssign)
                assignments = self._rebalance_assignments(assignments)
                # Update last coverage counts for next-step masking
                self._update_counts_from_assignments(assignments)
                self._send_assignments(assignments)

                # Env step (only velocities)
                actions_env = self._merge_actions_env(dAct_env, iAct_env)
                obs_next, rewards, dones, infos = self.envs.step(actions_env)

                # Split rewards (no extra dense shaping in Stage-1 runner)
                d_rew, i_rew = self._split_rewards(rewards)

                # Accumulate sliding-window stats
                self._accumulate_rewards_sw(d_rew, i_rew)
                self._handle_episode_end_sw(dones, infos)

                # Insert transitions
                self._insert(self.def_buffer, dV, dA, dL, dRS, dRSC, d_rew, dones, obs_next, step, group='def')
                self._insert(self.int_buffer, iV, iA, iL, iRS, iRSC, i_rew, dones, obs_next, step, group='int')

                obs = obs_next
                self.total_env_steps += self.n_rollout_threads

            # Train
            self._compute_and_train()
            self.def_buffer.after_update()
            self.int_buffer.after_update()

            # Log
            if ep % self.log_interval == 0:
                total_num_steps = (ep + 1) * self.episode_length * self.n_rollout_threads
                self._log_training_info_sw(total_num_steps, ep, episodes, pbar)
            if (ep % self.save_interval == 0 or ep == episodes - 1):
                self._save_all()

    # ------- helpers -------
    def _buf_init(self, share_obs, obs):
        d_share, i_share = self._split_share_obs(share_obs)
        d_obs, i_obs = self._split_obs(obs, obs)
        self.def_buffer.share_obs[0] = d_share.copy()
        self.def_buffer.obs[0] = d_obs.copy()
        self.int_buffer.share_obs[0] = i_share.copy()
        self.int_buffer.obs[0] = i_obs.copy()

    def _split_obs(self, obs_def_like, obs_int_like):
        dN = self.num_defenders
        if isinstance(obs_def_like, np.ndarray) and obs_def_like.ndim == 4:
            obs = obs_def_like  # from buffer
        else:
            obs = obs_def_like
        d_obs = obs[:, :dN]
        i_obs = obs[:, dN:]
        return d_obs, i_obs

    def _split_share_obs(self, share_obs):
        dN = self.num_defenders
        d_share = share_obs[:, :dN]
        i_share = share_obs[:, dN:]
        return d_share, i_share

    def _act_def(self, policy, buffer, step):
        # Build available-actions mask for the discrete head (coverage ≤2 rule)
        avail_mask = self._build_available_mask_def()
        avail_tensor = torch.as_tensor(avail_mask, dtype=torch.float32, device=self.device)
        values, actions, action_log_probs, rnn_states, rnn_states_critic = policy.get_actions(
            np.concatenate(buffer.share_obs[step]),
            np.concatenate(buffer.obs[step]),
            np.concatenate(buffer.rnn_states[step]),
            np.concatenate(buffer.rnn_states_critic[step]),
            np.concatenate(buffer.masks[step]),
            available_actions=avail_tensor,
            deterministic=False,
        )
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # Split hybrid actions: [.., vx, vy, cat]
        dAct_env = actions[..., :2]
        dAssign = actions[..., 2:3]  # keep as shape (...,1)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, dAct_env, dAssign

    def _act_int(self, policy, buffer, step):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = policy.get_actions(
            np.concatenate(buffer.share_obs[step]),
            np.concatenate(buffer.obs[step]),
            np.concatenate(buffer.rnn_states[step]),
            np.concatenate(buffer.rnn_states_critic[step]),
            np.concatenate(buffer.masks[step]),
        )
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions

    def _build_assignments_from_actions(self, a_actions):
        dN = self.num_defenders
        N = self.num_intruders
        mappings = []
        for t in range(self.n_rollout_threads):
            mapping = {}
            for d_idx in range(dN):
                cat = int(a_actions[t, d_idx, 0])
                intr_idx = -1 if cat == 0 else (cat - 1)
                if intr_idx >= N:
                    intr_idx = -1
                mapping[d_idx] = intr_idx
            mappings.append(mapping)
        return mappings

    def _rebalance_assignments(self, assignments):
        """Greedy rebalance: cap at 2 defenders per intruder and fill uncovered intruders if capacity allows."""
        dN = self.num_defenders
        N = self.num_intruders
        out = []
        for t in range(self.n_rollout_threads):
            mapping = assignments[t].copy()
            # Count per intruder
            counts = {i: 0 for i in range(N)}
            for d_idx, i_idx in mapping.items():
                if i_idx >= 0:
                    counts[i_idx] += 1
            # Release extras beyond 2 (keep arbitrary two)
            for i in range(N):
                if counts[i] > 2:
                    need_drop = counts[i] - 2
                    for d_idx, i_idx in list(mapping.items()):
                        if need_drop == 0:
                            break
                        if i_idx == i:
                            mapping[d_idx] = -1
                            need_drop -= 1
                    counts[i] = 2
            # Fill uncovered intruders using idle defenders
            idle = [d for d, i_idx in mapping.items() if i_idx < 0]
            # Simple threat order: by intruder index (Stage-1 no CPF); keep deterministic
            target_list = list(range(N))
            # First pass: ensure >=1
            for i in target_list:
                while counts[i] < 1 and idle:
                    d = idle.pop()
                    mapping[d] = i
                    counts[i] += 1
            # Second pass: try to ensure >=2
            for i in target_list:
                while counts[i] < 2 and idle:
                    d = idle.pop()
                    mapping[d] = i
                    counts[i] += 1
            out.append(mapping)
        return out

    def _update_counts_from_assignments(self, assignments):
        """Update last per-intruder coverage counts from the rebalanced assignments."""
        N = self.num_intruders
        for t in range(self.n_rollout_threads):
            cnt = np.zeros(N, dtype=np.int32)
            for _, i_idx in assignments[t].items():
                if i_idx >= 0:
                    cnt[i_idx] += 1
            self.last_counts[t] = cnt

    def _build_available_mask_def(self):
        """Build a (n_threads*num_defenders, N+1) mask for defender discrete head.
        Idle (index 0) is always allowed. For intruder j (1..N), allow if current
        last_counts[t,j-1] < 2. Captured masking is not applied in Stage-1.
        """
        nT = self.n_rollout_threads
        dN = self.num_defenders
        N = self.num_intruders
        masks = []
        for t in range(nT):
            base = np.ones((dN, N + 1), dtype=np.float32)
            # mark disallowed intruders with 0 where counts >= 2
            for j in range(N):
                if self.last_counts[t, j] >= 2:
                    base[:, j + 1] = 0.0
            # idle always allowed
            base[:, 0] = 1.0
            masks.append(base)
        return np.concatenate(masks, axis=0)

    def _send_assignments(self, assignments):
        if hasattr(self.envs, 'set_assignments'):
            self.envs.set_assignments(assignments)

    def _merge_actions_env(self, d_vel, i_vel):
        return np.concatenate([d_vel, i_vel], axis=1)

    def _split_rewards(self, rewards):
        dN = self.num_defenders
        d_rew = rewards[:, :dN]
        i_rew = rewards[:, dN:]
        return d_rew, i_rew

    def _insert(self, buffer, values, actions, action_log_probs, rnn_states, rnn_states_critic, rewards, dones, obs_next, step, group='def'):
        # Select dones mask for this group
        dN = self.num_defenders
        arr = np.asarray(dones)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if group == 'def':
            dones_group = arr[:, :dN].astype(bool)
        else:
            dones_group = arr[:, dN:].astype(bool)

        rnn_states[dones_group == True] = np.zeros(((dones_group == True).sum(), buffer.recurrent_N, buffer.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_group == True] = np.zeros(((dones_group == True).sum(), *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, buffer.num_agents, 1), dtype=np.float32)
        masks[dones_group == True] = np.zeros(((dones_group == True).sum(), 1), dtype=np.float32)

        # share obs next
        if self.use_centralized_V:
            share_obs = obs_next.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs_next
        d_share, i_share = self._split_share_obs(share_obs)
        d_obs, i_obs = self._split_obs(obs_next, obs_next)

        if group == 'def':
            share_use, obs_use = d_share, d_obs
        else:
            share_use, obs_use = i_share, i_obs

        buffer.insert(share_use, obs_use, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards if group == 'def' else rewards, masks)

    def _accumulate_rewards_sw(self, d_rew, i_rew):
        for t in range(self.n_rollout_threads):
            d_avg = float(np.mean(d_rew[t])) if self.num_defenders > 0 else 0.0
            i_avg = float(np.mean(i_rew[t])) if self.num_intruders > 0 else 0.0
            self.current_episode_rewards['defenders'][t] += d_avg
            self.current_episode_rewards['intruders'][t] += i_avg
            self.current_episode_steps[t] += 1

    def _handle_episode_end_sw(self, dones, infos):
        for t in range(self.n_rollout_threads):
            if np.asarray(dones)[t].all():
                info0 = infos[t][0] if len(infos[t]) > 0 else {}
                defense_success = int(info0.get('defense_success', 0))
                attack_success = int(info0.get('attack_success', 0))
                ep_steps = int(self.current_episode_steps[t])
                self.defense_success_window.append(defense_success)
                self.attack_success_window.append(attack_success)
                self.defender_reward_window.append(self.current_episode_rewards['defenders'][t])
                self.intruder_reward_window.append(self.current_episode_rewards['intruders'][t])
                self.episode_length_window.append(ep_steps)
                self.current_episode_rewards['defenders'][t] = 0.0
                self.current_episode_rewards['intruders'][t] = 0.0
                self.current_episode_steps[t] = 0

    def _log_training_info_sw(self, total_num_steps, ep, episodes, pbar):
        if len(self.defense_success_window) > 0:
            import time
            # No wall-clock start here; we can only report window stats
            train_infos = {
                'defense_success_rate': float(np.mean(self.defense_success_window)),
                'attack_success_rate': float(np.mean(self.attack_success_window)),
                'avg_defender_reward': float(np.mean(self.defender_reward_window)),
                'avg_intruder_reward': float(np.mean(self.intruder_reward_window)),
                'avg_episode_length': float(np.mean(self.episode_length_window)),
                'window_episodes': len(self.defense_success_window),
            }
            msg = (
                f"Episode {ep}/{episodes} | Steps {total_num_steps} | "
                f"Defense: {train_infos['defense_success_rate']:.2%} | "
                f"Attack: {train_infos['attack_success_rate']:.2%} | "
                f"DefReward: {train_infos['avg_defender_reward']:.2f} | "
                f"IntReward: {train_infos['avg_intruder_reward']:.2f}"
            )
            pbar.set_postfix_str(msg)
            self.log_train(train_infos, total_num_steps)

    def _compute_and_train(self):
        # Compute returns
        for buf, tr in ((self.def_buffer, self.def_trainer), (self.int_buffer, self.int_trainer)):
            self.def_trainer.prep_rollout()
            next_values = tr.policy.get_values(
                np.concatenate(buf.share_obs[-1]),
                np.concatenate(buf.rnn_states_critic[-1]),
                np.concatenate(buf.masks[-1])
            )
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            buf.compute_returns(next_values, tr.value_normalizer)

        # Train
        self.def_trainer.prep_training()
        self.int_trainer.prep_training()
        dv_info = self.def_trainer.train(self.def_buffer)
        iv_info = self.int_trainer.train(self.int_buffer)
        logs = {}
        for k, v in dv_info.items():
            logs[f'def/{k}'] = v
        for k, v in iv_info.items():
            logs[f'int/{k}'] = v
        self.log_train(logs, self.total_env_steps)

    def _save_all(self):
        torch.save(self.def_trainer.policy.actor.state_dict(), str(self.save_dir) + "/actor.pt")
        torch.save(self.def_trainer.policy.critic.state_dict(), str(self.save_dir) + "/critic.pt")
        torch.save(self.int_trainer.policy.actor.state_dict(), str(self.save_dir) + "/intruder_actor.pt")
        torch.save(self.int_trainer.policy.critic.state_dict(), str(self.save_dir) + "/intruder_critic.pt")
