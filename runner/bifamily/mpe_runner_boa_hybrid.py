#!/usr/bin/env python
"""
Stage-2 Bi-family Runner (Hybrid defender actions)
=================================================

Defenders: Tuple(Box(2), Discrete(N+1)) for (velocity, assignment)
Intruders: Box(2) velocity

Uses CPF-based dense shaping:
- Stage-1 period (min tau > tau_sw): -w_s1 * T_i * logdet(Pp_i)
- Stage-2 period: -w_s2 * T_i * delta_{d,i}

Assignments are rebalanced to avoid dog-pile and ensure ≥2 coverage if possible.
"""
from __future__ import annotations
import os
import time
import numpy as np
import torch
from gymnasium import spaces
from tqdm import trange

from runner.shared.base_runner import Runner
from utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class BiFamilyRunnerHybrid(Runner):
    def __init__(self, config):
        # Prevent base Runner from restoring a single shared policy with mismatched obs dims.
        orig_model_dir = getattr(config.get('all_args', None), 'model_dir', None)
        if 'all_args' in config and hasattr(config['all_args'], 'model_dir'):
            config['all_args'].model_dir = None
        super().__init__(config)
        self.stage1_model_dir = orig_model_dir

        self.num_defenders = int(getattr(self.all_args, 'num_defenders', self.num_agents))
        self.num_intruders = int(getattr(self.all_args, 'num_intruders', max(0, self.num_agents - self.num_defenders)))
        assert self.num_defenders + self.num_intruders == self.num_agents

        from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        obs_space = self.envs.observation_space[0]
        share_space = self.envs.share_observation_space[0] if self.use_centralized_V else obs_space
        box_action_space = self.envs.action_space[0]
        self.assign_action_space = spaces.Discrete(self.num_intruders + 1)
        self.hybrid_def_act_space = spaces.Tuple([box_action_space, self.assign_action_space])

        self.def_policy = Policy(self.all_args, obs_space, share_space, self.hybrid_def_act_space, device=self.device)
        self.int_policy = Policy(self.all_args, obs_space, share_space, box_action_space, device=self.device)

        self.def_trainer = TrainAlgo(self.all_args, self.def_policy, device=self.device)
        self.int_trainer = TrainAlgo(self.all_args, self.int_policy, device=self.device)

        self.def_buffer = SharedReplayBuffer(self.all_args, self.num_defenders, obs_space, share_space, self.hybrid_def_act_space)
        self.int_buffer = SharedReplayBuffer(self.all_args, self.num_intruders, obs_space, share_space, box_action_space)

        # Dense reward weights and kinematics
        self.w_s1 = float(getattr(self.all_args, 's1_logdet_weight', 1.0))
        self.w_s2 = float(getattr(self.all_args, 's2_delta_weight', 1.0))
        self.tau_sw = float(getattr(self.all_args, 'tau_switch', 60.0))
        self.capture_r = float(getattr(self.all_args, 'capture_r', 0.2))
        self.def_vmax = float(getattr(self.all_args, 'defender_max_speed', 1.0))
        self.lambda_threat = float(getattr(self.all_args, 'threat_lambda', 0.05))

        self.total_env_steps = 0
        # Mask state: last-step per-intruder coverage counts and captured flags
        self.last_counts = np.zeros((self.n_rollout_threads, self.num_intruders), dtype=np.int32)
        self.last_captured = np.zeros((self.n_rollout_threads, self.num_intruders), dtype=np.int32)

        # Optional: load Stage-1 shared model into defender policy
        def _safe_load(module, path):
            if not os.path.exists(path):
                return False
            try:
                ckpt = torch.load(path, map_location=self.device)
                cur = module.state_dict()
                filtered = {k: v for k, v in ckpt.items() if k in cur and v.shape == cur[k].shape}
                if not filtered:
                    return False
                cur.update(filtered)
                module.load_state_dict(cur)
                return True
            except Exception:
                return False

        if self.stage1_model_dir:
            _safe_load(self.def_policy.actor, os.path.join(self.stage1_model_dir, 'actor.pt'))
            _safe_load(self.def_policy.critic, os.path.join(self.stage1_model_dir, 'critic.pt'))
            # Also try loading intruder weights if present (for fair S2 warm-start)
            _safe_load(self.int_policy.actor, os.path.join(self.stage1_model_dir, 'intruder_actor.pt'))
            _safe_load(self.int_policy.critic, os.path.join(self.stage1_model_dir, 'intruder_critic.pt'))

    def run(self):
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self._buf_init(share_obs, obs)

        from collections import deque
        self.window_size = 100
        self.defense_success_window = deque(maxlen=self.window_size)
        self.attack_success_window = deque(maxlen=self.window_size)
        self.defender_reward_window = deque(maxlen=self.window_size)
        self.intruder_reward_window = deque(maxlen=self.window_size)
        self.episode_length_window = deque(maxlen=self.window_size)
        # sliding windows for CPF logdet (episode-average)
        # per-intruder average over last 100 episodes (across threads)
        self.logdet_intr_windows = [deque(maxlen=self.window_size) for _ in range(self.num_intruders)]
        # overall mean (across intruders) average over last 100 episodes
        self.logdet_mean_window = deque(maxlen=self.window_size)
        self.current_episode_rewards = {'defenders': np.zeros(self.n_rollout_threads), 'intruders': np.zeros(self.n_rollout_threads)}
        self.current_episode_steps = np.zeros(self.n_rollout_threads)
        # per-thread accumulators for Σ_t (per-intruder logdet) and steps
        self._ep_logdet_sum = np.zeros((self.n_rollout_threads, self.num_intruders), dtype=np.float64)

        self.log_per_step_logdet = False
        self.log_last_step_logdet = False
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        pbar = trange(episodes, desc=f"S2-BiHybrid-{self.algorithm_name}", leave=True)
        start = time.time()
        for ep in pbar:
            if self.use_linear_lr_decay:
                self.def_trainer.policy.lr_decay(ep, episodes)
                self.int_trainer.policy.lr_decay(ep, episodes)

            for step in range(self.episode_length):
                d_obs, i_obs = self._split_obs(obs)

                self.def_trainer.prep_rollout()
                self.int_trainer.prep_rollout()

                dV, dA, dL, dRS, dRSC, dAct_env, dAssign, _ = self._act_def(self.def_policy, self.def_buffer, step)
                iV, iA, iL, iRS, iRSC, iAct_env = self._act_int(self.int_policy, self.int_buffer, step)

                assignments = self._build_assignments_from_actions(dAssign)
                assignments = self._rebalance_assignments(assignments)
                self._update_counts_from_assignments(assignments)
                self._send_assignments(assignments)

                actions_env = self._merge_actions_env(dAct_env, iAct_env)
                obs_next, rewards, dones, infos = self.envs.step(actions_env)

                d_rew, i_rew = self._split_rewards(rewards)
                self._update_captured_from_infos(infos)
                self._apply_dense_rewards(d_rew, infos, d_obs, dAssign)

                # accumulate per-step CPF logdet per intruder for episode-average metric
                try:
                    for t in range(self.n_rollout_threads):
                        rep_info = infos[t][0] if len(infos[t]) > 0 else {}
                        L = rep_info.get('pz_logdet_list', None)
                        if L is not None:
                            for j in range(min(self.num_intruders, len(L))):
                                self._ep_logdet_sum[t, j] += float(L[j])
                except Exception:
                    pass

                self._accumulate_rewards_sw(d_rew, i_rew)
                self._handle_episode_end_sw(dones, infos)

                self._insert(self.def_buffer, dV, dA, dL, dRS, dRSC, d_rew, dones, obs_next, step, group='def')
                self._insert(self.int_buffer, iV, iA, iL, iRS, iRSC, i_rew, dones, obs_next, step, group='int')

                obs = obs_next
                # advance global env steps counter (for TB x-axis)
                self.total_env_steps += self.n_rollout_threads

            self._compute_and_train()
            self.def_buffer.after_update()
            self.int_buffer.after_update()

            if (ep % self.save_interval == 0 or ep == episodes - 1):
                self._save_all()
            if ep % self.log_interval == 0:
                total_num_steps = (ep + 1) * self.episode_length * self.n_rollout_threads
                self._log_training_info_sw(total_num_steps, ep, episodes, start, pbar)

    # ---- helpers ----
    def _buf_init(self, share_obs, obs):
        d_share, i_share = self._split_share_obs(share_obs)
        d_obs, i_obs = self._split_obs(obs)
        self.def_buffer.share_obs[0] = d_share.copy()
        self.def_buffer.obs[0] = d_obs.copy()
        self.int_buffer.share_obs[0] = i_share.copy()
        self.int_buffer.obs[0] = i_obs.copy()

    def _split_obs(self, obs):
        dN = self.num_defenders
        d_obs = obs[:, :dN]
        i_obs = obs[:, dN:]
        return d_obs, i_obs

    def _split_share_obs(self, share_obs):
        dN = self.num_defenders
        d_share = share_obs[:, :dN]
        i_share = share_obs[:, dN:]
        return d_share, i_share

    def _act_def(self, policy, buffer, step):
        avail_mask = self._build_available_mask_def()
        avail_tensor = torch.as_tensor(avail_mask, dtype=torch.float32, device=self.device)
        values, actions, action_log_probs, rnn_states, rnn_states_critic = policy.get_actions(
            np.concatenate(buffer.share_obs[step]),
            np.concatenate(buffer.obs[step]),
            np.concatenate(buffer.rnn_states[step]),
            np.concatenate(buffer.rnn_states_critic[step]),
            np.concatenate(buffer.masks[step]),
            available_actions=avail_tensor,
        )
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        dAct_env = actions[..., :2]
        dAssign = actions[..., 2:3]
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, dAct_env, dAssign, None

    def _act_int(self, policy, buffer, step):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = policy.get_actions(
            np.concatenate(buffer.share_obs[step]),
            np.concatenate(buffer.obs[step]),
            np.concatenate(buffer.rnn_states[step]),
            np.concatenate(buffer.rnn_states_critic[step]),
            np.concatenate(buffer.masks[step]))
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
        dN = self.num_defenders
        N = self.num_intruders
        out = []
        for t in range(self.n_rollout_threads):
            mapping = assignments[t].copy()
            counts = {i: 0 for i in range(N)}
            for d_idx, i_idx in mapping.items():
                if i_idx >= 0:
                    counts[i_idx] += 1
            # release extras > 2
            for i in range(N):
                if counts[i] > 2:
                    drop = counts[i] - 2
                    for d_idx, i_idx in list(mapping.items()):
                        if drop == 0:
                            break
                        if i_idx == i:
                            mapping[d_idx] = -1
                            drop -= 1
                    counts[i] = 2
            idle = [d for d, i_idx in mapping.items() if i_idx < 0]
            # ensure ≥1 then try ≥2
            for i in range(N):
                while counts[i] < 1 and idle:
                    d = idle.pop()
                    mapping[d] = i
                    counts[i] += 1
            for i in range(N):
                while counts[i] < 2 and idle:
                    d = idle.pop()
                    mapping[d] = i
                    counts[i] += 1
            out.append(mapping)
        return out

    def _update_counts_from_assignments(self, assignments):
        N = self.num_intruders
        for t in range(self.n_rollout_threads):
            cnt = np.zeros(N, dtype=np.int32)
            for _, i_idx in assignments[t].items():
                if i_idx >= 0:
                    cnt[i_idx] += 1
            self.last_counts[t] = cnt

    def _update_captured_from_infos(self, infos):
        N = self.num_intruders
        for t in range(self.n_rollout_threads):
            cap = np.zeros(N, dtype=np.int32)
            for info in infos[t]:
                for j in range(N):
                    key = f"intruder_{j}_captured"
                    if key in info and int(info[key]) == 1:
                        cap[j] = 1
            self.last_captured[t] = cap

    def _build_available_mask_def(self):
        nT = self.n_rollout_threads
        dN = self.num_defenders
        N = self.num_intruders
        masks = []
        for t in range(nT):
            base = np.ones((dN, N + 1), dtype=np.float32)
            base[:, 0] = 1.0  # idle allowed
            for j in range(N):
                if self.last_captured[t, j] == 1 or self.last_counts[t, j] >= 2:
                    base[:, j + 1] = 0.0
            masks.append(base)
        return np.concatenate(masks, axis=0)

    def _merge_actions_env(self, d_vel, i_vel):
        return np.concatenate([d_vel, i_vel], axis=1)

    def _split_rewards(self, rewards):
        dN = self.num_defenders
        return rewards[:, :dN], rewards[:, dN:]

    def _split_dones(self, dones):
        arr = np.asarray(dones)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        dN = self.num_defenders
        return arr[:, :dN].astype(bool), arr[:, dN:].astype(bool)

    def _insert(self, buffer, values, actions, action_log_probs, rnn_states, rnn_states_critic, rewards, dones, obs_next, step, group='def'):
        d_d, i_d = self._split_dones(dones)
        dones_group = d_d if group == 'def' else i_d
        rnn_states[dones_group == True] = np.zeros(((dones_group == True).sum(), buffer.recurrent_N, buffer.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_group == True] = np.zeros(((dones_group == True).sum(), *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, buffer.num_agents, 1), dtype=np.float32)
        masks[dones_group == True] = np.zeros(((dones_group == True).sum(), 1), dtype=np.float32)
        if self.use_centralized_V:
            share_obs = obs_next.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs_next
        d_share, i_share = self._split_share_obs(share_obs)
        d_obs, i_obs = self._split_obs(obs_next)
        share_use, obs_use = (d_share, d_obs) if group == 'def' else (i_share, i_obs)
        buffer.insert(share_use, obs_use, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    def _apply_dense_rewards(self, d_rew, infos, d_obs, assign_actions):
        eps = 1e-12
        dN = self.num_defenders
        N = self.num_intruders
        lam = self.lambda_threat
        for t in range(self.n_rollout_threads):
            info0 = infos[t][0] if len(infos[t]) > 0 else {}
            L = info0.get('pz_logdet_list', None)
            T = info0.get('pz_T_list', None)
            MU = info0.get('pz_mu_list', None)
            if L is None or T is None or MU is None:
                continue
            taus = []
            for Ti in T:
                Ti = max(float(Ti), eps)
                taus.append(-np.log(Ti) / max(lam, eps))
            stage1 = (min(taus) > self.tau_sw)
            for d_idx in range(dN):
                cat = int(assign_actions[t, d_idx, 0]) if assign_actions is not None else 0
                intr_idx = cat - 1 if cat > 0 else -1
                if intr_idx < 0 or intr_idx >= N:
                    continue
                Ti = float(T[intr_idx])
                logdet_i = float(L[intr_idx])
                if stage1:
                    contrib = - self.w_s1 * Ti * logdet_i
                else:
                    s_px = float(d_obs[t, d_idx, 0])
                    s_py = float(d_obs[t, d_idx, 1])
                    mu_i = MU[intr_idx]
                    dist = float(np.linalg.norm(np.array([mu_i[0] - s_px, mu_i[1] - s_py])))
                    delta = max(0.0, dist - self.capture_r) / max(self.def_vmax, eps)
                    contrib = - self.w_s2 * Ti * delta
                d_rew[t, d_idx, 0] += contrib

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
                # episode-average CPF logdet per intruder and overall mean (smooth over 100 episodes)
                if ep_steps > 0:
                    ep_avg_L = self._ep_logdet_sum[t, :] / ep_steps
                    # overall mean across intruders
                    self.logdet_mean_window.append(float(np.mean(ep_avg_L)))
                    # per-intruder windows
                    for j in range(self.num_intruders):
                        self.logdet_intr_windows[j].append(float(ep_avg_L[j]))
                # optional: last-step snapshot logging (disabled by default)
                if self.log_last_step_logdet:
                    try:
                        if 'pz_logdet_list' in info0:
                            logdet_list = info0['pz_logdet_list']
                            env_infos = {f"pz_ep/logdet_last/intruder_{i}": float(v) for i, v in enumerate(logdet_list)}
                            if 'pz_T_list' in info0:
                                T_list = info0['pz_T_list']
                                if len(T_list) == len(logdet_list) and len(T_list) > 0:
                                    weighted = float(np.sum(np.array(T_list) * np.array(logdet_list)))
                                    env_infos["pz_ep/logdet_weighted_last"] = weighted
                            self.log_env(env_infos, self.total_env_steps)
                    except Exception:
                        pass
                self.current_episode_rewards['defenders'][t] = 0.0
                self.current_episode_rewards['intruders'][t] = 0.0
                self.current_episode_steps[t] = 0
                self._ep_logdet_sum[t, :] = 0.0

    def _log_training_info_sw(self, total_num_steps, ep, episodes, start, pbar):
        end = time.time()
        fps = int(total_num_steps / max(end - start, 1e-6))
        if len(self.defense_success_window) > 0:
            train_infos = {
                'defense_success_rate': float(np.mean(self.defense_success_window)),
                'attack_success_rate': float(np.mean(self.attack_success_window)),
                'avg_defender_reward': float(np.mean(self.defender_reward_window)),
                'avg_intruder_reward': float(np.mean(self.intruder_reward_window)),
                'avg_episode_length': float(np.mean(self.episode_length_window)),
                'training_fps': fps,
                'window_episodes': len(self.defense_success_window),
            }
            # overall mean logdet sliding average
            if len(self.logdet_mean_window) > 0:
                train_infos['pz_ep/logdet_mean_avg'] = float(np.mean(self.logdet_mean_window))
            # per-intruder logdet sliding averages
            for j in range(self.num_intruders):
                if len(self.logdet_intr_windows[j]) > 0:
                    train_infos[f'pz_ep/logdet_avg/intruder_{j}'] = float(np.mean(self.logdet_intr_windows[j]))
            msg = (
                f"Episode {ep}/{episodes} | Steps {total_num_steps} | FPS {fps} | "
                f"Defense: {train_infos['defense_success_rate']:.2%} | "
                f"Attack: {train_infos['attack_success_rate']:.2%} | "
                f"DefReward: {train_infos['avg_defender_reward']:.2f} | "
                f"IntReward: {train_infos['avg_intruder_reward']:.2f}"
            )
            pbar.set_postfix_str(msg)
            self.log_train(train_infos, total_num_steps)

    def _compute_and_train(self):
        for buf, tr in ((self.def_buffer, self.def_trainer), (self.int_buffer, self.int_trainer)):
            tr.prep_rollout()
            next_values = tr.policy.get_values(
                np.concatenate(buf.share_obs[-1]),
                np.concatenate(buf.rnn_states_critic[-1]),
                np.concatenate(buf.masks[-1])
            )
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            buf.compute_returns(next_values, tr.value_normalizer)

        self.def_trainer.prep_training()
        self.int_trainer.prep_training()
        dv_info = self.def_trainer.train(self.def_buffer)
        iv_info = self.int_trainer.train(self.int_buffer)
        fam_logs = {**{f'def/{k}': v for k, v in dv_info.items()}, **{f'int/{k}': v for k, v in iv_info.items()}}
        self.log_train(fam_logs, self.total_env_steps)

    def _save_all(self):
        torch.save(self.def_trainer.policy.actor.state_dict(), str(self.save_dir) + "/defender_actor.pt")
        torch.save(self.def_trainer.policy.critic.state_dict(), str(self.save_dir) + "/defender_critic.pt")
        torch.save(self.int_trainer.policy.actor.state_dict(), str(self.save_dir) + "/intruder_actor.pt")
        torch.save(self.int_trainer.policy.critic.state_dict(), str(self.save_dir) + "/intruder_critic.pt")

    def _send_assignments(self, assignments):
        """Forward defender->intruder mapping per env to the vectorized env.
        assignments: list[dict[int,int]] length == n_rollout_threads, value -1 means idle.
        """
        if hasattr(self.envs, 'set_assignments'):
            self.envs.set_assignments(assignments)
