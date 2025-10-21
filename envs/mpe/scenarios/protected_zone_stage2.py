# -*- coding: utf-8 -*-
"""
Stage 2 Scenario: Protected-zone defense with CPF and Threat Metrics
===================================================================

This scenario keeps Stage 1 physics/termination while adding a complete
Centralized Particle Filter (CPF) per intruder to compute the posterior
position covariance and a look-ahead threat metric. At every step, the info
dict provides per-intruder logdet(P_pos) and a threat-weighted sum for
TensorBoard.

Assignments (which defender measures which intruder) can be provided via
world._pz_assignments as a dict {defender_idx: intruder_idx or -1}. If missing,
the scenario builds a feasible set that ensures ≥2 bearings per active intruder
by greedily selecting nearest defenders (count-agnostic, parameter-free).
"""

import numpy as np
from envs.mpe.core import World, Agent, Landmark, Wall
from envs.mpe.scenario import BaseScenario
from boa.cpf import (
    F_CA_dt, Q_CA_dt, predict_particles, update_weights_bearing,
    resample_systematic, ess, weighted_mean_and_cov, aoa_variance,
)
from boa.threat import (
    earliest_collision_time, threat_weight,
)


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.dim_p = 2
        world.dim_c = 0
        world.world_length = getattr(args, "episode_length", 300)
        world.dt = 0.1
        world.damping = 0.15

        # 基础几何
        world._pz_center = np.array([
            getattr(args, "protected_cx", 0.0),
            getattr(args, "protected_cy", 0.0)
        ], dtype=np.float32)
        world._pz_r = float(getattr(args, "protected_r", 0.5))
        world._pz_world_r = float(getattr(args, "world_r", 5.0))
        world._pz_capture = float(getattr(args, "capture_r", 0.2))

        # 运动参数
        def_spd = float(getattr(args, "defender_max_speed", 1.2))
        intr_spd = float(getattr(args, "intruder_max_speed", 1.0))
        accel = 4.0

        # Base shaping (Stage-1 compatible keys)
        world._pz_time_penalty = float(getattr(args, "time_penalty", -0.01))
        world._pz_vel_penalty = float(getattr(args, "vel_penalty", -0.001))
        world._pz_capture_reward = float(getattr(args, "capture_reward", 15.0))
        world._pz_entry_penalty = float(getattr(args, "entry_penalty", -15.0))
        world._pz_distance_reward = float(getattr(args, "distance_reward", 0.03))
        world._pz_formation_reward = float(getattr(args, "formation_reward", 0.08))
        world._pz_threat_weight_reward = float(getattr(args, "threat_weight_reward", 0.02))
        world._pz_intercept_reward = float(getattr(args, "intercept_reward", 0.05))
        world._pz_use_s1_rewards = bool(getattr(args, "use_stage1_rewards", True))

        # AoA noise model
        world._aoa_sigma0 = float(getattr(args, "bearing_sigma0", 0.02))
        world._aoa_r0 = float(getattr(args, "bearing_r0", 0.5))

        # CPF parameters (fully configurable)
        world._cpf_num_particles = int(getattr(args, "cpf_num_particles", 256))
        world._cpf_sigma_a = float(getattr(args, "cpf_sigma_a", 0.5))
        world._cpf_init_pos_std = float(getattr(args, "cpf_init_pos_std", 1.0))
        world._cpf_init_vel_std = float(getattr(args, "cpf_init_vel_std", 0.5))
        world._cpf_init_acc_std = float(getattr(args, "cpf_init_acc_std", 0.2))
        world._cpf_ess_ratio = float(getattr(args, "cpf_resample_ess_ratio", 0.5))

        # Threat look-ahead
        world._threat_lambda = float(getattr(args, "threat_lambda", 0.05))
        world._threat_tau_max = float(getattr(args, "threat_tau_max", 60.0))
        world._threat_tau_step = float(getattr(args, "threat_tau_step", 1.0))

        # Intruder sensing (range-limited KNN)
        world._intruder_sense_radius = float(getattr(args, "intruder_sense_radius", 2.0))
        world._intruder_max_neighbors = int(getattr(args, "intruder_max_neighbors", 2))

        # 统计变量
        world._pz_captured_ids = set()
        world._pz_zone_entry = False
        world._pz_total_captured = 0
        world._pz_zone_entered = False

        # 智能体
        M = int(getattr(args, "num_defenders", 5))
        N = int(getattr(args, "num_intruders", 2))
        world.agents = []
        self.defenders = []
        self.intruders = []

        for i in range(M):
            a = Agent()
            a.name = f"defender_{i}"
            a.adversary = False
            a.silent = True
            a.movable = True
            a.collide = True
            a.u_range = 1.0
            a.accel = accel
            a.max_speed = def_spd
            a.size = 0.06
            a.color = np.array([0.25, 0.75, 0.25])
            world.agents.append(a)
            self.defenders.append(a)

        for j in range(N):
            a = Agent()
            a.name = f"intruder_{j}"
            a.adversary = True
            a.silent = True
            a.movable = True
            a.collide = True
            a.u_range = 1.0
            a.accel = accel
            a.max_speed = intr_spd
            a.size = 0.05
            a.color = np.array([0.25, 0.25, 0.75])
            world.agents.append(a)
            self.intruders.append(a)

        # 边界墙
        R = world._pz_world_r
        wall_width = 0.05
        world.walls = [
            Wall(orient='H', axis_pos=+R, endpoints=(-R, R), width=wall_width, hard=True),
            Wall(orient='H', axis_pos=-R, endpoints=(-R, R), width=wall_width, hard=True),
            Wall(orient='V', axis_pos=-R, endpoints=(-R, R), width=wall_width, hard=True),
            Wall(orient='V', axis_pos=+R, endpoints=(-R, R), width=wall_width, hard=True),
        ]

        # 可视化地标
        landmark = Landmark()
        landmark.name = "protected_zone"
        landmark.collide = False
        landmark.movable = False
        landmark.size = world._pz_r
        landmark.boundary = False
        landmark.color = np.array([0.75, 0.25, 0.25])
        landmark.state.p_pos = world._pz_center.copy()
        landmark.state.p_vel = np.zeros(2, dtype=np.float32)
        world.landmarks = [landmark]

        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        world.world_step = 0
        world._pz_captured_ids.clear()
        world._pz_zone_entry = False
        world._pz_zone_entered = False
        world._pz_total_captured = 0

        if not hasattr(world, '_pz_already_captured'):
            world._pz_already_captured = set()
        else:
            world._pz_already_captured.clear()

        # Place defenders on the zone boundary (evenly spaced)
        m = len(self.defenders)
        if m > 0:
            angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
            for i, d in enumerate(self.defenders):
                ang = angles[i] + np.random.uniform(-0.2, 0.2)
                pos = world._pz_center + world._pz_r * np.array([np.cos(ang), np.sin(ang)])
                d.state.p_pos = pos.astype(np.float32)
                d.state.p_vel = np.zeros(2, dtype=np.float32)

        # Place intruders on the world boundary facing the center
        for it in self.intruders:
            it.movable = True
            ang = np.random.uniform(0, 2 * np.pi)
            pos = world._pz_center + world._pz_world_r * np.array([np.cos(ang), np.sin(ang)])
            it.state.p_pos = pos.astype(np.float32)
            to_center = world._pz_center - it.state.p_pos
            n = np.linalg.norm(to_center)
            if n > 1e-6:
                dirc = to_center / n
            else:
                dirc = np.array([1.0, 0.0])
            it.state.p_vel = (dirc * (0.5 * it.max_speed)).astype(np.float32)

        # Initialize a CPF per intruder
        rng = np.random
        world._cpf_filters = []
        Np = world._cpf_num_particles
        for it in self.intruders:
            # State: [px, vx, ax, py, vy, ay]
            mu0 = np.array([
                it.state.p_pos[0], 0.0, 0.0,
                it.state.p_pos[1], 0.0, 0.0
            ], dtype=np.float32)
            std = np.array([
                world._cpf_init_pos_std, world._cpf_init_vel_std, world._cpf_init_acc_std,
                world._cpf_init_pos_std, world._cpf_init_vel_std, world._cpf_init_acc_std,
            ], dtype=np.float32)
            X0 = rng.normal(loc=mu0, scale=std, size=(Np, 6)).astype(np.float32)
            w0 = np.full(Np, 1.0 / Np, dtype=np.float32)
            world._cpf_filters.append({
                'X': X0,
                'w': w0,
                'mu': mu0.copy(),
                'Px': np.diag(std ** 2).astype(np.float32),
            })

    # ========== CPF & threat ==========
    def _build_measurement_assignments(self, world: World) -> dict[int, list[int]]:
        """Return mapping intruder_idx -> list of defender indices assigned for AoA.
        If world._pz_assignments exists (defender -> intruder or -1), honor it while
        ensuring ≥2 per active intruder by greedy supplementation; otherwise build
        an assignment by selecting nearest defenders until constraints are met.
        """
        M = len(self.defenders)
        N = len(self.intruders)
        already = getattr(world, '_pz_already_captured', set())
        mapping = {i: [] for i in range(N) if i not in already}

        if hasattr(world, '_pz_assignments') and isinstance(world._pz_assignments, dict):
            for d_idx, i_idx in world._pz_assignments.items():
                i_idx = int(i_idx)
                if 0 <= i_idx < N and i_idx not in already:
                    mapping[i_idx].append(int(d_idx))

        assigned_defenders = set([d for lst in mapping.values() for d in lst])
        for i in range(N):
            if i in already:
                continue
            need = max(0, 2 - len(mapping[i]))
            if need <= 0:
                continue
            dists = []
            for d_idx, d in enumerate(self.defenders):
                if d_idx in assigned_defenders:
                    continue
                dist = float(np.linalg.norm(self.intruders[i].state.p_pos - d.state.p_pos))
                dists.append((dist, d_idx))
            dists.sort(key=lambda x: x[0])
            for _, d_idx in dists[:need]:
                mapping[i].append(d_idx)
                assigned_defenders.add(d_idx)
        return mapping

    def _cpf_step(self, world: World):
        """Run one CPF predict/update and compute per-intruder threat & logdet."""
        dt = world.dt
        F = F_CA_dt(dt)
        Q = Q_CA_dt(dt, world._cpf_sigma_a)
        rng = np.random
        sigma0 = world._aoa_sigma0
        r0 = world._aoa_r0

        assign = self._build_measurement_assignments(world)
        center = world._pz_center
        zone_r = world._pz_r

        for j, it in enumerate(self.intruders):
            if j in getattr(world, '_pz_already_captured', set()):
                world._cpf_filters[j]['logdet'] = 0.0
                world._cpf_filters[j]['T'] = 0.0
                continue
            filt = world._cpf_filters[j]
            X = filt['X']
            w = filt['w']
            X = predict_particles(X, F, Q, rng)

            d_idx_list = assign.get(j, [])
            d_positions, z_list, var_list = [], [], []
            for d_idx in d_idx_list:
                d = self.defenders[d_idx]
                true_bearing = float(np.arctan2(it.state.p_pos[1] - d.state.p_pos[1], it.state.p_pos[0] - d.state.p_pos[0]))
                r = float(np.linalg.norm(it.state.p_pos - d.state.p_pos))
                var = aoa_variance(r, sigma0, r0)
                meas = rng.normal(loc=true_bearing, scale=np.sqrt(var))
                d_positions.append(d.state.p_pos.copy())
                z_list.append(float(meas))
                var_list.append(float(var))

            w = update_weights_bearing(X, w, d_positions, z_list, var_list)
            if ess(w) < world._cpf_ess_ratio * len(w):
                X, w = resample_systematic(X, w, rng)

            mu, Px = weighted_mean_and_cov(X, w)
            world._cpf_filters[j]['X'] = X
            world._cpf_filters[j]['w'] = w
            world._cpf_filters[j]['mu'] = mu
            world._cpf_filters[j]['Px'] = Px

            Jp = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.float32)
            Pp = Jp @ Px @ Jp.T
            detP = float(np.linalg.det(Pp))
            logdetP = float(np.log(max(detP, 1e-12)))
            world._cpf_filters[j]['logdet'] = logdetP

            tau = earliest_collision_time(
                mu_x=mu,
                Px=Px,
                zone_center=center,
                zone_radius=zone_r,
                sigma_a=world._cpf_sigma_a,
                tau_max=world._threat_tau_max,
                tau_step=world._threat_tau_step,
            )
            Tj = threat_weight(tau, world._threat_lambda)
            world._cpf_filters[j]['T'] = Tj

    # ========== Reward / Observation / Done / Info ==========
    def reward(self, agent: Agent, world: World):
        """Stage-2 reward = Stage-1 shaping terms (truth-based geometry) + CPF-based dense terms (via runner).
        This function returns Stage-1 style reward; runner may add extra dense terms.
        """
        if getattr(world, '_pz_use_s1_rewards', True):
            if agent.adversary:
                return float(self._intruder_reward_s1(agent, world))
            else:
                return float(self._defender_reward_s1(agent, world))
        # fallback: minimal shaping
        reward = world._pz_time_penalty
        reward += world._pz_vel_penalty * np.linalg.norm(agent.state.p_vel)
        if world._pz_zone_entry:
            reward += -1.0 if not agent.adversary else +1.0
        return float(reward)

    def observation(self, agent: Agent, world: World):
        obs = []
        # 1) self state
        obs.extend(agent.state.p_pos.tolist())  # 2
        obs.extend(agent.state.p_vel.tolist())  # 2
        # 2) vector to center
        to_center = world._pz_center - agent.state.p_pos
        obs.extend(to_center.tolist())          # 2  => base 6

        if agent.adversary:
            # Intruder: K-NN neighbors (defenders + intruders) within radius, capped at kmax
            rs = float(getattr(world, '_intruder_sense_radius', 2.0))
            kmax = int(getattr(world, '_intruder_max_neighbors', 2))
            rels = []
            # defenders
            for d in self.defenders:
                dv = (d.state.p_pos - agent.state.p_pos)
                dist = float(np.linalg.norm(dv))
                if dist <= rs:
                    rels.append((dist, dv, d.state.p_vel - agent.state.p_vel))
            # other intruders
            for it in self.intruders:
                if it is agent:
                    continue
                dv = (it.state.p_pos - agent.state.p_pos)
                dist = float(np.linalg.norm(dv))
                if dist <= rs:
                    rels.append((dist, dv, it.state.p_vel - agent.state.p_vel))
            rels.sort(key=lambda x: x[0])
            rels = rels[:kmax]
            for _, dv, vv in rels:
                obs.extend(dv.tolist())  # 2
                obs.extend(vv.tolist())  # 2  => up to 4*kmax
            # pad neighbors to fixed kmax slots for determinism
            need_pad = kmax - len(rels)
            if need_pad > 0:
                obs.extend([0.0] * (4 * need_pad))
        else:
            # Defender: use CPF posteriors for intruders (fixed-size per intruder)
            already = getattr(world, '_pz_already_captured', set())
            for j, _it in enumerate(self.intruders):
                mu = world._cpf_filters[j]['mu']
                # posterior mean position/velocity (from 6-state)
                obs.extend((mu[0:1].tolist() + mu[3:4].tolist()))  # px, py (2)
                obs.extend((mu[1:2].tolist() + mu[4:5].tolist()))  # vx, vy (2)
                obs.append(float(world._cpf_filters[j].get('logdet', 0.0)))   # 1
                obs.append(float(world._cpf_filters[j].get('T', 0.0)))        # 1
                obs.append(1.0 if j in already else 0.0)                      # 1  => 7 per intruder
            # peers (other defenders): relative pos/vel, fixed-size
            for d in self.defenders:
                if d is agent:
                    continue
                obs.extend((d.state.p_pos - agent.state.p_pos).tolist())  # 2
                obs.extend(d.state.p_vel.tolist())                        # 2  => 4 per other defender

        # Unify observation length across all agents to the defender-sized template:
        # target_len = 6 + 7*N_intruders + 4*(N_defenders-1)
        N = len(self.intruders)
        M = len(self.defenders)
        target_len = 6 + 7 * N + 4 * max(M - 1, 0)

        arr = np.array(obs, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        if arr.shape[0] < target_len:
            pad = np.zeros(target_len - arr.shape[0], dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > target_len:
            arr = arr[:target_len]
        arr = np.clip(arr, -10.0, 10.0)
        return arr

    def done(self, agent: Agent, world: World):
        if world.world_step >= world.world_length:
            return True
        if world._pz_zone_entry:
            return True
        already = getattr(world, '_pz_already_captured', set())
        if len(already) >= len(self.intruders):
            return True
        return False

    def info(self, agent: Agent, world: World):
        info = {}
        # Capture & breach logic
        if agent.adversary:
            idx = self.intruders.index(agent) if agent in self.intruders else -1
            dist = np.linalg.norm(agent.state.p_pos - world._pz_center)
            already = getattr(world, '_pz_already_captured', set())
            info[f"intruder_{idx}_captured"] = float(1 if idx in already else 0)
            info[f"intruder_{idx}_breakthrough"] = float(1 if (world._pz_zone_entry and dist <= world._pz_r) else 0)
        else:
            # Only the first defender writes global statistics
            if agent.name.endswith("_0"):
                already = getattr(world, '_pz_already_captured', set())
                defense_success = 1 if (len(already) >= len(self.intruders) and not world._pz_zone_entry) else 0
                attack_success = 1 if world._pz_zone_entry else 0

                # CPF-based logdet and threat
                logdet_list = []
                T_list = []
                for j in range(len(self.intruders)):
                    if j in already:
                        logdet_list.append(0.0)
                        T_list.append(0.0)
                    else:
                        logdet_list.append(float(world._cpf_filters[j].get('logdet', 0.0)))
                        T_list.append(float(world._cpf_filters[j].get('T', 0.0)))
                weighted = float(np.sum(np.array(T_list) * np.array(logdet_list))) if len(logdet_list) > 0 else 0.0

                mu_list = []
                for j in range(len(self.intruders)):
                    mu_j = world._cpf_filters[j]['mu']
                    mu_list.append([float(mu_j[0]), float(mu_j[3])])

                info.update({
                    "defense_success": float(defense_success),
                    "attack_success": float(attack_success),
                    "episode_length": float(world.world_step),
                    "total_captured": float(len(already)),
                    "total_intruders": float(len(self.intruders)),
                    # per-step logging keys for runner
                    "pz_logdet_list": [float(x) for x in logdet_list],
                    "pz_T_list": [float(x) for x in T_list],
                    "pz_logdet_weighted": weighted,
                    "pz_mu_list": mu_list,
                })

        # Optional distances
        if len(self.intruders) > 0 and len(self.defenders) > 0:
            total, cnt = 0.0, 0
            for d in self.defenders:
                for it in self.intruders:
                    total += np.linalg.norm(d.state.p_pos - it.state.p_pos)
                    cnt += 1
            if cnt > 0:
                info["avg_defender_intruder_distance"] = float(total / cnt)
        return info

    # ===== Stage-1 style rewards (ported for S2 on top of CPF) =====
    def _get_threat_weights_from_cpf(self, world: World):
        T = []
        already = getattr(world, '_pz_already_captured', set())
        for j in range(len(self.intruders)):
            if j in already:
                T.append(0.0)
            else:
                T.append(float(world._cpf_filters[j].get('T', 0.0)))
        return T

    def _defender_reward_s1(self, agent: Agent, world: World):
        reward = 0.0

        # personal capture (nearest defender gets full credit)
        personal_capture = 0.0
        s = agent.state.p_pos
        for j in world._pz_captured_ids:
            it = self.intruders[j]
            dself = np.linalg.norm(s - it.state.p_pos)
            closest = True
            for d2 in self.defenders:
                if d2 is agent:
                    continue
                if np.linalg.norm(d2.state.p_pos - it.state.p_pos) < dself:
                    closest = False
                    break
            if closest:
                personal_capture += world._pz_capture_reward
        reward += personal_capture

        # team capture bonus (10%)
        reward += 0.1 * world._pz_capture_reward * len(world._pz_captured_ids)

        # breach penalty
        if world._pz_zone_entry:
            reward += world._pz_entry_penalty

        # spacing among defenders (collision avoidance + near-ideal spacing)
        reward += self._compute_optimal_spacing_reward(agent, world)

        # threat focus (use CPF T)
        reward += self._compute_threat_focus_reward(agent, world)

        # interception encouragement (threat-weighted closeness to intruders)
        reward += self._compute_intercept_reward(agent, world)

        # time/velocity regularization
        reward += world._pz_time_penalty
        reward += world._pz_vel_penalty * np.linalg.norm(agent.state.p_vel)

        return reward

    def _intruder_reward_s1(self, agent: Agent, world: World):
        reward = 0.0
        try:
            idx = self.intruders.index(agent)
        except ValueError:
            idx = -1
        already = getattr(world, '_pz_already_captured', set())
        if idx in already:
            return 0.0

        # breach reward (positive)
        dist_c = np.linalg.norm(agent.state.p_pos - world._pz_center)
        if dist_c <= world._pz_r:
            reward += abs(world._pz_entry_penalty)

        # approach center (within band)
        thr = world._pz_r + 2.5
        if dist_c <= thr:
            ratio = (thr - dist_c) / thr
            reward += world._pz_distance_reward * ratio * 2.0

        # direction alignment towards center
        to_c = world._pz_center - agent.state.p_pos
        n_to = np.linalg.norm(to_c)
        spd = np.linalg.norm(agent.state.p_vel)
        if n_to > 1e-6 and spd > 1e-6:
            dirc = to_c / n_to
            vu = agent.state.p_vel / spd
            align = float(np.dot(dirc, vu))
            if align > 0:
                reward += world._pz_distance_reward * 1.0 * align

        # avoid other intruders
        reward += self._compute_intruder_spacing_reward(agent, world)

        # avoid defenders (weak)
        if len(self.defenders) > 0:
            mind = min(np.linalg.norm(agent.state.p_pos - d.state.p_pos) for d in self.defenders)
            reward += world._pz_distance_reward * 0.01 * mind

        # time/velocity regularization
        reward += world._pz_time_penalty
        reward += world._pz_vel_penalty * np.linalg.norm(agent.state.p_vel)
        return reward

    def _compute_intercept_reward(self, agent: Agent, world: World):
        if agent.adversary:
            return 0.0
        s = agent.state.p_pos
        R = max(world._pz_world_r, 1e-6)
        already = getattr(world, '_pz_already_captured', set())
        T = self._get_threat_weights_from_cpf(world)
        total = 0.0
        for i, it in enumerate(self.intruders):
            if i in already:
                continue
            d = np.linalg.norm(s - it.state.p_pos)
            closeness = max(0.0, (R - d) / R)
            Ti = float(T[i]) if i < len(T) else 0.0
            total += Ti * closeness
        return world._pz_intercept_reward * total

    def _compute_threat_focus_reward(self, agent: Agent, world: World):
        T = self._get_threat_weights_from_cpf(world)
        if len(T) == 0:
            return 0.0
        maxT = max(T)
        if maxT <= 0.0:
            return 0.0
        j = int(np.argmax(T))
        it = self.intruders[j]
        d = np.linalg.norm(agent.state.p_pos - it.state.p_pos)
        maxD = 2 * world._pz_world_r
        closeness = (maxD - d) / max(maxD, 1e-6)
        return world._pz_threat_weight_reward * closeness * maxT

    def _compute_optimal_spacing_reward(self, agent: Agent, world: World):
        if len(self.defenders) < 2:
            return 0.0
        spacing_reward = 0.0
        s = agent.state.p_pos
        min_safety = 0.25
        ideal = world._pz_r * 1.2
        dists = []
        for d2 in self.defenders:
            if d2 is agent:
                continue
            dist = np.linalg.norm(s - d2.state.p_pos)
            dists.append(dist)
            if dist < min_safety:
                spacing_reward += -world._pz_distance_reward * 5.0 * (min_safety - dist) / min_safety
            elif dist < ideal * 1.5:
                score = 1.0 - abs(dist - ideal) / max(ideal, 1e-6)
                spacing_reward += world._pz_distance_reward * 0.5 * max(0.0, score)
        if len(dists) > 1:
            std = float(np.std(dists))
            spacing_reward += world._pz_distance_reward * 0.2 / (1.0 + std)
        return spacing_reward

    def _compute_intruder_spacing_reward(self, agent: Agent, world: World):
        if len(self.intruders) < 2:
            return 0.0
        reward = 0.0
        s = agent.state.p_pos
        min_safe = 0.2
        for it in self.intruders:
            if it is agent:
                continue
            dist = np.linalg.norm(s - it.state.p_pos)
            if dist < min_safe:
                reward += -world._pz_distance_reward * 3.0 * (min_safe - dist) / min_safe
            elif dist < min_safe * 2:
                reward += world._pz_distance_reward * 0.1 * (dist - min_safe) / min_safe
        return reward

    def _compute_step_events(self, world: World):
        world._pz_captured_ids.clear()
        world._pz_zone_entry = False

        # Capture event
        for j, it in enumerate(self.intruders):
            if j in getattr(world, '_pz_already_captured', set()):
                it.state.p_vel = np.zeros(2)
                it.movable = False
                continue
            for d in self.defenders:
                if np.linalg.norm(it.state.p_pos - d.state.p_pos) <= world._pz_capture:
                    if not hasattr(world, '_pz_already_captured'):
                        world._pz_already_captured = set()
                    world._pz_already_captured.add(j)
                    world._pz_captured_ids.add(j)
                    world._pz_total_captured += 1
                    it.state.p_vel = np.zeros(2)
                    it.movable = False
                    break

        # Breach event
        for it in self.intruders:
            if np.linalg.norm(it.state.p_pos - world._pz_center) <= world._pz_r:
                world._pz_zone_entry = True
                if not world._pz_zone_entered:
                    world._pz_zone_entered = True
                break

    def post_step_callback(self, world: World):
        # Step events, freeze captured intruders, then run CPF
        self._compute_step_events(world)
        already = getattr(world, '_pz_already_captured', set())
        for i, it in enumerate(self.intruders):
            if i in already:
                if hasattr(it, 'action') and hasattr(it.action, 'u'):
                    it.action.u = np.zeros(2, dtype=np.float32)
                it.state.p_vel = np.zeros(2, dtype=np.float32)
                it.movable = False
        self._cpf_step(world)
