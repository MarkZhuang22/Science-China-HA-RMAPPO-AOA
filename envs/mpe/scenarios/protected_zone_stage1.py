# -*- coding: utf-8 -*-
"""
Stage-1 scenario: protected-zone defense (basic motion and sensing prep)
=======================================================================

Goals:
1) Defenders learn basic motion and position control.
2) Intruders learn to move toward the protected zone.
3) Both sides learn pursuit/evade behavior.
4) Defenders practice threat assessment and non-collinear cooperation.
5) Full observation is provided to ease learning and prepare for Stage-2 AoA.

Notes:
- Uses simple shaping rewards and full observability.
- Keeps the physics model, boundary, capture/breach events, and basic control.
"""

import numpy as np
from envs.mpe.core import World, Agent, Landmark, Wall
from envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        """Build the simplified Stage-1 world."""
        world = World()
        world.dim_p = 2  # 2D position
        world.dim_c = 0  # no communication
        world.world_length = getattr(args, "episode_length", 300)
        world.dt = 0.1
        world.damping = 0.15  # consistent with core.py

        # Geometry
        world._pz_center = np.array([
            getattr(args, "protected_cx", 0.0),
            getattr(args, "protected_cy", 0.0)
        ], dtype=np.float32)
        world._pz_r = float(getattr(args, "protected_r", 0.5))
        world._pz_world_r = float(getattr(args, "world_r", 5.0))
        world._pz_capture = float(getattr(args, "capture_r", 0.2))

        # Motion parameters
        def_spd = float(getattr(args, "defender_max_speed", 1.2))  # defender slightly faster to help interception
        intr_spd = float(getattr(args, "intruder_max_speed", 1.0))  # intruder speed
        accel = 4.0  # moderate accel for smoother control

        # Reward parameters
        world._pz_capture_reward = float(getattr(args, "capture_reward", 15.0))
        world._pz_entry_penalty = float(getattr(args, "entry_penalty", -15.0))
        world._pz_time_penalty = float(getattr(args, "time_penalty", -0.01))
        world._pz_vel_penalty = float(getattr(args, "vel_penalty", -0.001))
        world._pz_distance_reward = float(getattr(args, "distance_reward", 0.03))
        
        # Stage-1 extras (formation/threat shaping)
        world._pz_formation_reward = float(getattr(args, "formation_reward", 0.08))
        world._pz_threat_weight_reward = float(getattr(args, "threat_weight_reward", 0.02))
        # Intercept encouragement (defender dense term)
        world._pz_intercept_reward = float(getattr(args, "intercept_reward", 0.05))
        # Constant proxy for position covariance logdet (align S1/S2 defender obs length)
        world._pz_logdet_const = float(np.log(max(1e-6, (world._pz_r ** 2))))

        # Counters & flags
        world._pz_captured_ids = set()
        world._pz_zone_entry = False
        world._pz_total_captured = 0
        world._pz_total_entries = 0
        world._pz_episode_success = False
        
        # Threat weights (placeholder for Stage-1 shaping)
        world._pz_threat_weights = []       # 每个进攻机的威胁权重

        # ========== 创建智能体 ==========
        M = int(getattr(args, "num_defenders", 5))      # 防守机数量
        N = int(getattr(args, "num_intruders", 2))      # 进攻机数量
        world.agents = []
        self.defenders = []
        self.intruders = []

        # 创建防守机
        for i in range(M):
            agent = Agent()
            agent.name = f"defender_{i}"
            agent.adversary = False
            agent.silent = True
            agent.movable = True
            agent.collide = True  # 启用碰撞检测，避免重合
            agent.u_range = 1.0
            agent.accel = accel
            agent.max_speed = def_spd
            agent.size = 0.06  # 稍微增大防守机，便于碰撞检测
            agent.color = np.array([0.25, 0.75, 0.25])  # 绿色 - 防守方
            world.agents.append(agent)
            self.defenders.append(agent)

        # 创建进攻机
        for j in range(N):
            agent = Agent()
            agent.name = f"intruder_{j}"
            agent.adversary = True
            agent.silent = True
            agent.movable = True  # 明确设置为可移动
            agent.collide = True  # 启用碰撞检测，避免重合
            agent.u_range = 1.0
            agent.accel = accel
            agent.max_speed = intr_spd
            agent.size = 0.05  # 进攻机保持标准大小
            agent.color = np.array([0.25, 0.25, 0.75])  # 蓝色 - 进攻方
            world.agents.append(agent)
            self.intruders.append(agent)

        # ========== 创建边界墙 ==========
        R = world._pz_world_r
        wall_width = 0.05
        world.walls = [
            Wall(orient='H', axis_pos=+R, endpoints=(-R, R), width=wall_width, hard=True),  # 上
            Wall(orient='H', axis_pos=-R, endpoints=(-R, R), width=wall_width, hard=True),  # 下
            Wall(orient='V', axis_pos=-R, endpoints=(-R, R), width=wall_width, hard=True),  # 左
            Wall(orient='V', axis_pos=+R, endpoints=(-R, R), width=wall_width, hard=True),  # 右
        ]

        # ========== 创建保护区地标（用于可视化） ==========
        landmark = Landmark()
        landmark.name = "protected_zone"
        landmark.collide = False
        landmark.movable = False
        landmark.size = world._pz_r
        landmark.boundary = False
        landmark.color = np.array([0.75, 0.25, 0.25])  # 红色保护区
        landmark.state.p_pos = world._pz_center.copy()
        landmark.state.p_vel = np.zeros(2, dtype=np.float32)
        world.landmarks = [landmark]

        # 初始化世界
        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        """重置世界状态"""
        # ========== 重置计数器 ==========
        world.world_step = 0
        world._pz_captured_ids.clear()
        world._pz_zone_entry = False
        world._pz_zone_entered = False  # 重置进区标志
        world._pz_total_captured = 0
        world._pz_total_entries = 0
        world._pz_episode_success = False
        
        # ========== 重置抓捕状态（修复episode长度为1的问题）==========
        if hasattr(world, '_pz_already_captured'):
            world._pz_already_captured.clear()
        else:
            world._pz_already_captured = set()
        
        # ========== Stage1新增：重置威胁权重状态 ==========
        world._pz_threat_weights = [0.0] * len(self.intruders)

        # ========== 重置智能体状态 ==========
        # 防守机：均匀分布在保护区边界上
        defender_count = len(self.defenders)
        if defender_count > 0:
            angles = np.linspace(0, 2 * np.pi, defender_count, endpoint=False)
            for i, defender in enumerate(self.defenders):
                angle = angles[i] + np.random.uniform(-0.2, 0.2)  # 添加小幅随机扰动
                pos_x = world._pz_center[0] + world._pz_r * np.cos(angle)
                pos_y = world._pz_center[1] + world._pz_r * np.sin(angle)
                defender.state.p_pos = np.array([pos_x, pos_y], dtype=np.float32)
                defender.state.p_vel = np.zeros(2, dtype=np.float32)

        # 进攻机：随机分布在世界边界上，面向保护区
        for intruder in self.intruders:
            # 重置可移动状态（清除之前被捕获的冻结状态）
            intruder.movable = True
            
            # 在世界边界上随机选择位置
            angle = np.random.uniform(0, 2 * np.pi)
            pos_x = world._pz_center[0] + world._pz_world_r * np.cos(angle)
            pos_y = world._pz_center[1] + world._pz_world_r * np.sin(angle)
            intruder.state.p_pos = np.array([pos_x, pos_y], dtype=np.float32)
            
            # 初始速度：朝向保护区中心，带有随机扰动
            to_center = world._pz_center - intruder.state.p_pos
            to_center_norm = np.linalg.norm(to_center)
            if to_center_norm > 0:
                to_center = to_center / to_center_norm
                # 添加随机扰动
                angle_noise = np.random.uniform(-0.3, 0.3)
                cos_noise, sin_noise = np.cos(angle_noise), np.sin(angle_noise)
                to_center = np.array([
                    to_center[0] * cos_noise - to_center[1] * sin_noise,
                    to_center[0] * sin_noise + to_center[1] * cos_noise
                ], dtype=np.float32)
            
            init_speed = np.random.uniform(0.3, 0.7) * intruder.max_speed
            intruder.state.p_vel = to_center * init_speed

    def reward(self, agent: Agent, world: World):
        """计算智能体奖励 - Stage1增强版：添加威胁权重系统"""
        
        # ========== 每步只计算一次的全局事件 ==========
        if not hasattr(world, '_pz_events_computed') or world._pz_events_computed != world.world_step:
            self._compute_step_events(world)
            self._compute_threat_weights(world)  # Stage1新增
            world._pz_events_computed = world.world_step

        if agent.adversary:
            # ========== 进攻方奖励 ==========
            return self._intruder_reward(agent, world)
        else:
            # ========== 防守方奖励 ==========
            return self._defender_reward(agent, world)

    def _compute_step_events(self, world: World):
        """计算本步的全局事件（抓捕、进区等）"""
        world._pz_captured_ids.clear()
        world._pz_zone_entry = False

        # ========== 检查抓捕事件 ==========
        for j, intruder in enumerate(self.intruders):
            # 检查这架进攻机是否已经被抓捕过（避免重复计数）
            if j in getattr(world, '_pz_already_captured', set()):
                # 确保被捕获的智能体保持静止状态
                intruder.state.p_vel = np.zeros(2)
                intruder.movable = False
                continue
                
            for defender in self.defenders:
                distance = np.linalg.norm(intruder.state.p_pos - defender.state.p_pos)
                if distance <= world._pz_capture:
                    world._pz_captured_ids.add(j)
                    # 标记为永久被捕，避免重复计数
                    if not hasattr(world, '_pz_already_captured'):
                        world._pz_already_captured = set()
                    world._pz_already_captured.add(j)
                    world._pz_total_captured += 1
                    
                    # 立即停止被捕获的智能体
                    intruder.state.p_vel = np.zeros(2)
                    intruder.movable = False
                    break  # 一个进攻机只能被一个防守机抓捕

        # ========== 检查进区事件 ==========
        for intruder in self.intruders:
            distance_to_center = np.linalg.norm(intruder.state.p_pos - world._pz_center)
            if distance_to_center <= world._pz_r:
                world._pz_zone_entry = True
                # 只在首次进区时计数，避免重复计数
                if not getattr(world, '_pz_zone_entered', False):
                    world._pz_total_entries += 1
                    world._pz_zone_entered = True
                break  # 有一个进入即可

        # ========== 检查回合成功条件 ==========
        # 使用永久抓捕状态而不是当前步抓捕
        already_captured = getattr(world, '_pz_already_captured', set())
        if len(already_captured) >= len(self.intruders) and not world._pz_zone_entry:
            world._pz_episode_success = True

    def _compute_threat_weights(self, world: World):
        """Stage1修复版：只计算活跃进攻机的威胁权重"""
        world._pz_threat_weights = []
        already_captured = getattr(world, '_pz_already_captured', set())
        
        for i, intruder in enumerate(self.intruders):
            # 被捕获的进攻机威胁权重为0，不参与后续计算
            if i in already_captured:
                world._pz_threat_weights.append(0.0)
                continue
            
            # 只有活跃进攻机才计算威胁权重
            distance_to_zone = np.linalg.norm(intruder.state.p_pos - world._pz_center) - world._pz_r
            distance_to_zone = max(distance_to_zone, 0.1)  # 避免除零和数值爆炸
            # 使用更稳定的距离威胁计算
            distance_threat = 1.0 / (1.0 + distance_to_zone)  # 范围: (0, 1]
            
            # 2. 速度威胁（越快威胁越大）
            speed = np.linalg.norm(intruder.state.p_vel)
            speed_threat = speed / intruder.max_speed if intruder.max_speed > 0 else 0.0
            
            # 3. 方向威胁（朝向保护区威胁更大）
            to_center = world._pz_center - intruder.state.p_pos
            to_center_norm = np.linalg.norm(to_center)
            if to_center_norm > 1e-6 and speed > 1e-6:
                to_center_unit = to_center / to_center_norm
                vel_unit = intruder.state.p_vel / speed
                direction_alignment = np.dot(to_center_unit, vel_unit)
                direction_threat = max(0.0, direction_alignment)  # 0到1，朝向保护区为正
            else:
                direction_threat = 0.0
            
            # 综合威胁权重
            total_threat = distance_threat * (1.0 + speed_threat + 0.5 * direction_threat)
            world._pz_threat_weights.append(total_threat)

    def _defender_reward(self, agent: Agent, world: World):
        """防守方奖励计算 - 修复版：避免奖励重复和爆炸"""
        reward = 0.0

        # ========== 修复：个人抓捕奖励（只有实际抓捕者获得） ==========
        personal_capture_reward = 0.0
        agent_pos = agent.state.p_pos
        
        for j in world._pz_captured_ids:
            intruder = self.intruders[j]
            distance = np.linalg.norm(agent_pos - intruder.state.p_pos)
            if distance <= world._pz_capture:
                # 只有距离最近的防守机获得抓捕奖励
                closest_defender = True
                for other_defender in self.defenders:
                    if other_defender != agent:
                        other_distance = np.linalg.norm(other_defender.state.p_pos - intruder.state.p_pos)
                        if other_distance < distance:
                            closest_defender = False
                            break
                
                if closest_defender:
                    personal_capture_reward += world._pz_capture_reward
        
        reward += personal_capture_reward

        # ========== 团队合作奖励：所有防守机共享小额奖励 ==========
        team_bonus = len(world._pz_captured_ids) * world._pz_capture_reward * 0.1  # 10%团队奖励
        reward += team_bonus

        # ========== 进区惩罚 ==========
        if world._pz_zone_entry:
            reward += world._pz_entry_penalty

        # ========== 主要奖励：防守机间最优间距（合并spacing和cooperation） ==========
        optimal_spacing_reward = self._compute_optimal_spacing_reward(agent, world)
        reward += optimal_spacing_reward

        # ========== 威胁感知奖励：优先关注最高威胁目标 ==========
        threat_focus_reward = self._compute_threat_focus_reward(agent, world)
        reward += threat_focus_reward

        # ========== 最优夹角布阵奖励 ==========
        formation_reward = self._compute_formation_reward(agent, world)
        reward += formation_reward

        # ========== 拦截鼓励：靠近所有未捕获的进攻机（按威胁加权） ==========
        intercept_bonus = self._compute_intercept_reward(agent, world)
        reward += intercept_bonus

        # ========== 基础惩罚 ==========
        reward += world._pz_time_penalty
        velocity_magnitude = np.linalg.norm(agent.state.p_vel)
        reward += world._pz_vel_penalty * velocity_magnitude

        return float(reward)

    def _compute_intercept_reward(self, agent: Agent, world: World):
        """Encourage defenders to approach intruders (threat-weighted closeness)."""
        # Only for defenders
        if agent.adversary:
            return 0.0
        M = len(self.defenders)
        N = len(self.intruders)
        if N == 0:
            return 0.0
        # Ensure threat weights exist
        try:
            _ = world._pz_threat_weights
        except AttributeError:
            world._pz_threat_weights = [0.0] * N
        # Compute closeness to all active intruders, weighted by threat
        s = agent.state.p_pos
        R = max(world._pz_world_r, 1e-6)
        already_captured = getattr(world, '_pz_already_captured', set())
        total = 0.0
        for i, intr in enumerate(self.intruders):
            if i in already_captured:
                continue
            d = np.linalg.norm(s - intr.state.p_pos)
            closeness = max(0.0, (R - d) / R)  # in [0,1]
            Ti = 0.0
            if i < len(world._pz_threat_weights):
                Ti = float(world._pz_threat_weights[i])
            total += Ti * closeness
        return world._pz_intercept_reward * total

    def _intruder_reward(self, agent: Agent, world: World):
        """进攻方奖励计算 - 优化版：增加防碰撞机制"""
        reward = 0.0

        # ========== 获取当前进攻机的索引 ==========
        try:
            agent_idx = self.intruders.index(agent)
        except ValueError:
            agent_idx = -1

        # ========== 检查是否已被抓捕，如果是则不给任何奖励 ==========
        already_captured = getattr(world, '_pz_already_captured', set())
        if agent_idx in already_captured:
            return 0.0  # 被捕获后不再获得任何奖励

        # ========== 进区奖励 ==========
        distance_to_center = np.linalg.norm(agent.state.p_pos - world._pz_center)
        if distance_to_center <= world._pz_r:
            reward += abs(world._pz_entry_penalty)  # 成功进区的大奖励

        # ========== 接近保护区奖励（强化朝向目标的行为） ==========
        zone_approach_threshold = world._pz_r + 2.5  # 优化奖励范围
        if distance_to_center <= zone_approach_threshold:
            progress_ratio = (zone_approach_threshold - distance_to_center) / zone_approach_threshold
            approach_reward = world._pz_distance_reward * progress_ratio * 2.0  # 降低系数，避免过度奖励
            reward += approach_reward
            
        # ========== 方向奖励：强化朝向保护区移动 ==========
        to_center = world._pz_center - agent.state.p_pos
        to_center_norm = np.linalg.norm(to_center)
        agent_speed = np.linalg.norm(agent.state.p_vel)
        
        if to_center_norm > 1e-6 and agent_speed > 1e-6:
            to_center_unit = to_center / to_center_norm
            vel_unit = agent.state.p_vel / agent_speed
            direction_alignment = np.dot(to_center_unit, vel_unit)
            if direction_alignment > 0:
                direction_reward = world._pz_distance_reward * 1.0 * direction_alignment  # 增强方向奖励
                reward += direction_reward

        # ========== 进攻机间防碰撞奖励 ==========
        intruder_spacing_reward = self._compute_intruder_spacing_reward(agent, world)
        reward += intruder_spacing_reward

        # ========== 避开防守机奖励（适度降低权重） ==========
        if len(self.defenders) > 0:
            min_defender_distance = float('inf')
            for defender in self.defenders:
                distance = np.linalg.norm(agent.state.p_pos - defender.state.p_pos)
                min_defender_distance = min(min_defender_distance, distance)
            
            # 更温和的避开奖励
            avoidance_reward = world._pz_distance_reward * 0.01 * min_defender_distance  # 进一步降低权重
            reward += avoidance_reward

        # ========== 基础惩罚 ==========
        reward += world._pz_time_penalty
        velocity_magnitude = np.linalg.norm(agent.state.p_vel)
        reward += world._pz_vel_penalty * velocity_magnitude

        return float(reward)

    def observation(self, agent: Agent, world: World):
        """构建智能体观测 - 优化版：区分活跃和被捕获智能体"""
        obs_list = []
        already_captured = getattr(world, '_pz_already_captured', set())

        if agent.adversary:
            # ========== 进攻方观测 ==========
            # 1. 自身状态
            obs_list.extend(agent.state.p_pos.tolist())  # 自身位置
            obs_list.extend(agent.state.p_vel.tolist())  # 自身速度

            # 2. 到保护区中心的相对位置
            to_center = world._pz_center - agent.state.p_pos
            obs_list.extend(to_center.tolist())

            # 3. 所有防守机的相对位置和速度
            for defender in self.defenders:
                relative_pos = defender.state.p_pos - agent.state.p_pos
                obs_list.extend(relative_pos.tolist())
                obs_list.extend(defender.state.p_vel.tolist())

            # 4. 其他进攻机的相对位置和速度（包含活跃状态标记）
            for i, other_intruder in enumerate(self.intruders):
                if other_intruder != agent:
                    relative_pos = other_intruder.state.p_pos - agent.state.p_pos
                    obs_list.extend(relative_pos.tolist())
                    # 如果被捕获，速度信息为0
                    if i in already_captured:
                        obs_list.extend([0.0, 0.0])  # 被捕获的进攻机速度为0
                    else:
                        obs_list.extend(other_intruder.state.p_vel.tolist())

        else:
            # Defender observation (aligned with Stage-2 length)
            # 1) self state
            obs_list.extend(agent.state.p_pos.tolist())  # 2
            obs_list.extend(agent.state.p_vel.tolist())  # 2
            # 2) vector to protected center
            to_center = world._pz_center - agent.state.p_pos
            obs_list.extend(to_center.tolist())          # 2  => base 6

            # Ensure threat weights are available (used below)
            if not hasattr(world, '_pz_threat_weights'):
                world._pz_threat_weights = [0.0] * len(self.intruders)

            # 3) per-intruder block: rel pos(2) + vel(2) + proxy_logdet(1) + threat T(1) + captured(1) => 7
            for i, intruder in enumerate(self.intruders):
                relative_pos = intruder.state.p_pos - agent.state.p_pos
                obs_list.extend(relative_pos.tolist())
                if i in already_captured:
                    obs_list.extend([0.0, 0.0])
                else:
                    obs_list.extend(intruder.state.p_vel.tolist())
                obs_list.append(float(getattr(world, '_pz_logdet_const', 0.0)))
                Ti = 0.0
                if i < len(world._pz_threat_weights):
                    Ti = float(world._pz_threat_weights[i])
                obs_list.append(Ti)
                obs_list.append(1.0 if i in already_captured else 0.0)

            # 4) other defenders: rel pos(2) + vel(2)
            for other_defender in self.defenders:
                if other_defender is agent:
                    continue
                relative_pos = other_defender.state.p_pos - agent.state.p_pos
                obs_list.extend(relative_pos.tolist())
                obs_list.extend(other_defender.state.p_vel.tolist())

        # Cast to numpy, sanitize, and align to S2 defender-sized template length
        obs = np.array(obs_list, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        # Defender-sized template length in S2: 6 + 7*N + 4*(M-1)
        N = len(self.intruders)
        M = len(self.defenders)
        target_len = 6 + 7 * N + 4 * max(M - 1, 0)
        if obs.shape[0] < target_len:
            pad = np.zeros(target_len - obs.shape[0], dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=0)
        elif obs.shape[0] > target_len:
            obs = obs[:target_len]
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def done(self, agent: Agent, world: World):
        """判断回合是否结束"""
        # 时间限制
        if world.world_step >= world.world_length:
            return True
        
        # 有进攻机进入保护区（任意一架进入即结束）
        if world._pz_zone_entry:
            return True
        
        # 所有进攻机都被抓捕（使用永久抓捕状态）
        already_captured = getattr(world, '_pz_already_captured', set())
        if len(already_captured) >= len(self.intruders):
            return True
        
        return False

    def info(self, agent: Agent, world: World):
        """返回训练统计信息"""
        # 防守方和进攻方都返回各自的信息
        info_dict = {}
        
        if agent.adversary:
            # ========== 进攻方统计信息 ==========
            agent_idx = (
                self.intruders.index(agent)
                if agent in self.intruders else -1
            )
            
            # 当前回合进攻方奖励（单独记录）
            info_dict[f"intruder_{agent_idx}_reward"] = float(
                self._intruder_reward(agent, world)
            )
            
            # 进攻成功指标
            distance_to_center = np.linalg.norm(
                agent.state.p_pos - world._pz_center
            )
            info_dict[f"intruder_{agent_idx}_zone_distance"] = float(
                max(distance_to_center - world._pz_r, 0.0)
            )
            
            # 被抓捕状态
            already_captured = getattr(world, '_pz_already_captured', set())
            info_dict[f"intruder_{agent_idx}_captured"] = float(
                1 if agent_idx in already_captured else 0
            )
            
            # 本步是否成功突破
            breakthrough = (
                world._pz_zone_entry and distance_to_center <= world._pz_r
            )
            info_dict[f"intruder_{agent_idx}_breakthrough"] = float(
                1 if breakthrough else 0
            )
            
        else:
            # ========== 防守方统计信息 ==========
            agent_idx = (
                self.defenders.index(agent)
                if agent in self.defenders else -1
            )
            
            # 当前回合防守方奖励（单独记录）
            info_dict[f"defender_{agent_idx}_reward"] = float(
                self._defender_reward(agent, world)
            )
            
            # 防守效率指标
            if len(self.intruders) > 0:
                min_distance_to_intruder = float('inf')
                for intruder in self.intruders:
                    distance = np.linalg.norm(
                        agent.state.p_pos - intruder.state.p_pos
                    )
                    min_distance_to_intruder = min(
                        min_distance_to_intruder, distance
                    )
                info_dict[f"defender_{agent_idx}_min_intruder_distance"] = (
                    float(min_distance_to_intruder)
                )
            
            # 只有第一个防守机返回全局统计，避免重复
            if agent.name.endswith("_0"):
                # ========== 简化的episode级别统计 ==========
                already_captured = getattr(
                    world, '_pz_already_captured', set()
                )
                
                # 简单的0/1标记（Runner会计算滑动平均）
                defense_success = 1 if (
                    len(already_captured) >= len(self.intruders) and
                    not world._pz_zone_entry
                ) else 0
                
                attack_success = 1 if world._pz_zone_entry else 0
                
                info_dict.update({
                    # episode结果标记 (0或1)
                    "defense_success": float(defense_success),
                    "attack_success": float(attack_success),
                    
                    # 有用的训练指标
                    "episode_length": float(world.world_step),
                    "total_captured": float(len(already_captured)),
                    "total_intruders": float(len(self.intruders)),
                })

        # ========== 距离统计 ==========
        if len(self.intruders) > 0 and len(self.defenders) > 0:
            # 计算平均防守-进攻距离
            total_distance = 0.0
            count = 0
            for defender in self.defenders:
                for intruder in self.intruders:
                    distance = np.linalg.norm(
                        defender.state.p_pos - intruder.state.p_pos
                    )
                    total_distance += distance
                    count += 1
            
            if count > 0:
                info_dict["avg_defender_intruder_distance"] = float(
                    total_distance / count
                )

            # 计算进攻机到保护区的最近距离
            min_distance_to_zone = float('inf')
            for intruder in self.intruders:
                distance = (
                    np.linalg.norm(intruder.state.p_pos - world._pz_center) -
                    world._pz_r
                )
                min_distance_to_zone = min(
                    min_distance_to_zone, max(distance, 0.0)
                )
            
            info_dict["min_intruder_to_zone_distance"] = float(
                min_distance_to_zone
            )

        return info_dict

    def _compute_formation_reward(self, agent: Agent, world: World):
        """计算防共线奖励 - 优化版：只考虑活跃进攻机"""
        if len(self.defenders) < 2:
            return 0.0
        
        formation_reward = 0.0
        agent_pos = agent.state.p_pos
        already_captured = getattr(world, '_pz_already_captured', set())
        
        # 只对活跃进攻机计算防共线奖励
        active_intruders = [
            intruder for i, intruder in enumerate(self.intruders) 
            if i not in already_captured
        ]
        
        if len(active_intruders) == 0:
            return 0.0
        
        # 对每个活跃进攻机计算防共线奖励
        for intruder in active_intruders:
            target_pos = intruder.state.p_pos
            
            # 与其他防守机形成的角度分集
            for other_defender in self.defenders:
                if other_defender == agent:
                    continue
                    
                vec1 = agent_pos - target_pos
                vec2 = other_defender.state.p_pos - target_pos
                
                if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_diff = np.arccos(cos_angle)
                    
                    # 奖励90度角差（最优AoA几何）
                    optimal_angle = np.pi / 2
                    angle_score = 1.0 - abs(angle_diff - optimal_angle) / optimal_angle
                    formation_reward += max(0.0, angle_score)
        
        # 归一化
        num_pairs = len(self.defenders) - 1
        num_targets = len(active_intruders)
        if num_pairs > 0 and num_targets > 0:
            formation_reward /= (num_pairs * num_targets)
        
        return world._pz_formation_reward * formation_reward

    def _compute_optimal_spacing_reward(self, agent: Agent, world: World):
        """计算最优间距奖励 - 防止重合并维持理想协作距离"""
        if len(self.defenders) < 2:
            return 0.0
        
        spacing_reward = 0.0
        agent_pos = agent.state.p_pos
        
        # 强防碰撞机制
        min_safety_distance = 0.25  # 最小安全距离
        ideal_distance = world._pz_r * 1.2  # 理想协作距离
        
        distances_to_others = []
        for other_defender in self.defenders:
            if other_defender != agent:
                distance = np.linalg.norm(agent_pos - other_defender.state.p_pos)
                distances_to_others.append(distance)
                
                # 强烈惩罚过近距离（防碰撞）
                if distance < min_safety_distance:
                    collision_penalty = -world._pz_distance_reward * 5.0 * (min_safety_distance - distance) / min_safety_distance
                    spacing_reward += collision_penalty
                # 奖励理想距离
                elif distance < ideal_distance * 1.5:
                    distance_score = 1.0 - abs(distance - ideal_distance) / ideal_distance
                    spacing_bonus = world._pz_distance_reward * 0.5 * max(0.0, distance_score)
                    spacing_reward += spacing_bonus
        
        # 奖励均匀分布
        if len(distances_to_others) > 1:
            distance_std = np.std(distances_to_others)
            uniformity_bonus = world._pz_distance_reward * 0.2 / (1.0 + distance_std)
            spacing_reward += uniformity_bonus
            
        return spacing_reward

    def _compute_threat_focus_reward(self, agent: Agent, world: World):
        """计算威胁专注奖励 - 优先关注最高威胁目标"""
        if len(world._pz_threat_weights) == 0:
            return 0.0
        
        max_threat_value = max(world._pz_threat_weights)
        if max_threat_value <= 0.0:
            return 0.0
            
        max_threat_idx = np.argmax(world._pz_threat_weights)
        max_threat_intruder = self.intruders[max_threat_idx]
        
        # 计算到最高威胁目标的距离
        distance = np.linalg.norm(agent.state.p_pos - max_threat_intruder.state.p_pos)
        
        # 奖励接近最高威胁目标（不需要是最近的）
        max_distance = 2 * world._pz_world_r
        closeness_score = (max_distance - distance) / max_distance
        threat_focus_reward = world._pz_threat_weight_reward * closeness_score * max_threat_value
        
        return threat_focus_reward

    def _compute_intruder_spacing_reward(self, agent: Agent, world: World):
        """计算进攻机间距奖励 - 防止进攻机相撞"""
        if len(self.intruders) < 2:
            return 0.0
        
        spacing_reward = 0.0
        agent_pos = agent.state.p_pos
        min_safe_distance = 0.2  # 进攻机最小安全距离
        
        for other_intruder in self.intruders:
            if other_intruder != agent:
                distance = np.linalg.norm(agent_pos - other_intruder.state.p_pos)
                
                if distance < min_safe_distance:
                    # 强烈惩罚进攻机间的碰撞
                    collision_penalty = -world._pz_distance_reward * 3.0 * (min_safe_distance - distance) / min_safe_distance
                    spacing_reward += collision_penalty
                elif distance < min_safe_distance * 2:
                    # 奖励合理距离
                    spacing_bonus = world._pz_distance_reward * 0.1 * (distance - min_safe_distance) / min_safe_distance
                    spacing_reward += spacing_bonus
                    
        return spacing_reward

    def post_step_callback(self, world: World):
        """每步后处理：完全冻结被抓捕的进攻机"""
        already_captured = getattr(world, '_pz_already_captured', set())
        
        # 完全冻结被抓捕的进攻机
        for i, intruder in enumerate(self.intruders):
            if i in already_captured:
                # 1. 清零动作（防止新的动作输入）
                if hasattr(intruder, 'action') and hasattr(intruder.action, 'u'):
                    intruder.action.u = np.zeros(2, dtype=np.float32)
                
                # 2. 清零速度（停止移动）
                intruder.state.p_vel = np.zeros(2, dtype=np.float32)
                
                # 3. 强制设置为不可移动（物理层面阻止移动）
                intruder.movable = False
