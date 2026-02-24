"""RetroRacer Gymnasium environment with physics and rendering."""

import math
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from racing.track import Track, make_track, load_track_json, generate_random_track


# -------------------------
# Utility math
# -------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def lerp(a, b, t):
    return a + (b - a) * t


# -------------------------
# Track geometry helpers
# -------------------------
def closest_on_segment(p, a, b):
    ap = p - a
    ab = b - a
    denom = float(ab[0]*ab[0] + ab[1]*ab[1])
    if denom <= 1e-8:
        return a.copy(), 0.0
    t = float((ap[0]*ab[0] + ap[1]*ab[1]) / denom)
    t = clamp(t, 0.0, 1.0)
    proj = a + t * ab
    return proj, t


def find_nearest_track_point(track: Track, p: np.ndarray):
    """
    Finds nearest projection of p onto the polyline segments.
    Returns:
      i: segment start index
      proj: nearest point
      t: segment param
      dist: euclidean distance to centerline
      s: progress distance along track in [0,total)
      tangent: unit direction along segment at proj
    """
    pts = track.pts
    best_d2 = 1e18
    best_i = 0
    best_proj = None
    best_t = 0.0

    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i+1) % len(pts)]
        proj, t = closest_on_segment(p, a, b)
        d2 = float(np.sum((p - proj) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_i, best_proj, best_t = i, proj, t

    dist = math.sqrt(best_d2)
    s = float(track.cum_len[best_i] + best_t * track.seg_len[best_i])
    a = track.pts[best_i]
    b = track.pts[(best_i+1) % len(track.pts)]
    tang = b - a
    tang_norm = float(np.linalg.norm(tang)) + 1e-8
    tangent = tang / tang_norm
    return best_i, best_proj, best_t, dist, s, tangent


# -------------------------
# Retro Racer Gym Env
# -------------------------
class RetroRacerEnv(gym.Env):
    """
    Continuous control top-down racer.

    Observation (float32 vector):
      [0] speed_norm
      [1] sin(heading_error)
      [2] cos(heading_error)
      [3] cross_track_error_norm (signed)
      [4..] lookahead heading errors (sin/cos pairs) for k waypoints ahead
    Action:
      steering in [-1,1]
      throttle in [0,1]
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        seed=0,
        track_seed=123,
        lookahead=6,
        dt=1/30,
        max_steps=2000,
        track_half_width=42.0,
        world_scale=1.0,
        track_file=None,
        randomize_track=False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.lookahead = int(lookahead)
        self.randomize_track = randomize_track
        self._track_half_width = track_half_width

        if track_file and os.path.exists(track_file):
            ctrl_pts, hw = load_track_json(track_file)
            self.track = make_track(control_pts=ctrl_pts, half_width=hw)
        elif randomize_track:
            self.track = generate_random_track(half_width=track_half_width)
        else:
            self.track = make_track(half_width=track_half_width)

        # Car state
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = 0.0
        self.heading = 0.0
        self.step_count = 0

        # For progress reward
        self.prev_s = 0.0
        self.lap_progress = 0.0

        # Action space: steer [-1,1], throttle [-1,1] (negative = brake)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # obs: base(4) + lookahead heading(2*k) + lookahead curvature(k)
        obs_dim = 4 + 3 * self.lookahead
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Physics params
        self.max_speed = 380.0
        self.accel = 300.0          # peak acceleration (at low speed)
        self.brake_decel = 250.0    # braking force
        self.drag = 0.5             # aerodynamic drag coefficient
        self.engine_brake = 40.0    # deceleration when coasting (no throttle)
        self.steer_rate = 3.5
        self.steer_speed_scale = 0.55
        self.offtrack_grace = 6
        self.offtrack_counter = 0

        # Rendering
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None
        self.opponents = []  # list of (pos, heading, lap_progress) for race mode
        self._delay_flip = False  # when True, render() skips display.flip()

        self._rng = np.random.default_rng(seed)

    def _reset_car_at_start(self):
        p0 = self.track.pts[0].copy()
        p1 = self.track.pts[1].copy()
        tang = p1 - p0
        ang = math.atan2(float(tang[1]), float(tang[0]))
        self.pos = p0.copy()
        self.heading = ang
        self.vel = 0.0
        self.step_count = 0
        self.offtrack_counter = 0

        _, _, _, _, s, _ = find_nearest_track_point(self.track, self.pos)
        self.prev_s = s
        self.lap_progress = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a new random track each episode for domain randomization
        if self.randomize_track:
            self.track = generate_random_track(half_width=self._track_half_width)
        self._reset_car_at_start()
        obs = self._get_obs()
        info = {"lap_progress": self.lap_progress}
        return obs, info

    def _get_obs(self):
        i, proj, t, dist, s, tangent = find_nearest_track_point(self.track, self.pos)

        perp = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        sign = 1.0 if float(np.dot(self.pos - proj, perp)) >= 0.0 else -1.0
        cte = sign * dist

        track_ang = math.atan2(float(tangent[1]), float(tangent[0]))
        heading_err = wrap_angle(track_ang - float(self.heading))

        speed_norm = float(self.vel / self.max_speed)
        cte_norm = float(cte / self.track.half_width)

        feats = [
            speed_norm,
            math.sin(heading_err),
            math.cos(heading_err),
            cte_norm,
        ]

        # Lookahead: heading errors + curvature
        N = len(self.track.pts)
        curvatures = []
        prev_ang = track_ang
        for k in range(1, self.lookahead + 1):
            idx = (i + k * 3) % N
            a = self.track.pts[idx]
            b = self.track.pts[(idx + 1) % N]
            tang2 = b - a
            tang2 = tang2 / (float(np.linalg.norm(tang2)) + 1e-8)
            ang2 = math.atan2(float(tang2[1]), float(tang2[0]))

            # Heading error relative to car
            herr2 = wrap_angle(ang2 - float(self.heading))
            feats.append(math.sin(herr2))
            feats.append(math.cos(herr2))

            # Curvature: angular change between consecutive lookahead tangents
            curv = wrap_angle(ang2 - prev_ang)
            curvatures.append(curv)
            prev_ang = ang2

        # Append curvatures (scaled — high values = sharp turn ahead)
        for c in curvatures:
            feats.append(float(c / math.pi))  # normalized to [-1, 1]

        # Cache max upcoming curvature for reward calculation
        self._upcoming_curvature = max(abs(c) for c in curvatures) if curvatures else 0.0

        return np.array(feats, dtype=np.float32)

    def step(self, action):
        steer = float(action[0])
        throttle = float(action[1])
        steer = clamp(steer, -1.0, 1.0)
        throttle = clamp(throttle, -1.0, 1.0)

        dt = self.dt
        is_braking = throttle < -0.05

        # Non-linear acceleration: strong at low speed, weaker at high speed
        speed_ratio = self.vel / (self.max_speed + 1e-8)
        accel_curve = 1.0 - 0.5 * speed_ratio

        if throttle > 0.15:
            # Accelerating
            net_accel = throttle * self.accel * accel_curve - self.drag * self.vel * speed_ratio
        elif throttle < -0.05:
            # Active braking (DOWN key / negative throttle)
            brake_force = abs(throttle) * self.brake_decel
            net_accel = -(brake_force + self.drag * self.vel * speed_ratio)
        elif throttle < 0.05:
            # Coasting: engine braking + drag
            net_accel = -(self.engine_brake + self.drag * self.vel * speed_ratio)
        else:
            # Light throttle
            net_accel = throttle * self.accel * accel_curve * 0.3 - self.drag * self.vel * speed_ratio

        self.vel += net_accel * dt
        self.vel = clamp(self.vel, 0.0, self.max_speed)

        speed_factor = clamp(self.vel / (self.max_speed * self.steer_speed_scale + 1e-8), 0.0, 1.0)

        # Understeer: steering becomes less effective at high speed
        speed_pct = self.vel / (self.max_speed + 1e-8)
        grip_factor = 1.0 / (1.0 + 2.5 * speed_pct * speed_pct)

        # Brake-turn lockup: braking + steering = loss of grip
        # Simulates tire grip circle — can't brake and turn at full grip
        if is_braking and abs(steer) > 0.2:
            brake_intensity = abs(throttle)  # 0-1
            steer_intensity = abs(steer)     # 0-1
            lockup = brake_intensity * steer_intensity  # 0-1
            grip_factor *= max(0.15, 1.0 - 0.8 * lockup)  # up to 80% grip loss

        turn_rate = steer * self.steer_rate * (0.25 + 0.75 * speed_factor) * grip_factor
        self.heading += turn_rate * dt
        self.heading = wrap_angle(self.heading)

        forward = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        self.pos += forward * float(self.vel * dt)

        self.step_count += 1

        _, proj, _, dist, s, tangent = find_nearest_track_point(self.track, self.pos)

        ds = s - self.prev_s
        if ds < -0.5 * self.track.total:
            ds += self.track.total
        elif ds > 0.5 * self.track.total:
            ds -= self.track.total
        self.prev_s = s
        self.lap_progress += ds

        perp = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        sign = 1.0 if float(np.dot(self.pos - proj, perp)) >= 0.0 else -1.0
        cte = sign * dist

        track_ang = math.atan2(float(tangent[1]), float(tangent[0]))
        heading_err = wrap_angle(track_ang - float(self.heading))

        on_track = dist <= self.track.half_width

        if on_track:
            self.offtrack_counter = 0
        else:
            self.offtrack_counter += 1

        # Reward: rebalanced for high-speed braking physics
        speed_norm = self.vel / self.max_speed
        cte_norm = abs(cte) / self.track.half_width
        heading_norm = abs(heading_err) / math.pi

        reward = 0.0
        reward += 1.2 * (ds / (self.max_speed * self.dt + 1e-8))  # progress
        reward += 0.15 * speed_norm                                # encourage speed
        reward -= 0.5 * cte_norm                                   # stay centered
        reward -= 0.4 * heading_norm                                # face the right way
        reward -= 0.1 * (steer * steer)                            # smooth steering

        # Idle penalty: sitting still is NOT an option
        if speed_norm < 0.15:
            reward -= 0.3  # constant drain for going too slow

        # Curvature-aware speed penalty: HARSH punishment for going fast into turns
        upcoming_curv = getattr(self, '_upcoming_curvature', 0.0)
        curv_norm = min(upcoming_curv / 0.4, 1.0)  # 0=straight, 1=very sharp
        if curv_norm > 0.1:
            reward -= 1.0 * curv_norm * speed_norm * speed_norm  # quadratic

        # Reward braking before sharp turns (teaches the AI to use negative throttle)
        if curv_norm > 0.2 and throttle < -0.1:
            reward += 0.3 * curv_norm * abs(throttle)

        terminated = False
        truncated = False

        if self.offtrack_counter > self.offtrack_grace:
            terminated = True
            reward -= 10.0  # crash penalty (not so harsh that sitting still beats driving)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.lap_progress >= self.track.total:
            reward += 50.0  # big lap bonus
            terminated = True

        obs = self._get_obs()
        info = {
            "progress_ds": ds,
            "lap_progress": self.lap_progress,
            "cte": float(cte),
            "speed": float(self.vel),
            "offtrack": not on_track,
        }

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return obs, float(reward), terminated, truncated, info

    # -------------------------
    # Rendering
    # -------------------------
    def _init_pygame(self):
        import pygame
        if self._pygame_inited:
            return
        pygame.init()
        pygame.display.set_caption("RetroRacer")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Consolas", 18)

        self.W, self.H = 960, 740
        self._screen = pygame.display.set_mode((self.W, self.H))
        self._pygame_inited = True

    def _world_to_screen(self, p_world, cam_pos, zoom):
        x = (p_world[0] - cam_pos[0]) * zoom + self.W / 2
        y = (p_world[1] - cam_pos[1]) * zoom + self.H / 2
        return (int(x), int(y))

    def _compute_track_edges(self):
        """Compute averaged normals at each vertex for smooth edge offset."""
        pts = self.track.pts
        N = len(pts)
        outer = []
        inner = []
        for i in range(N):
            prev = pts[(i - 1) % N]
            curr = pts[i]
            nxt = pts[(i + 1) % N]

            tang1 = curr - prev
            tang1 = tang1 / (float(np.linalg.norm(tang1)) + 1e-8)
            tang2 = nxt - curr
            tang2 = tang2 / (float(np.linalg.norm(tang2)) + 1e-8)

            avg_tang = tang1 + tang2
            avg_tang = avg_tang / (float(np.linalg.norm(avg_tang)) + 1e-8)
            normal = np.array([-avg_tang[1], avg_tang[0]], dtype=np.float32)

            outer.append(curr + normal * self.track.half_width)
            inner.append(curr - normal * self.track.half_width)
        return outer, inner

    def render(self):
        import pygame
        self._init_pygame()
        scr = self._screen
        scr.fill((10, 10, 18))

        zoom = 0.72
        cam = self.pos.copy()

        pts = self.track.pts
        N = len(pts)

        edges_outer, edges_inner = self._compute_track_edges()

        outer_scr = [self._world_to_screen(p, cam, zoom) for p in edges_outer]
        inner_scr = [self._world_to_screen(p, cam, zoom) for p in edges_inner]
        center_scr = [self._world_to_screen(p, cam, zoom) for p in pts]

        # Road surface
        ROAD = (30, 32, 44)
        for i in range(N):
            j = (i + 1) % N
            quad = [outer_scr[i], outer_scr[j], inner_scr[j], inner_scr[i]]
            pygame.draw.polygon(scr, ROAD, quad)

        # Edge lines
        pygame.draw.lines(scr, (0, 230, 195), True, outer_scr, 2)
        pygame.draw.lines(scr, (0, 230, 195), True, inner_scr, 2)

        # Center dashes
        for i in range(0, N, 4):
            j = (i + 2) % N
            pygame.draw.line(scr, (180, 50, 160), center_scr[i], center_scr[j], 1)

        # Finish line (checkered pattern at start/finish)
        fl_outer = np.array(outer_scr[0], dtype=np.float32)
        fl_inner = np.array(inner_scr[0], dtype=np.float32)
        n_checks = 8
        for ci in range(n_checks):
            t0 = ci / n_checks
            t1 = (ci + 1) / n_checks
            p0 = fl_outer * (1 - t0) + fl_inner * t0
            p1 = fl_outer * (1 - t1) + fl_inner * t1
            color = (240, 240, 240) if ci % 2 == 0 else (40, 40, 50)
            # Draw as a thick line segment
            pygame.draw.line(scr, color, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 3)

        # Car — pixel-art F1 car
        car_scr = self._world_to_screen(self.pos, cam, zoom)
        ang = self.heading
        cx, cy = car_scr

        def _rot(lx, ly):
            """Local car coords → screen coords. +lx = forward, +ly = left."""
            rx = lx * math.cos(ang) - ly * math.sin(ang)
            ry = lx * math.sin(ang) + ly * math.cos(ang)
            return (int(cx + rx), int(cy + ry))

        # Body (tapered — wide rear, narrow nose)
        body = [_rot(18, -4), _rot(18, 4), _rot(-12, 5), _rot(-12, -5)]
        pygame.draw.polygon(scr, (220, 30, 40), body)  # Ferrari red

        # Nose cone
        nose = [_rot(18, -3), _rot(24, 0), _rot(18, 3)]
        pygame.draw.polygon(scr, (180, 25, 35), nose)

        # Rear wing
        rw = [_rot(-14, -7), _rot(-12, -7), _rot(-12, 7), _rot(-14, 7)]
        pygame.draw.polygon(scr, (160, 165, 175), rw)  # silver

        # Front wing
        fw = [_rot(20, -6), _rot(22, -6), _rot(22, 6), _rot(20, 6)]
        pygame.draw.polygon(scr, (160, 165, 175), fw)  # silver

        # Cockpit
        cock = [_rot(6, -2), _rot(8, 0), _rot(6, 2), _rot(1, 2), _rot(-1, 0), _rot(1, -2)]
        pygame.draw.polygon(scr, (30, 40, 70), cock)  # dark blue

        # Wheels (4 rectangles)
        for wx, wy in [(14, 5), (14, -5), (-10, 6), (-10, -6)]:
            wh = [_rot(wx-2, wy-1.5), _rot(wx+2, wy-1.5),
                  _rot(wx+2, wy+1.5), _rot(wx-2, wy+1.5)]
            pygame.draw.polygon(scr, (60, 60, 65), wh)
            pygame.draw.polygon(scr, (100, 100, 110), wh, 1)  # outline

        # Driver helmet dot
        pygame.draw.circle(scr, (255, 230, 50), _rot(4, 0), 2)

        # Opponent cars
        opp_colors = [(30, 100, 220), (40, 180, 60), (200, 120, 30)]
        for oi, opp in enumerate(self.opponents):
            opp_pos, opp_heading = opp[0], opp[1]
            opp_scr = self._world_to_screen(opp_pos, cam, zoom)
            oang = opp_heading
            ocx, ocy = opp_scr
            color = opp_colors[oi % len(opp_colors)]
            darker = (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40))

            def _orot(lx, ly):
                rx = lx * math.cos(oang) - ly * math.sin(oang)
                ry = lx * math.sin(oang) + ly * math.cos(oang)
                return (int(ocx + rx), int(ocy + ry))

            # Body
            obody = [_orot(18, -4), _orot(18, 4), _orot(-12, 5), _orot(-12, -5)]
            pygame.draw.polygon(scr, color, obody)
            # Nose
            onose = [_orot(18, -3), _orot(24, 0), _orot(18, 3)]
            pygame.draw.polygon(scr, darker, onose)
            # Wings
            orw = [_orot(-14, -7), _orot(-12, -7), _orot(-12, 7), _orot(-14, 7)]
            pygame.draw.polygon(scr, (160, 165, 175), orw)
            ofw = [_orot(20, -6), _orot(22, -6), _orot(22, 6), _orot(20, 6)]
            pygame.draw.polygon(scr, (160, 165, 175), ofw)
            # Cockpit
            ocockt = [_orot(6, -2), _orot(8, 0), _orot(6, 2), _orot(1, 2), _orot(-1, 0), _orot(1, -2)]
            pygame.draw.polygon(scr, (30, 40, 70), ocockt)
            # Wheels
            for wx, wy in [(14, 5), (14, -5), (-10, 6), (-10, -6)]:
                owh = [_orot(wx-2, wy-1.5), _orot(wx+2, wy-1.5),
                       _orot(wx+2, wy+1.5), _orot(wx-2, wy+1.5)]
                pygame.draw.polygon(scr, (60, 60, 65), owh)
                pygame.draw.polygon(scr, (100, 100, 110), owh, 1)
            # Helmet
            pygame.draw.circle(scr, (200, 200, 210), _orot(4, 0), 2)

        # HUD
        lap_pct = self.lap_progress / self.track.total * 100 if self.track.total > 0 else 0
        hud = f"SPD {self.vel:5.1f}  |  STEP {self.step_count:4d}  |  LAP {lap_pct:5.1f}%  |  OFF {self.offtrack_counter:02d}"
        text = self._font.render(hud, True, (180, 200, 240))
        scr.blit(text, (14, 12))

        # Race HUD — show positions
        if self.opponents:
            for oi, opp in enumerate(self.opponents):
                if len(opp) >= 3:
                    ai_pct = opp[2] / self.track.total * 100 if self.track.total > 0 else 0
                    lead = lap_pct - ai_pct
                    if lead > 0:
                        pos_str = f"YOU lead by {lead:.1f}%"
                    elif lead < 0:
                        pos_str = f"AI leads by {-lead:.1f}%"
                    else:
                        pos_str = "TIED"
                    race_hud = f"AI LAP {ai_pct:5.1f}%  |  {pos_str}"
                    race_text = self._font.render(race_hud, True, (100, 160, 240))
                    scr.blit(race_text, (14, 36))

        if not self._delay_flip:
            pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(scr)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._pygame_inited:
            import pygame
            pygame.quit()
            self._pygame_inited = False
