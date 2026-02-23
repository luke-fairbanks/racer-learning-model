"""Visual track editor with click-to-place, drag, delete, and live preview."""

import os
import numpy as np

from racing.track import _catmull_rom, save_track_json, load_track_json


def track_editor(load_file=None, save_path="tracks/custom.json"):
    """Visual track editor. Click to place points, drag to move, right-click to delete."""
    import pygame

    pygame.init()
    W, H = 960, 740
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("RetroRacer — Track Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    small_font = pygame.font.SysFont("Consolas", 13)

    cam_x, cam_y = 0.0, 0.0
    zoom = 0.65

    control_pts = []
    half_width = 42.0

    if load_file and os.path.exists(load_file):
        control_pts_loaded, half_width = load_track_json(load_file)
        control_pts = [list(p) for p in control_pts_loaded]

    dragging_idx = None
    panning = False
    pan_start = None
    cam_start = None
    status_msg = "Click to place points. Right-click to delete. S = save, T = test drive."
    status_timer = 0

    def world_to_screen(wx, wy):
        sx = (wx - cam_x) * zoom + W / 2
        sy = (wy - cam_y) * zoom + H / 2
        return int(sx), int(sy)

    def screen_to_world(sx, sy):
        wx = (sx - W / 2) / zoom + cam_x
        wy = (sy - H / 2) / zoom + cam_y
        return wx, wy

    def find_point_near(sx, sy, radius=14):
        for i, pt in enumerate(control_pts):
            px, py = world_to_screen(pt[0], pt[1])
            if (px - sx) ** 2 + (py - sy) ** 2 < radius ** 2:
                return i
        return None

    def build_spline_preview():
        if len(control_pts) < 3:
            return []
        tuples = [tuple(p) for p in control_pts]
        N = len(tuples)
        pts_per_seg = 10
        all_pts = []
        for i in range(N):
            p0 = tuples[(i - 1) % N]
            p1 = tuples[i]
            p2 = tuples[(i + 1) % N]
            p3 = tuples[(i + 2) % N]
            seg = _catmull_rom(p0, p1, p2, p3, pts_per_seg)
            all_pts.extend(seg)
        return all_pts

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_s:
                    if len(control_pts) >= 3:
                        save_track_json([tuple(p) for p in control_pts], save_path, half_width)
                        status_msg = f"Saved to {save_path} ({len(control_pts)} pts)"
                        status_timer = 120
                    else:
                        status_msg = "Need at least 3 points to save!"
                        status_timer = 90

                elif event.key == pygame.K_c:
                    control_pts.clear()
                    status_msg = "Cleared all points"
                    status_timer = 60

                elif event.key == pygame.K_t:
                    if len(control_pts) >= 3:
                        save_track_json([tuple(p) for p in control_pts], save_path, half_width)
                        pygame.quit()
                        from racing.trainer import human_drive
                        human_drive(track_file=save_path)
                        # Re-init editor after driving
                        pygame.init()
                        screen = pygame.display.set_mode((W, H))
                        pygame.display.set_caption("RetroRacer — Track Editor")
                        clock = pygame.time.Clock()
                        font = pygame.font.SysFont("Consolas", 16)
                        small_font = pygame.font.SysFont("Consolas", 13)
                        status_msg = "Back from test drive!"
                        status_timer = 90
                    else:
                        status_msg = "Need at least 3 points to test!"
                        status_timer = 90

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    half_width = min(80, half_width + 2)
                    status_msg = f"Track width: {half_width:.0f}"
                    status_timer = 60
                elif event.key == pygame.K_MINUS:
                    half_width = max(15, half_width - 2)
                    status_msg = f"Track width: {half_width:.0f}"
                    status_timer = 60

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    idx = find_point_near(*event.pos)
                    if idx is not None:
                        dragging_idx = idx
                    else:
                        wx, wy = screen_to_world(*event.pos)
                        control_pts.append([wx, wy])
                        status_msg = f"Point {len(control_pts)} placed"
                        status_timer = 40

                elif event.button == 3:
                    idx = find_point_near(*event.pos)
                    if idx is not None:
                        control_pts.pop(idx)
                        status_msg = f"Deleted point (now {len(control_pts)})"
                        status_timer = 40

                elif event.button == 2:
                    panning = True
                    pan_start = event.pos
                    cam_start = (cam_x, cam_y)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_idx = None
                elif event.button == 2:
                    panning = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging_idx is not None:
                    wx, wy = screen_to_world(*event.pos)
                    control_pts[dragging_idx] = [wx, wy]
                elif panning and pan_start:
                    dx = (event.pos[0] - pan_start[0]) / zoom
                    dy = (event.pos[1] - pan_start[1]) / zoom
                    cam_x = cam_start[0] - dx
                    cam_y = cam_start[1] - dy

            elif event.type == pygame.MOUSEWHEEL:
                old_wx, old_wy = screen_to_world(*mouse_pos)
                zoom *= 1.1 if event.y > 0 else 0.9
                zoom = max(0.1, min(3.0, zoom))
                new_wx, new_wy = screen_to_world(*mouse_pos)
                cam_x -= (new_wx - old_wx)
                cam_y -= (new_wy - old_wy)

        # --- DRAW ---
        screen.fill((10, 10, 18))

        # Grid
        grid_size = 100
        for gx in range(-2000, 2001, grid_size):
            sx, _ = world_to_screen(gx, 0)
            if 0 <= sx <= W:
                pygame.draw.line(screen, (18, 18, 28), (sx, 0), (sx, H), 1)
        for gy in range(-2000, 2001, grid_size):
            _, sy = world_to_screen(0, gy)
            if 0 <= sy <= H:
                pygame.draw.line(screen, (18, 18, 28), (0, sy), (W, sy), 1)

        # Origin crosshair
        ox, oy = world_to_screen(0, 0)
        pygame.draw.line(screen, (40, 40, 50), (ox - 10, oy), (ox + 10, oy), 1)
        pygame.draw.line(screen, (40, 40, 50), (ox, oy - 10), (ox, oy + 10), 1)

        # Spline preview with road width
        spline_pts = build_spline_preview()
        if len(spline_pts) > 2:
            scr_pts = [world_to_screen(p[0], p[1]) for p in spline_pts]

            np_pts = np.array(spline_pts, dtype=np.float32)
            N_sp = len(np_pts)
            outer_pts = []
            inner_pts = []
            for i in range(N_sp):
                prev = np_pts[(i - 1) % N_sp]
                curr = np_pts[i]
                nxt = np_pts[(i + 1) % N_sp]
                t1 = curr - prev
                t1 = t1 / (np.linalg.norm(t1) + 1e-8)
                t2 = nxt - curr
                t2 = t2 / (np.linalg.norm(t2) + 1e-8)
                avg = t1 + t2
                avg = avg / (np.linalg.norm(avg) + 1e-8)
                n = np.array([-avg[1], avg[0]])
                outer_pts.append(world_to_screen(*(curr + n * half_width)))
                inner_pts.append(world_to_screen(*(curr - n * half_width)))

            for i in range(N_sp):
                j = (i + 1) % N_sp
                quad = [outer_pts[i], outer_pts[j], inner_pts[j], inner_pts[i]]
                pygame.draw.polygon(screen, (25, 28, 38), quad)

            pygame.draw.lines(screen, (0, 150, 130), True, outer_pts, 1)
            pygame.draw.lines(screen, (0, 150, 130), True, inner_pts, 1)
            pygame.draw.lines(screen, (0, 200, 170, 120), True, scr_pts, 2)

        # Control point connections
        if len(control_pts) >= 2:
            line_pts = [world_to_screen(p[0], p[1]) for p in control_pts]
            pygame.draw.lines(screen, (60, 60, 80), True, line_pts, 1)

        # Control points
        for i, pt in enumerate(control_pts):
            sx, sy = world_to_screen(pt[0], pt[1])
            hover = find_point_near(*mouse_pos)
            color = (255, 100, 100) if i == hover else (255, 210, 30)
            if i == 0:
                color = (100, 255, 100)
            pygame.draw.circle(screen, color, (sx, sy), 7)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 7, 1)
            label = small_font.render(str(i), True, (200, 200, 200))
            screen.blit(label, (sx + 10, sy - 6))

        # HUD
        help_lines = [
            f"Points: {len(control_pts)}  |  Width: {half_width:.0f}  |  Zoom: {zoom:.2f}",
            "Click=place  Drag=move  RightClick=delete  MidClick=pan  Scroll=zoom",
            "S=save  C=clear  T=test drive  +/-=width  Esc=quit",
        ]
        y_off = 8
        for line in help_lines:
            text = small_font.render(line, True, (140, 160, 180))
            screen.blit(text, (10, y_off))
            y_off += 18

        if status_timer > 0:
            status_timer -= 1
            text = font.render(status_msg, True, (80, 255, 180))
            screen.blit(text, (10, H - 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
