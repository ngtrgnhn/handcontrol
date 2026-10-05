#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gesture_control.py - AI Python: Dùng cử chỉ bàn tay (webcam) để điều khiển máy tính.

Nguyên tắc (theo yêu cầu):
    * Bàn tay di chuyển LÊN     -> vuốt lên      (phím Page Up / mũi tên Lên)
    * Bàn tay di chuyển XUỐNG   -> vuốt xuống    (phím Page Down / mũi tên Xuống)
    * Bàn tay di chuyển QUA TRÁI-> vuốt trái     (phím Page Left... dùng phím Mũi tên Trái)
    * Bàn tay di chuyển QUA PHẢI-> vuốt phải     (phím Mũi tên Phải)

Cách hoạt động:
    1. Mediapipe HandLandmarker phát hiện 21 điểm khớp của bàn tay trong từng khung hình webcam.
    2. Chương trình theo dõi vị trí tâm bàn tay (trung bình các khớp) và tính "vận tốc" theo trục X/Y.
    3. Khi cổ tay đi vào VÙNG LỬA (trigger zone - ô chữ nhật ở giữa màn hình) rồi rời khỏi vùng đó,
       hướng di chuyển dominant (lên/xuống/trái/phải) được suy ra từ độ lệch tổng cộng -> tạo một cú "vuốt".
       Cách này lọc bỏ hoàn toàn rung lắc nhỏ khi tay đứng yên.
    4. Mỗi cú vuốt được ánh xạ thành phím thật bằng pynput (mặc định: PageUp/PageDown/Left/Right),
       hoặc có thể gửi cuộn chuột (scroll wheel) với --mode scroll.

Yêu cầu thư viện:
    pip install mediapipe opencv-python pynput numpy

Model (tải tự động nếu thiếu):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Chạy:
    python gesture_control.py                 # vuốt = PageUp/PageDown/Left/Right
    python gesture_control.py --mode arrow    # vuốt = mũi tên (hợp cho video/player/slideshow)
    python gesture_control.py --mode scroll   # vuốt = cuộn chuột thật sự (TikTok/Reels/browser)
    python gesture_control.py --no-preview    # không hiển thị cửa sổ camera
    python gesture_control.py --simulate      # kiểm tra logic nhận diện (headless, không cần webcam)

Thoát: nhấn 'q' trên cửa sổ preview, Ctrl+C trong terminal, hoặc giơ HAI TAY > 1.5 giây.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ------------------------------------------------------------------ #
#  Các hằng số mặc định (có thể override bằng command line)          #
# ------------------------------------------------------------------ #
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = "hand_landmarker.task"

# Khớp cổ tay (wrist) và các đầu ngón trong mô hình 21 điểm của MediaPipe
WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]

UP, DOWN, LEFT, RIGHT, NONE = "UP", "DOWN", "LEFT", "RIGHT", "NONE"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ #
#  Ánh xạ phím                                                       #
# ------------------------------------------------------------------ #
class KeyBackend:
    """Gửi phím thật ra hệ thống bằng pynput."""

    def __init__(self, mapping: dict):
        from pynput.keyboard import Controller, Key  # import trễ để chạy --simulate được
        self._ctrl = Controller()
        self._Key = Key
        self.mapping = mapping  # direction -> 'pageup'|'pagedown'|'left'|'right'|'up'|'down'

    def press(self, direction: str) -> str:
        key_name = self.mapping.get(direction)
        if not key_name:
            return ""
        special = {"pageup": "page_up", "pagedown": "page_down",
                   "home": "home", "end": "end"}
        if key_name in special:
            key = getattr(self._Key, special[key_name])
        elif key_name in ("left", "right", "up", "down"):
            key = getattr(self._Key, key_name)
        else:  # ký thường
            key = key_name
        self._ctrl.press(key)
        self._ctrl.release(key)
        return str(key_name)


class ScrollBackend:
    """Cuộn chuột thật sự (wheel) - hợp với TikTok/Reels/feed/web."""

    def __init__(self, amount: int = 5):
        from pynput.mouse import Controller
        self._mouse = Controller()
        self.amount = amount

    def scroll(self, direction: str) -> str:
        dy = 0
        dx = 0
        if direction == UP:
            dy = self.amount          # cuộn lên
        elif direction == DOWN:
            dy = -self.amount         # cuộn xuống
        elif direction == LEFT:
            dx = self.amount
        elif direction == RIGHT:
            dx = -self.amount
        if dx or dy:
            self._mouse.scroll(dx, dy)
        return f"scroll({dx},{dy})"


class NullBackend:
    """Không gửi gì (dùng khi --dry-run / --simulate)."""

    def press(self, direction: str) -> str:
        return f"<{direction}>"

    def scroll(self, direction: str) -> str:
        return f"<{direction}>"


# ------------------------------------------------------------------ #
#  Bộ phát hiện vuốt (thuật toán lõi, độc lập với webcam)             #
# ------------------------------------------------------------------ #
@dataclass
class SwipeConfig:
    # Vùng lửa (toạ độ tỉ lệ 0..1 trong khung hình mirror). Vuốt = Tay VÀO rồi RA khỏi vùng này.
    zone_x_min: float = 0.30
    zone_x_max: float = 0.70
    zone_y_min: float = 0.28
    zone_y_max: float = 0.72
    # Quãng đường tối thiểu (tỉ lệ khung hình) từ điểm neo tới điểm thoát để tính là 1 cú vuốt
    min_dist: float = 0.14
    # Bước cuốn điểm neo khi tay di chuyển trong vùng (tỉ lệ khung hình)
    anchor_step: float = 0.05
    # Thời gian tối đa được phép nằm trong vùng lửa (giây) - vuốt phải dứt khoát
    max_hold: float = 1.6
    # Số giây chờ giữa 2 cú vuốt liên tiếp (cooldown)
    cooldown: float = 0.35
    # Cửa sổ thời gian giữ lịch sử toạ độ (hiển thị/debug)
    window: float = 0.45
    # Nếu tay biến mất quá lâu thì reset trạng thái
    reset_after_outside: float = 1.2
    # Bán kính "vùng chết": sau một cú vuốt, tay phải rút ra xa vùng ít nhất chừng này
    # mới được ARM lại (chống vuốt giả dây chuyền khi tay vung qua vung nhiều lan)
    deadzone_r: float = 0.08


@dataclass
class SwipeDetector:
    cfg: SwipeConfig = field(default_factory=SwipeConfig)

    def __post_init__(self) -> None:
        self._hist: deque[tuple[float, float, float]] = deque()  # (t, x, y)
        self._in_zone = False
        self._enter_time: Optional[float] = None
        self._anchor_pt: tuple[float, float] = (0.5, 0.5)   # điểm neo dùng để đo hướng vuốt
        self._start_pt: tuple[float, float] = (0.5, 0.5)    # điểm tay vừa vào vùng lửa
        self._last_fire_ts: float = -10.0
        self._last_seen_ts: float = 0.0

    # ---------- helpers ----------
    @staticmethod
    def _angle_to_dir(dx: float, dy: float) -> str:
        """Quy đổi vector (dx, dy) thành hướng vuốt. dy dương = xuống dưới màn hình."""
        if dx == 0 and dy == 0:
            return NONE
        ang = math.degrees(math.atan2(dy, dx))  # -180..180 ; 0 = phải, 90 = xuống
        # Chia 4 cung 90°
        if -45 <= ang < 45:
            return RIGHT
        if 45 <= ang < 135:
            return DOWN
        if ang >= 135 or ang < -135:
            return LEFT
        return UP  # -135..-45

    def _zone_contains(self, x: float, y: float) -> bool:
        c = self.cfg
        return (c.zone_x_min <= x <= c.zone_x_max) and (c.zone_y_min <= y <= c.zone_y_max)

    def reset(self) -> None:
        self._hist.clear()
        self._in_zone = False
        self._enter_time = None
        self._anchor_pt = (0.5, 0.5)
        self._start_pt = (0.5, 0.5)

    # ---------- API chính ----------
    def update(self, x: float, y: float, ts: Optional[float] = None) -> Optional[str]:
        """Cập nhật vị trí tâm bàn tay (tỉ lệ 0..1, ảnh đã mirror). Trả về hướng vừa vuốt hoặc None.

        Máy trạng thái "vùng lửa" (trigger-zone swipe) - 2 trang thai:
          IDLE      : tay ngoai vung lua. Tay di VAO vung -> ghi diem khoi dau (start)
                      va diem neo (anchor).
          TRACKING  : khi tay con trong vung, diem neo duoc cuon theo moi lan tay dich
                      qua anchor_step nhung BI CHIẾU về vòng tròn bán kính min_dist quanh
                      diem khoi dau. Nho vay:
                        * tay dung yen / lo lửng giua vung -> khong tao vuot gia;
                        * tay doi huong ( trai roi phai ) -> vector thoat bi triet tieu;
                        * moi cu vuot dut khoat luon con du "duong kinh" de do huong.
          TRIGGER   : tay RUOI KHOI vung, neu quang duong tu diem neo >= min_dist va
                      thoi gian trong vung <= max_hold (va da het cooldown) -> phat sinh
                      cu vuot theo huong dominant cua vector (neo -> hien tai).
                      Neu bi chan (ngan/cham/trung cooldown) thi phien vuot chi ket thuc
                      khi tay ve "vung an toan" (xa hon deadzone) -> chong vuot day
                      chuyen khi tay vung lien tuc quanh vien vung lua.
        """
        if ts is None:
            ts = time.monotonic()
        c = self.cfg

        # Ghi lịch sử (phục vụ debug/hiển thị)
        self._hist.append((ts, x, y))
        while self._hist and ts - self._hist[0][0] > max(c.window, 0.25):
            self._hist.popleft()

        inside = self._zone_contains(x, y)

        # Tay bien mat qua lau (mat track) -> lam moi trang thai
        if self._last_seen_ts and ts - self._last_seen_ts > c.reset_after_outside and not inside:
            self.reset()
        self._last_seen_ts = ts

        fired: Optional[str] = None

        # Vung an toan (safe zone): tay o day du xa vung lua de ket thuc phien vuot
        # va cho phep bat dau mot cu vuot moi.
        zx = (c.zone_x_min + c.zone_x_max) / 2.0
        zy = (c.zone_y_min + c.zone_y_max) / 2.0
        safe = (abs(x - zx) > (c.zone_x_max - c.zone_x_min) / 2.0 + c.deadzone_r or
                abs(y - zy) > (c.zone_y_max - c.zone_y_min) / 2.0 + c.deadzone_r)

        if inside:
            # --- TAY ĐANG Ở TRONG VÙNG LỬA ---
            if not self._in_zone:
                # Lần đầu đi vào vùng trong "phiên vuốt" này -> lấy làm điểm khởi đầu
                self._in_zone = True
                self._enter_time = ts
                self._start_pt = (x, y)
                self._anchor_pt = (x, y)
            else:
                # Điểm neo được cuốn theo tay, NHƯNG không bao giờ vượt quá min_dist
                # so với điểm khởi đầu. Nhờ vậy:
                #   * tay đứng yên/lơ lửng giữa vùng -> không phát sinh vuốt giả;
                #   * tay đổi hướng (trái rồi phải) -> vector thoát bị triệt tiêu;
                #   * một cú vuốt dứt khoát -> luôn còn đủ "đường kính" để đo hướng.
                sx, sy = self._start_pt
                ax, ay = self._anchor_pt
                if math.hypot(x - ax, y - ay) > c.anchor_step:
                    nx, ny = x, y
                    r = math.hypot(nx - sx, ny - sy)
                    if r > c.min_dist:                      # chiếu ngược về vòng tròn bán kính min_dist
                        k = c.min_dist / r
                        nx, ny = sx + (nx - sx) * k, sy + (ny - sy) * k
                    self._anchor_pt = (nx, ny)
        elif self._in_zone:
            # --- TAY VUA RUOI KHỎI VUNG LUA: danh gia cu vuot ---
            ex, ey = self._anchor_pt
            dx, dy = x - ex, y - ey
            dist = math.hypot(dx, dy)
            in_dur = ts - (self._enter_time or ts)
            if (dist >= c.min_dist and in_dur <= c.max_hold
                    and (ts - self._last_fire_ts) >= c.cooldown):
                fired = self._angle_to_dir(dx, dy)
                self._last_fire_ts = ts
                self._in_zone = False
                self._enter_time = None
            elif safe:
                # Qua ngan / qua cham / dang cooldown -> huy phien khi tay da ve vung an toan
                self._in_zone = False
                self._enter_time = None

        return fired


# ------------------------------------------------------------------ #
#  Phát hiện bàn tay qua webcam                                      #
# ------------------------------------------------------------------ #
def ensure_model(path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    log(f"Không thấy model '{path}', đang tải về ...")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, path)
        log(f"Đã tải model về {path}")
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Tự tải model thất bại ({e}). Hãy tải thủ công:\n  {MODEL_URL}\nvà lưu thành '{path}'."
        )
    return path


class HandTracker:
    """Bọc MediaPipe Tasks HandLandmarker (VIDEO mode) + lấy tâm bàn tay & đếm ngón."""

    def __init__(self, model_path: str, num_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        base = mp_tasks.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t_offset = time.monotonic()

    def process(self, frame_bgr: np.ndarray, ts_ms: int):
        mp = self._mp
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        res = self._landmarker.detect_for_video(rgb, ts_ms)
        hands = []
        for lm in res.hand_landmarks:
            pts = np.array([(p.x, p.y) for p in lm], dtype=np.float32)
            center = pts.mean(axis=0)                      # tâm bàn tay (tỉ lệ)
            spread = float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1]))
            # Đếm ngón duỗi: đầu ngón cao hơn (y nhỏ hơn) khớp PIP tương ứng
            extended = 0
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 16)):
                if lm[tip].y < lm[pip].y - 0.02:
                    extended += 1
            hands.append({"center": (float(center[0]), float(center[1])),
                          "spread": spread,
                          "extended": extended,
                          "points": pts})
        return hands


# ------------------------------------------------------------------ #
#  Vẽ overlay lên cửa sổ preview                                     #
# ------------------------------------------------------------------ #
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def draw_overlay(frame, hands, detector: SwipeDetector, status: str, last_action: str):
    if cv2 is None:
        return frame
    h, w = frame.shape[:2]
    c = detector.cfg
    # Vùng lửa
    x0, x1 = int(c.zone_x_min * w), int(c.zone_x_max * w)
    y0, y1 = int(c.zone_y_min * h), int(c.zone_y_max * h)
    color = (0, 220, 0) if detector._in_zone else (0, 160, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(frame, "TRIGGER ZONE (dua tay vao roi vuot ra)",
                (x0, max(y0 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for hd in hands:
        cx, cy = hd["center"]
        px, py = int(cx * w), int(cy * h)
        cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)
        cv2.circle(frame, (px, py), 16, (255, 255, 255), 2)
        for (mx, my) in hd["points"]:
            cv2.circle(frame, (int(mx * w), int(my * h)), 2, (0, 255, 255), -1)

    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if last_action:
        cv2.putText(frame, f"Action: {last_action}", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


# ------------------------------------------------------------------ #
#  Vòng lặp chính                                                    #
# ------------------------------------------------------------------ #
KEY_MAPS = {
    "page": {UP: "pageup", DOWN: "pagedown", LEFT: "left", RIGHT: "right"},
    "arrow": {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"},
    "scroll": {},  # dùng ScrollBackend
}


def run(args) -> None:
    cfg = SwipeConfig(
        min_dist=args.min_dist,
        cooldown=args.cooldown,
        speed_thresh=args.speed_thresh,
    )
    detector = SwipeDetector(cfg)

    if args.mode == "scroll":
        backend = ScrollBackend(amount=args.scroll_amount)
    elif args.dry_run or args.simulate:
        backend = NullBackend()
    else:
        backend = KeyBackend(KEY_MAPS[args.keys])

    tracker = HandTracker(ensure_model(args.model), num_hands=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Không mở được webcam. Kiểm tra --camera (thử 0,1,2).")

    exit_key_stop = None
    if not args.no_preview:
        from pynput import keyboard as kb
        stop_evt = threading.Event()

        def on_press(k):
            if k == kb.Key.esc:
                stop_evt.set()
        listener = kb.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        exit_key_stop = stop_evt

    log("Sẵn sàng! Đưa bàn tay vào ô vàng giữa màn hình rồi VUỐT ra ngoài theo hướng mong muốn.")
    log("  Len  = Vuot len | Xuong = Vuot xuong | Trai/Phai = Vuot ngang. Esc/q de thoat.")

    last_action = ""
    two_hands_since: Optional[float] = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame_idx += 1
            # Mirror cho cảm giác tự nhiên (tay trái/phải đều được chấp nhận)
            frame = cv2.flip(frame, 1)
            ts_ms = int((time.monotonic() - tracker._t_offset) * 1000)
            hands = tracker.process(frame, ts_ms)

            status = f"Hands: {len(hands)} | Mode: {args.mode}"

            # Thoát bằng giơ 2 tay
            if len(hands) >= 2:
                if two_hands_since is None:
                    two_hands_since = time.monotonic()
                status += " | GIU 2 TAY = THOAT"
                if time.monotonic() - two_hands_since > args.exit_hold:
                    log("Phat hien 2 tay > {:.1f}s -> thoat.".format(args.exit_hold))
                    break
            else:
                two_hands_since = None

            # Chọn tay chủ đạo: ưu tiên tay đang ở trong vùng lửa, nếu không thì tay gần tâm nhất
            primary = None
            if hands:
                in_zone = [hd for hd in hands if detector._zone_contains(*hd["center"])]
                pool = in_zone or hands
                cx_img, cy_img = 0.5, 0.5
                primary = min(pool, key=lambda hd: (hd["center"][0]-cx_img)**2 + (hd["center"][1]-cy_img)**2)

            direction = NONE
            if primary is not None:
                cx, cy = primary["center"]
                direction = detector.update(cx, cy) or NONE

            if direction != NONE:
                if args.mode == "scroll":
                    desc = backend.scroll(direction)
                else:
                    desc = backend.press(direction)
                last_action = f"{direction} -> {desc}"
                log(f"SWIPE {direction:<5} | {last_action}")

            if not args.no_preview:
                draw_overlay(frame, hands, detector, status, last_action)
                cv2.imshow("Gesture Control - q/Esc to quit", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q')):
                    break
            if exit_key_stop is not None and exit_key_stop.is_set():
                break
    finally:
        cap.release()
        if cv2 is not None and not args.no_preview:
            cv2.destroyAllWindows()
        log("Da dung. Tam biet!")


# ------------------------------------------------------------------ #
#  Chế độ SIMULATE: kiểm chứng thuật toán vuốt mà không cần webcam    #
# ------------------------------------------------------------------ #
def simulate(args) -> None:
    """Kiểm chứng thuật toán nhận diện vuốt bằng quỹ đạo toạ độ giả lập (không cần webcam)."""
    log("=== CHE DO SIMULATE: kiem thu thuat toan nhan dien vuot 4 huong ===")

    DT = 0.05          # ~20 fps
    COOLDOWN = 0.35

    def line(p0, p1, n):
        return [(p0[0] + (p1[0]-p0[0]) * i / n, p0[1] + (p1[1]-p0[1]) * i / n)
                for i in range(1, n + 1)]

    class Player:
        """Phat mot quy dao va ghi lai cac huong vuot phat sinh."""
        def __init__(self):
            self.det = SwipeDetector(SwipeConfig(cooldown=COOLDOWN))
            self.t = 0.0

        def play(self, pts):
            fired = []
            for (x, y) in pts:
                self.t += DT
                d = self.det.update(x, y, ts=self.t)
                if d:
                    fired.append(d)
            return fired

        def wait(self, seconds):
            """Giữ tay o xa vung lua trong `seconds` (de het cooldown + ARM lai)."""
            n = max(1, int(seconds / DT))
            self.play([(0.05, 0.05)] * n)

    ok = True

    def check(name, got, expect):
        nonlocal ok
        good = (got == expect)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<34} got={got} expect={expect}")

    # ---------- 1) Bon huong co ban (vuot qua tam vung lua) ----------
    for direction, target in ((UP, (0.50, 0.02)), (DOWN, (0.50, 0.98)),
                              (LEFT, (0.02, 0.50)), (RIGHT, (0.98, 0.50))):
        pl = Player()
        pl.wait(1.0)                                   # tay bat dau o goc tren-trai (ARM san)
        enter = line((0.05, 0.05), (0.50, 0.50), 6)     # di vao tam vung lua
        out = line((0.50, 0.50), target, 8)             # vuot thang ra ngoai
        got = pl.play(enter + out)
        check(f"VUOT {direction}", got, [direction])

    # ---------- 2) Vuot cheo: duoi cung cua goc se quyet dinh ----------
    pl = Player(); pl.wait(1.0)
    got = pl.play(line((0.05, 0.05), (0.50, 0.50), 6) + line((0.50, 0.50), (0.97, 0.03), 8))
    check("VUOT CHEO TREN-PHAI (ang=-45.6)", got, [UP])
    pl = Player(); pl.wait(1.0)
    got = pl.play(line((0.05, 0.05), (0.50, 0.50), 6) + line((0.50, 0.50), (0.97, 0.05), 8))
    check("VUOT CHEO NGHIENG PHAI (ang>-45)", got, [RIGHT])

    # ---------- 3) Chong vuot gia ----------
    pl = Player(); pl.wait(1.0)
    # 3a. Tay rung lac lien tuc trong vung -> khong duoc phat sinh gi
    jitter = [(0.50 + 0.010*math.sin(i*0.6), 0.50 + 0.010*math.cos(i*0.6)) for i in range(60)]
    check("RUNG LAC TRONG VUNG", pl.play(jitter), [])

    pl = Player(); pl.wait(1.0)
    # 3b. Tay di vao roi thoat ra rat ngan (< min_dist) -> khong phai vuot
    short = line((0.05, 0.05), (0.50, 0.50), 6) + line((0.50, 0.50), (0.50, 0.60), 4)
    check("THOAT NGAN (du cu ly)", pl.play(short), [])

    pl = Player(); pl.wait(1.0)
    # 3c. Doi huong trong vung (xuong roi len) -> vector thoat triet tieu
    zigzag = (line((0.05, 0.05), (0.50, 0.50), 6)
              + line((0.50, 0.50), (0.50, 0.70), 5)
              + line((0.50, 0.70), (0.50, 0.30), 6)
              + line((0.50, 0.30), (0.50, 0.02), 5))
    check("DOI HUONG TRONG VUNG (triet tieu)", pl.play(zigzag), [])

    pl = Player(); pl.wait(1.0)
    # 3d. Nam Yen trong vung qua lau (> max_hold) roi thoat -> bi chan
    slow = (line((0.05, 0.05), (0.50, 0.50), 6)
            + [(0.50, 0.50)] * 40            # ~2s nam yen trong vung
            + line((0.50, 0.50), (0.50, 0.02), 8))
    check("NAM YEN LAU ROI THOAT (max_hold)", pl.play(slow), [])

    pl = Player(); pl.wait(1.0)
    # 3e. Hai lan vuot lien tuc KHONG rut tay ra xa -> lan 2 bi chan (deadzone)
    first = line((0.05, 0.05), (0.50, 0.50), 6) + line((0.50, 0.50), (0.50, 0.02), 8)
    back_in = line((0.50, 0.02), (0.50, 0.50), 6)      # quay lai vung ngay, khong ra xa
    second = line((0.50, 0.50), (0.50, 0.02), 8)
    check("VUOT LAP LIEN TIEN (deadzone)", pl.play(first + back_in + second), [UP])

    # ---------- 4) Chuoi vuot that (co rut tay ve vi tri xua) ----------
    pl = Player()
    seq = []
    for target in ((0.50, 0.02), (0.50, 0.98), (0.98, 0.50), (0.02, 0.50)):
        pl.wait(0.8)                                    # rut tay ra goc va cho cooldown
        seq += pl.play(line((0.05, 0.05), (0.50, 0.50), 6) + line((0.50, 0.50), target, 8))
    check("CHUOI LEN-XUONG-PHAI-TRAI", seq, [UP, DOWN, RIGHT, LEFT])

    # ---------- 5) Anh xa thanh phim/cuon ----------
    if args.mode == "scroll":
        backend = NullBackend()
        acts = [backend.scroll(d) for d in (UP, DOWN, LEFT, RIGHT)]
    else:
        backend = NullBackend()
        KEY_MAPS_LOCAL = {"page": {UP: "pageup", DOWN: "pagedown", LEFT: "left", RIGHT: "right"},
                          "arrow": {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}}
        m = KEY_MAPS_LOCAL[args.keys]
        orig = backend.press
        acts = [f"{d}->{m[d]}" for d in (UP, DOWN, LEFT, RIGHT)]
    print("  Anh xa ban phim (" + args.keys + "): " + ", ".join(acts))

    print("=" * 62)
    if ok:
        log(">>> TAT CA KIEM TRA DA DAT. Thuat toan san sang dung voi webcam.")
    else:
        log(">>> CO KIEM TRA KHONG DAT! Xem chi tiet o tren.")
        sys.exit(1)


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dieu khien may tinh bang cu chi ban tay (MediaPipe + OpenCV + pynput)")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Duong dan file hand_landmarker.task")
    p.add_argument("--camera", type=int, default=0, help="Chi so webcam (0,1,...)")
    p.add_argument("--mode", choices=["swipe", "scroll"], default="swipe",
                   help="swipe = gui phim; scroll = cuon chuot that")
    p.add_argument("--keys", choices=["page", "arrow"], default="page",
                   help="Kieu phim khi mode=swipe: page (PageUp/Down+Left/Right) hoac arrow (4 mui ten)")
    p.add_argument("--scroll-amount", type=int, default=5, help="Do lon cuon moi lan (mode=scroll)")
    p.add_argument("--min-dist", type=float, default=0.10, help="Vuong goc toi thieu de tinh la 1 vuot (0..1)")
    p.add_argument("--speed-thresh", type=float, default=0.004, help="Van toc toi thieu nap chuyen dong")
    p.add_argument("--cooldown", type=float, default=0.65, help="So giay cho giua 2 cu vuot")
    p.add_argument("--exit-hold", type=float, default=1.5, help="Giu 2 tay bao lau (giay) de thoat")
    p.add_argument("--no-preview", action="store_true", help="Khong hien cua so webcam")
    p.add_argument("--dry-run", action="store_true", help="Nhan dien nhung KHONG gui phim/chuot that")
    p.add_argument("--simulate", action="store_true", help="Chi kiem thu thuat toan (headless)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.simulate:
        simulate(args)
        return
    if cv2 is None:
        raise SystemExit("Thieu OpenCV. Cai dat: pip install opencv-python")
    run(args)


if __name__ == "__main__":
    main()
