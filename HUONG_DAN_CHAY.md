# Hướng dẫn chạy (How to run) — gesture_control.py

AI Python dùng cử chỉ bàn tay (webcam) để điều khiển máy tính:
vuốt lên / xuống / trái / phải → gửi phím PageUp/PageDown/Left/Right,
hoặc mũi tên, hoặc cuộn chuột thật.

## 1. Yêu cầu hệ thống
- Python **3.9 – 3.12** (Mediapipe chưa hỗ trợ bản mới nhất; kiểm tra: `python --version`)
- Có **webcam**
- Hệ điều hành có giao diện đồ hoạ:
  - **Windows / macOS**: chạy trực tiếp được.
  - **Linux**: cần X11 hoặc Wayland. Trên WSL2 cần cài `mesa-utils`, `xdg-user-dirs` và để `DISPLAY` trỏ về desktop Windows.
- Quyền truy cập webcam (macOS: System Settings → Privacy → Camera).

## 2. Cài đặt

```bash
# (Khuyên dùng) tạo môi trường ảo
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

pip install --upgrade pip
pip install mediapipe opencv-python numpy pynput
```

Model AI (`hand_landmarker.task`, ~7MB) đã có sẵn trong thư mục.
Nếu thiếu, chương trình tự tải từ Google MediaPipe models.

## 3. Kiểm tra logic trước khi cần webcam (headless)

```bash
python gesture_control.py --simulate
```
Không cần camera/màn hình — chỉ kiểm tra thuật toán nhận diện 4 hướng vuốt.

## 4. Chạy thật với webcam

```bash
# Mặc định: vuốt = PageUp / PageDown / Left / Right (lướt web, TikTok, Reels...)
python gesture_control.py

# Vuốt = 4 phím MŨI TÊN (video player, slideshow, game...)
python gesture_control.py --keys arrow

# Vuốt = CUỘN CHUỘT thật sự
python gesture_control.py --mode scroll

# Không hiện cửa sổ camera (chạy nền)
python gesture_control.py --no-preview

# Chỉ nhận diện, in ra log, KHÔNG gửi phím thật (an toàn lúc test)
python gesture_control.py --dry-run

# Chọn webcam khác (1, 2, ...)
python gesture_control.py --camera 1
```

Xem đầy đủ tuỳ chọn: `python gesture_control.py --help`

## 5. Cách sử dụng bằng tay
1. Chạy lệnh → cửa sổ camera hiện ra, có **ô vùng lửa (trigger zone)** ở giữa.
2. Giơ **một bàn tay** vào ô giữa màn hình, sau đó **vuốt nhanh** ra ngoài theo hướng muốn:
   - Tay đi lên → vuốt lên (Page Up / ↑ / cuộn lên)
   - Tay đi xuống → vuốt xuống (Page Down / ↓ / cuộn xuống)
   - Tay qua trái → vuốt trái (← )
   - Tay qua phải → vuốt phải (→ )
3. Đưa focus về ứng dụng cần điều khiển (trình duyệt, TikTok, PowerPoint...).
4. **Thoát**: nhấn `q` trên cửa sổ preview, `Ctrl+C` trong terminal, hoặc giơ **hai tay** giữ > 1.5 giây.

## 6. Tinh chỉnh (nếu nhận diện quá nhạy / quá chậm)

| Cờ | Mặc định | Tác dụng |
|---|---|---|
| `--min-dist` | 0.10 | Giảm (vd 0.06) = dễ kích hoạt hơn; tăng = chặt hơn |
| `--speed-thresh` | 0.004 | Vận tốc tối thiểu để tính là chuyển động chủ ý |
| `--cooldown` | 0.65 | Thời gian chờ giữa 2 cú vuốt (giây) |
| `--scroll-amount` | 5 | Cường độ mỗi lần cuộn (mode scroll) |

## 7. Lỗi thường gặp

- **`ImportError: pynput ... X server / DISPLAY`** (Linux): cần chạy trong môi trường đồ hoạ,
  hoặc dùng `--simulate` / `--dry-run` để test không gửi phím.
- **Không mở được webcam**: thử `--camera 1`; đóng các app đang dùng camera (Zoom, Teams).
- **Mediapipe không cài được trên Python 3.13+**: dùng Python 3.10–3.12
  (`pyenv install 3.11` hoặc conda: `conda create -n gesture python=3.11`).
- **Trên macOS bị chặn camera**: cấp quyền Camera cho Terminal/Python.
- **Vuốt không ăn**: vào vùng lửa rồi thoát RA NGOÀI hẳn ô, động tác dứt khoát;
  xem log in ra để biết chương trình có detect hướng nào không.

## 8. Chạy khi khởi động (tuỳ chọn)
- **Windows**: tạo shortcut tới
  `pythonw C:\đường\dẫn\gesture_control.py` trong thư mục Startup.
- **macOS**: thêm LaunchAgent, hoặc chạy qua Automator.
- **Linux**: thêm systemd user service với `--no-preview`.
