# handcontrol — AI Python: Điều khiển máy tính bằng cử chỉ bàn tay

Dùng webcam + MediaPipe nhận diện bàn tay. Bàn tay đi vào ô giữa màn hình rồi vuốt ra ngoài:

| Cử chỉ | Kết quả |
|---|---|
| Tay di chuyển **lên** | Vuốt lên → `Page Up` (hoặc `↑` / cuộn lên) |
| Tay di chuyển **xuống** | Vuốt xuống → `Page Down` (hoặc `↓` / cuộn xuống) |
| Tay di chuyển **qua trái** | Vuốt trái → `Left` |
| Tay di chuyển **qua phải** | Vuốt phải → `Right` |

## Chạy nhanh

```bash
pip install -r requirements.txt      # mediapipe opencv-python numpy pynput
python gesture_control.py --simulate # test thuật toán, không cần webcam
python gesture_control.py            # chạy thật với webcam
```

Các chế độ khác:

```bash
python gesture_control.py --keys arrow     # vuốt = 4 phím mũi tên
python gesture_control.py --mode scroll    # vuốt = cuộn chuột thật
python gesture_control.py --dry-run        # chỉ log, không gửi phím
python gesture_control.py --no-preview     # không hiện cửa sổ camera
```

👉 Hướng dẫn chi tiết (cài đặt, quyền webcam, lỗi thường gặp, tinh chỉnh): **[HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md)**

Thoát: nhấn `q` trên cửa sổ preview, `Ctrl+C`, hoặc giơ 2 tay giữ >1.5 giây.
