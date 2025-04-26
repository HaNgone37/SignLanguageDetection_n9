import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # Thay đổi thành 1 hoặc 2 nếu webcam không hoạt động

# Thiết lập tham số
offset = 20
imgSize = 300

# Đường dẫn thư mục data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

assigned_labels = ["A", "B", "C", "D", "E", "F", "G"]
# assigned_labels = ["H", "I", "J", "K", "L", "M", "N"]
# assigned_labels = ["O", "P", "Q", "R", "S", "T", "U"]
# assigned_labels = ["V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# Tạo thư mục cho các ký hiệu được giao nếu chưa tồn tại
for label in assigned_labels:
    if not os.path.exists(os.path.join(DATA_DIR, label)):
        os.makedirs(os.path.join(DATA_DIR, label))

# Khởi tạo biến
current_label_idx = 0  # Chỉ số của ký hiệu hiện tại trong danh sách được giao
counter = 0

print(f"Bắt đầu thu thập dữ liệu cho: {assigned_labels[current_label_idx]}")
print("Nhấn 's' để chụp ảnh, 'n' để chuyển nhãn tiếp theo, 'q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Chuyển ảnh sang RGB để xử lý với MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmark tay
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy bounding box của tay
            h, w, c = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Cắt vùng tay với offset
            x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
            x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)
            imgCrop = frame[y_min:y_max, x_min:x_max]

            if imgCrop.size == 0:
                continue

            # Tạo ảnh nền trắng
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Điều chỉnh kích thước và đặt vùng tay vào ảnh trắng
            imgCropShape = imgCrop.shape
            h_crop, w_crop, _ = imgCropShape
            aspectRatio = h_crop / w_crop

            if aspectRatio > 1:
                k = imgSize / h_crop
                wCal = int(k * w_crop)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w_crop
                hCal = int(k * h_crop)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Hiển thị ảnh đã cắt và ảnh trắng
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Hiển thị nhãn hiện tại và số ảnh đã chụp
    cv2.putText(frame, f"Label: {assigned_labels[current_label_idx]} | Images: {counter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    # Xử lý phím bấm
    key = cv2.waitKey(1)
    if key == ord("s"):  # Nhấn 's' để chụp ảnh
        counter += 1
        cv2.imwrite(os.path.join(DATA_DIR, assigned_labels[current_label_idx], f"{counter}_{time.time()}.jpg"), imgWhite)
        print(f"Đã lưu ảnh cho {assigned_labels[current_label_idx]} | Tổng: {counter}")

    elif key == ord("n"):  # Nhấn 'n' để chuyển sang nhãn tiếp theo
        current_label_idx = (current_label_idx + 1) % len(assigned_labels)
        counter = 0
        print(f"\nThu thập dữ liệu cho: {assigned_labels[current_label_idx]}")

    elif key == ord("q"):  # Nhấn 'q' để thoát
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()