import cv2
import mediapipe as mp
import numpy as np
import os

# Khởi tạo MediaPipe Hands - cho phép nhận diện 2 tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Hàm đếm số ngón tay đang giơ của 1 bàn tay
def count_fingers_one_hand(hand_landmarks, hand_label):
    """
    hand_label: "Left" hoặc "Right" để xác định tay trái hay phải
    """
    landmarks = hand_landmarks.landmark
    finger_tips = [4, 8, 12, 16, 20]  # Đầu ngón: cái, trỏ, giữa, áp út, út
    finger_pips = [2, 6, 10, 14, 18]   # Khớp giữa
    fingers_up = []
    
    # ===== NGÓN CÁI (xử lý đặc biệt) =====
    # Ngón cái nằm ngang, cần kiểm tra theo trục X
    thumb_tip = landmarks[finger_tips[0]]      # Điểm 4 - đầu ngón cái
    thumb_ip = landmarks[3]                     # Điểm 3 - khớp gần ngón cái
    thumb_mcp = landmarks[2]                    # Điểm 2 - khớp xa ngón cái
    
    # Tay phải: ngón cái giơ lên thì tip.x < ip.x
    # Tay trái: ngón cái giơ lên thì tip.x > ip.x
    if hand_label == "Right":
        # Tay phải - ngón cái bên trái
        if thumb_tip.x < thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    else:
        # Tay trái - ngón cái bên phải
        if thumb_tip.x > thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    
    # ===== 4 NGÓN CÒN LẠI (trỏ, giữa, áp út, út) =====
    # Các ngón này đứng thẳng, kiểm tra theo trục Y
    for i in range(1, 5):
        tip = landmarks[finger_tips[i]]
        pip = landmarks[finger_pips[i]]
        
        # Ngón giơ lên thì tip.y < pip.y (trục Y từ trên xuống)
        if tip.y < pip.y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    
    return sum(fingers_up)

# Load ảnh từ folder images
def load_image_for_number(number):
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in image_extensions:
        image_path = f'images/{number}{ext}'
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img, (600, 600))
                return img
    
    return create_default_image(number)

# Tạo ảnh mặc định
def create_default_image(number, width=600, height=600):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(number)
    font_scale = 15
    thickness = 25
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Màu sắc cho các số từ 0-10
    colors = [
        (100, 100, 100),  # 0 - Xám
        (52, 152, 219),   # 1 - Xanh dương
        (46, 204, 113),   # 2 - Xanh lá
        (241, 196, 15),   # 3 - Vàng
        (155, 89, 182),   # 4 - Tím
        (230, 126, 34),   # 5 - Cam
        (231, 76, 60),    # 6 - Đỏ
        (26, 188, 156),   # 7 - Xanh ngọc
        (243, 156, 18),   # 8 - Cam đậm
        (142, 68, 173),   # 9 - Tím đậm
        (192, 57, 43)     # 10 - Đỏ đậm
    ]
    
    color = colors[number % len(colors)]
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    cv2.putText(img, f"Khong tim thay anh {number}.jpg", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return img

# Pre-load ảnh từ 0-10
print("Đang load ảnh...")
loaded_images = {}
for i in range(11):
    loaded_images[i] = load_image_for_number(i)
    image_path_found = False
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if os.path.exists(f'images/{i}{ext}'):
            print(f"✓ Đã load: images/{i}{ext}")
            image_path_found = True
            break
    if not image_path_found:
        print(f"✗ Không tìm thấy ảnh cho số {i}, dùng ảnh mặc định")

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("LỖI: Không thể mở camera!")
    print("Vui lòng kiểm tra:")
    print("- Camera có được kết nối không?")
    print("- Các ứng dụng khác có đang dùng camera không?")
    exit()

current_total_fingers = 0
display_image = loaded_images[0].copy()

print("\n=== NHẬN DIỆN 2 BÀN TAY ===")
print("Giơ 0-10 ngón tay (2 bàn tay) để hiển thị ảnh tương ứng")
print("Nhấn 'q' để thoát, 'ESC' cũng được")
print("=" * 40)
print("\nĐang mở cửa sổ camera...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể đọc từ camera")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Vẽ khung header
    cv2.rectangle(frame, (0, 0), (640, 70), (50, 50, 50), -1)
    
    total_fingers = 0
    hand_labels = []
    
    if results.multi_hand_landmarks:
        # Đếm tổng số ngón từ tất cả các tay
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Lấy thông tin tay trái hay phải
            hand_label = results.multi_handedness[idx].classification[0].label
            
            # Vẽ bàn tay lên frame với màu khác nhau
            if idx == 0:
                # Tay thứ nhất - màu xanh lá
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                )
            else:
                # Tay thứ hai - màu xanh dương
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
                )
            
            # Đếm ngón của từng tay - TRUYỀN HAND_LABEL VÀO
            fingers = count_fingers_one_hand(hand_landmarks, hand_label)
            total_fingers += fingers
            
            # Tạo nhãn hiển thị (Left -> Trai, Right -> Phai)
            hand_name = "Trai" if hand_label == "Left" else "Phai"
            hand_labels.append(f"{hand_name}: {fingers}")
        
        # Nếu tổng số ngón thay đổi, cập nhật ảnh
        if total_fingers != current_total_fingers:
            current_total_fingers = total_fingers
            display_image = loaded_images[total_fingers].copy()
            print(f"Phát hiện: {total_fingers} ngón tay ({', '.join(hand_labels)})")
        
        # Hiển thị thông tin
        cv2.putText(frame, f"Tong: {total_fingers} ngon", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # Hiển thị chi tiết từng tay
        if len(hand_labels) > 0:
            detail_text = " | ".join(hand_labels)
            cv2.putText(frame, detail_text, (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Chua phat hien ban tay...", (15, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)
        
        # Reset về 0 khi không phát hiện tay
        if current_total_fingers != 0:
            current_total_fingers = 0
            display_image = loaded_images[0].copy()
    
    cv2.imshow('Camera - Nhan dien 2 ban tay', frame)
    cv2.imshow('Hinh anh hien thi', display_image)
    
    # Đưa cửa sổ lên trên (chỉ chạy 1 lần)
    if 'windows_positioned' not in locals():
        cv2.setWindowProperty('Camera - Nhan dien 2 ban tay', cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty('Hinh anh hien thi', cv2.WND_PROP_TOPMOST, 1)
        windows_positioned = True
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nĐã thoát chương trình!")