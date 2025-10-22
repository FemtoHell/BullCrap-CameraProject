import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === NHẬN DIỆN ĐỘNG TÁC TAY ===
def detect_hand_gesture(hand_landmarks, hand_label):
    """Nhận diện động tác tay"""
    landmarks = hand_landmarks.landmark
    
    # Kiểm tra từng ngón
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    if hand_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x
    
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    
    fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    count = sum(fingers)
    
    # Các động tác cụ thể
    if count == 1 and index_up:
        return "point_one"  # Chỉ 1 ngón trỏ
    elif count == 5:
        return "open_hand"  # Tất cả ngón
    else:
        return None

def check_hands_cover_face(hand_results, face_results):
    """Kiểm tra 2 tay có che mặt không"""
    if face_results.multi_face_landmarks is None:
        return False
    
    if hand_results.multi_hand_landmarks is None:
        return False
    
    if len(hand_results.multi_hand_landmarks) < 2:
        return False
    
    # Lấy vị trí mặt
    face_landmarks = face_results.multi_face_landmarks[0]
    face_center_x = face_landmarks.landmark[1].x
    face_center_y = face_landmarks.landmark[1].y
    
    # Kiểm tra cả 2 tay gần mặt
    hands_near = 0
    for hand_landmarks in hand_results.multi_hand_landmarks:
        hand_center_x = hand_landmarks.landmark[9].x
        hand_center_y = hand_landmarks.landmark[9].y
        
        distance = np.sqrt((face_center_x - hand_center_x)**2 + (face_center_y - hand_center_y)**2)
        if distance < 0.2:
            hands_near += 1
    
    return hands_near >= 2

# === NHẬN DIỆN BIỂU CẢM MẶT ===
def detect_face_expression(face_landmarks):
    """Nhận diện biểu cảm mặt"""
    if face_landmarks is None:
        return None
    
    landmarks = face_landmarks.landmark
    
    # Tính độ mở miệng
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    
    mouth_height = abs(upper_lip.y - lower_lip.y)
    mouth_width = abs(mouth_left.x - mouth_right.x)
    mouth_ratio = mouth_height / (mouth_width + 0.001)
    
    # Phân loại
    if mouth_ratio > 0.5:
        return "mouth_wide_open"  # Há miệng rất to
    elif mouth_ratio > 0.3:
        return "mouth_open"  # Lè lưỡi / miệng mở vừa
    elif mouth_ratio > 0.15:
        return "smile"  # Cười
    else:
        return "neutral"

# === MAPPING ĐỘNG TÁC → ẢNH CON KHỈ ===
GESTURE_TO_IMAGE = {
    # Động tác 1: Giơ 1 ngón + Cười
    ("point_one", "smile"): 1,
    
    # Động tác 2: Che mặt
    ("cover_face", "any"): 2,
    
    # Động tác 3: Lè lưỡi + Giơ 1 ngón
    ("point_one", "mouth_open"): 3,
    
    # Động tác 4: Lè lưỡi (không cần động tác tay cụ thể)
    ("any", "mouth_open"): 4,
    
    # Động tác 5: Há miệng sốc
    ("any", "mouth_wide_open"): 5,
    
    # Động tác 6: Cười + Giơ 1 ngón (giống 1, có thể là backup)
    ("point_one", "neutral"): 6,
}

# Load ảnh
def load_image_for_number(number):
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in image_extensions:
        image_path = f'images/{number}{ext}'
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return cv2.resize(img, (600, 600))
    return None

# Tạo ảnh mặc định
def create_default_image(width=600, height=600):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (60, 60, 60)
    
    cv2.putText(img, "Lam dong tac de xem con khi!", (30, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(img, "Try: Cuoi, Le luoi, Che mat, Chi 1 ngon...", (30, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    return img

# Pre-load ảnh
print("Đang load ảnh...")
loaded_images = {}

for i in range(1, 7):
    img = load_image_for_number(i)
    if img is not None:
        loaded_images[i] = img
        print(f"✓ Đã load: images/{i}.jpg")
    else:
        print(f"✗ Không tìm thấy ảnh {i}")

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("LỖI: Không thể mở camera!")
    exit()

# Biến hiển thị
current_image = create_default_image()
last_gesture_time = time.time()
stable_gesture_key = None
stable_count = 0

print("\n" + "="*60)
print("=== GƯƠNG PHẢN CHIẾU CON KHỈ ===")
print("="*60)
print("Làm động tác → Con khỉ làm theo!")
print("Thử các động tác:")
print("  • Cười 😊")
print("  • Lè lưỡi 😛")
print("  • Há miệng sốc 😲")
print("  • Che mặt 🙈")
print("  • Giơ 1 ngón trỏ 👆")
print("  • Kết hợp: Cười + Giơ 1 ngón")
print("  • Kết hợp: Lè lưỡi + Giơ 1 ngón")
print("\nNhấn 'q' để thoát")
print("="*60)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Nhận diện
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    
    # Vẽ header
    cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
    
    # Phát hiện động tác hiện tại
    current_hand = None
    current_face = None
    
    # Kiểm tra che mặt trước
    if check_hands_cover_face(hand_results, face_results):
        current_hand = "cover_face"
        current_face = "any"
    else:
        # Phát hiện mặt
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            current_face = detect_face_expression(face_landmarks)
        
        # Phát hiện tay
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                
                # Vẽ bàn tay
                color = (0, 255, 0) if idx == 0 else (255, 0, 0)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )
                
                # Nhận diện động tác
                gesture = detect_hand_gesture(hand_landmarks, hand_label)
                if gesture and current_hand is None:
                    current_hand = gesture
    
    # Tìm ảnh phù hợp
    matched_image_number = None
    gesture_key = None
    
    if current_hand or current_face:
        # Thử match chính xác
        hand_key = current_hand if current_hand else "any"
        face_key = current_face if current_face else "any"
        gesture_key = (hand_key, face_key)
        
        if gesture_key in GESTURE_TO_IMAGE:
            matched_image_number = GESTURE_TO_IMAGE[gesture_key]
        else:
            # Thử match với "any"
            for (h, f), img_num in GESTURE_TO_IMAGE.items():
                if (h == hand_key or h == "any") and (f == face_key or f == "any"):
                    matched_image_number = img_num
                    gesture_key = (h, f)
                    break
    
    # Cập nhật ảnh với độ ổn định
    if gesture_key == stable_gesture_key:
        stable_count += 1
    else:
        stable_gesture_key = gesture_key
        stable_count = 1
    
    # Chỉ đổi ảnh khi động tác ổn định (giữ ~0.3 giây)
    if stable_count > 9 and matched_image_number and matched_image_number in loaded_images:  # 9 frames ≈ 0.3s
        current_image = loaded_images[matched_image_number].copy()
        last_gesture_time = time.time()
    
    # Hiển thị trạng thái
    status_hand = current_hand if current_hand else "Khong co"
    status_face = current_face if current_face else "Khong ro"
    
    gesture_names = {
        "point_one": "Chi 1 ngon",
        "open_hand": "Mo ban tay",
        "cover_face": "Che mat",
        "mouth_open": "Le luoi",
        "mouth_wide_open": "Ha mieng soc",
        "smile": "Cuoi",
        "neutral": "Binh thuong"
    }
    
    display_hand = gesture_names.get(status_hand, status_hand)
    display_face = gesture_names.get(status_face, status_face)
    
    cv2.putText(frame, f"Tay: {display_hand}", (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Mat: {display_face}", (15, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Hiển thị số ảnh đang hiện
    if matched_image_number:
        cv2.putText(frame, f"-> Anh {matched_image_number}", (400, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Reset về mặc định nếu không có động tác trong 3 giây
    if time.time() - last_gesture_time > 3.0 and not (current_hand or current_face):
        current_image = create_default_image()
    
    # Hiển thị
    cv2.imshow('Camera - Ban', frame)
    cv2.imshow('Con khi - Phan chieu', current_image)
    
    if 'windows_positioned' not in locals():
        cv2.setWindowProperty('Camera - Ban', cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty('Con khi - Phan chieu', cv2.WND_PROP_TOPMOST, 1)
        windows_positioned = True
    
    # Phím thoát
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()
print("\nĐã thoát!")