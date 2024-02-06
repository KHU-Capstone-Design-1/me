#import cv2
import mediapipe as mp

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 노트북 카메라를 사용하여 VideoCapture 초기화
cap = cv2.VideoCapture(0)  # 0 또는 1로 변경하여 노트북에 있는 카메라를 선택

while cap.isOpened():
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        break

    # 프레임을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 감지 수행
    results = hands.process(rgb_frame)

    # 감지된 손이 있을 경우 랜드마크 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 랜드마크의 좌표 얻기
            hand_center = (0, 0)
            for lm_id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                # Calculate hand center
                hand_center = (cx, cy)

            #손 위치에 따른 방향
            if hand_center[0] < w // 3:  
                print("Left turn - 좌회전")
               

            elif hand_center[0] > 2 * w // 3:  
                print("Right turn - 우회전")
              

            elif hand_center[1] < h // 3:  
                print("Forward - 전진")
           

            elif hand_center[1] > 2 * h // 3:  
                print("Backward - 후진")
                

    # 화면에 출력
    cv2.imshow('Hand Tracking', frame)

    # 'ESC' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
