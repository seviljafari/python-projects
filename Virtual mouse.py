import cv2

# ۱. بارگذاری Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ۲. باز کردن وبکم
cap = cv2.VideoCapture(0)

# ۳. شمارنده عکس
img_count = 0

while True:
    # ۴. خواندن فریم
    ret, frame = cap.read()
    if not ret:
        break
    
    # ۵. تبدیل به خاکستری
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ۶. تشخیص چهره
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # ۷. حلقه روی چهره‌ها
    for (x, y, w, h) in faces:
        # رسم مستطیل
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # نوشتن شماره چهره
        cv2.putText(frame, f"Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # ۸. نمایش تعداد چهره‌ها
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # ۹. نمایش راهنما
    cv2.putText(frame, "S: Save | Q: Quit", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # ۱۰. نمایش فریم
    cv2.imshow("Face Detection", frame)
    
    # ۱۱. کلیدها
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        # ذخیره عکس
        img_count += 1
        cv2.imwrite(f"E:/cp-vision/face_{img_count}.jpg", frame)
        print(f"✅ عکس {img_count} ذخیره شد.")

cap.release()
cv2.destroyAllWindows()