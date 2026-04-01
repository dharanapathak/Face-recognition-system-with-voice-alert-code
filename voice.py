import cv2
import pyttsx3
import time

engine = pyttsx3.init()

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ret, frame1 = camera.read()
ret, frame2 = camera.read()

last_alert_time = 0

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)

        current_time = time.time()

        if current_time - last_alert_time > 5:
            filename = f"alert_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame1)

            # 🎤 Voice Alert
            engine.say("Alert! Motion detected")
            engine.runAndWait()

            last_alert_time = current_time

    cv2.imshow("AI Smart Surveillance System", frame1)

    frame1 = frame2
    ret, frame2 = camera.read()

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()