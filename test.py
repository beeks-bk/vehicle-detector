import cv2
from ultralytics import YOLO


MIN_CONFIDENCE = 0.55
MAX_AREA_RATIO = 0.35
MAX_WIDTH_RATIO = 0.65
MAX_HEIGHT_RATIO = 0.90
MIN_WIDTH_RATIO = 0.12
MIN_HEIGHT_RATIO = 0.20
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.45
EDGE_MARGIN_RATIO = 0.02


def select_best_detection(result, frame_shape, min_conf=MIN_CONFIDENCE):
    frame_h, frame_w = frame_shape[:2]
    accepted = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area_ratio = (width * height) / float(frame_w * frame_h)
        width_ratio = width / float(frame_w)
        height_ratio = height / float(frame_h)
        aspect_ratio = height / width
        margin_x = frame_w * EDGE_MARGIN_RATIO
        margin_y = frame_h * EDGE_MARGIN_RATIO
        touches_edges = sum(
            [
                x1 <= margin_x,
                y1 <= margin_y,
                x2 >= frame_w - margin_x,
                y2 >= frame_h - margin_y,
            ]
        )
        confidence = float(box.conf[0])
        if confidence < min_conf:
            continue
        if width_ratio < MIN_WIDTH_RATIO or height_ratio < MIN_HEIGHT_RATIO:
            continue
        if width_ratio > MAX_WIDTH_RATIO or height_ratio > MAX_HEIGHT_RATIO:
            continue
        if area_ratio > MAX_AREA_RATIO:
            continue
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue
        if touches_edges >= 2:
            continue
        accepted.append(
            {
                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": confidence,
            }
        )

    if not accepted:
        return None

    return max(accepted, key=lambda item: item["confidence"])


def draw_detection_overlay(frame, detection, label):
    output = frame.copy()
    status_text = "NOT DETECTED"
    status_color = (0, 0, 255)

    if detection is not None:
        x1, y1, x2, y2 = detection["xyxy"]
        conf = detection["confidence"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{label} {conf:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        status_text = "DETECTED"
        status_color = (0, 200, 0)

    cv2.putText(
        output,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        2,
    )
    return output

model = YOLO('best.pt')
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)
    detection = select_best_detection(results[0], frame.shape)
    annotated_frame = draw_detection_overlay(frame, detection, "blue_activa_tn37da6696")
    cv2.imshow("My Vehicle Detector", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
