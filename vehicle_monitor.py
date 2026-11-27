import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from telegram import Bot
from ultralytics import YOLO


# ------------------------------------------------------------
# Project configuration
# These values control how the monitoring system behaves.
# ------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CAMERA_INDEX = 1

MIN_CONFIDENCE = 0.55
MODEL_CONFIDENCE = 0.25

MIN_WIDTH_RATIO = 0.12
MIN_HEIGHT_RATIO = 0.20
MAX_WIDTH_RATIO = 0.80
MAX_HEIGHT_RATIO = 0.95
MAX_AREA_RATIO = 0.60
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.45
EDGE_MARGIN_RATIO = 0.02

STABLE_FRAMES_REQUIRED = 3
MOVE_FRAMES_REQUIRED = 2
MOVE_DISTANCE_THRESHOLD = 50
RETURN_DISTANCE_THRESHOLD = 35
MISSING_SECONDS_THRESHOLD = 5           # first alert: "possible movement detected"
MOVING_CONFIRMED_THRESHOLD = 10         # second alert: "vehicle is moving"
RESET_AFTER_MISSING_SECONDS = 20        # give up and reset to WAITING
ALERT_COOLDOWN_SECONDS = 0

# Set to True to print per-frame state to the console. This is the fastest
# way to see WHY an alert isn't firing (e.g. mode stuck at WAITING because
# stable_frames never reaches STABLE_FRAMES_REQUIRED).
DEBUG = True

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8605979367:AAFupcTfjKawgFPP22RCW05pJ1QK3jrjefE")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1549560572")


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    center: tuple[int, int]


@dataclass
class MonitorState:
    reference_point: Optional[tuple[int, int]] = None
    mode: str = "WAITING"
    stable_frames: int = 0
    moving_frames: int = 0
    missing_since: Optional[float] = None
    last_alert_time: float = 0.0


def create_detector(model_path: str) -> YOLO:
    return YOLO(model_path)


def print_telegram_status() -> None:
    if BOT_TOKEN and CHAT_ID:
        print("Telegram alerts enabled.")
    else:
        print("Telegram alerts disabled: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")


def open_camera(camera_index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return camera


def calculate_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def calculate_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def select_best_detection(result, frame_shape) -> Optional[Detection]:
    """
    Returns the best qualifying detection, or None.

    NOTE: if you're seeing "no alert ever fires", the vehicle may be getting
    filtered out here every frame. Turn on DEBUG to see rejection reasons,
    or temporarily loosen MIN_CONFIDENCE / the ratio filters to confirm.
    """
    frame_height, frame_width = frame_shape[:2]
    best_detection = None
    rejected_reasons = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        confidence = float(box.conf[0])

        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = height / max(1.0, width)
        margin_x = frame_width * EDGE_MARGIN_RATIO
        margin_y = frame_height * EDGE_MARGIN_RATIO
        touches_edges = sum(
            [
                x1 <= margin_x,
                y1 <= margin_y,
                x2 >= frame_width - margin_x,
                y2 >= frame_height - margin_y,
            ]
        )

        if confidence < MIN_CONFIDENCE:
            rejected_reasons.append(f"confidence {confidence:.2f} < {MIN_CONFIDENCE}")
            continue
        if width < frame_width * MIN_WIDTH_RATIO or height < frame_height * MIN_HEIGHT_RATIO:
            rejected_reasons.append("too small")
            continue
        if width > frame_width * MAX_WIDTH_RATIO or height > frame_height * MAX_HEIGHT_RATIO:
            rejected_reasons.append("too large")
            continue
        if area > frame_width * frame_height * MAX_AREA_RATIO:
            rejected_reasons.append("area too large")
            continue
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            rejected_reasons.append(f"aspect ratio {aspect_ratio:.2f} out of range")
            continue
        if touches_edges >= 2:
            rejected_reasons.append("touches too many edges")
            continue

        current_box = (int(x1), int(y1), int(x2), int(y2))
        current_detection = Detection(
            box=current_box,
            confidence=confidence,
            center=calculate_center(current_box),
        )

        if best_detection is None or confidence > best_detection.confidence:
            best_detection = current_detection

    if DEBUG and best_detection is None and rejected_reasons:
        print(f"[DEBUG] All candidate boxes rejected: {rejected_reasons}")

    return best_detection


async def send_telegram_message(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not sent: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
        return

    try:
        await Bot(token=BOT_TOKEN).send_message(chat_id=CHAT_ID, text=message)
        print("Telegram message sent.")
    except Exception as error:
        print(f"Telegram message failed: {error}")


def send_alert_if_allowed(state: MonitorState, message: str) -> None:
    current_time = time.time()
    if current_time - state.last_alert_time < ALERT_COOLDOWN_SECONDS:
        return

    print(f"[ALERT] {message}")
    asyncio.run(send_telegram_message(message))
    state.last_alert_time = current_time


def start_monitoring(state: MonitorState, detection: Detection) -> None:
    state.reference_point = detection.center
    state.mode = "PARKED"
    state.moving_frames = 0
    send_alert_if_allowed(state, "Vehicle parked. Monitoring started.")


def handle_detected_vehicle(
    state: MonitorState,
    detection: Detection,
) -> tuple[Optional[float], float]:
    state.stable_frames += 1
    state.missing_since = None

    if state.reference_point is None and state.stable_frames >= STABLE_FRAMES_REQUIRED:
        start_monitoring(state, detection)

    distance = None
    if state.reference_point is not None:
        distance = calculate_distance(detection.center, state.reference_point)

        if distance > MOVE_DISTANCE_THRESHOLD:
            state.moving_frames += 1
        else:
            state.moving_frames = 0

        if state.moving_frames >= MOVE_FRAMES_REQUIRED and state.mode != "MOVING":
            state.mode = "MOVING"
            send_alert_if_allowed(
                state,
                f"Alert: vehicle is moving. Distance from parked point: {distance:.0f}px",
            )
        elif state.mode in ("MOVING", "SUSPECTED_MOVING") and distance <= RETURN_DISTANCE_THRESHOLD:
            state.reference_point = detection.center
            state.mode = "PARKED"
            state.moving_frames = 0
            send_alert_if_allowed(state, "Vehicle returned and parked successfully.")

    if DEBUG:
        print(
            f"[DEBUG] DETECTED mode={state.mode} stable_frames={state.stable_frames} "
            f"conf={detection.confidence:.2f} distance={distance}"
        )

    return distance, 0.0


def handle_missing_vehicle(state: MonitorState) -> tuple[Optional[float], float]:
    state.stable_frames = 0
    state.moving_frames = 0

    if state.missing_since is None:
        state.missing_since = time.time()

    missing_time = time.time() - state.missing_since

    if state.mode == "PARKED" and missing_time >= MISSING_SECONDS_THRESHOLD:
        state.mode = "SUSPECTED_MOVING"
        send_alert_if_allowed(
            state,
            "Vehicle not detected for 5 seconds. Possible movement detected.",
        )
    elif state.mode == "SUSPECTED_MOVING" and missing_time >= MOVING_CONFIRMED_THRESHOLD:
        state.mode = "MOVING"
        send_alert_if_allowed(
            state,
            "Alert: vehicle is moving.",
        )
    elif state.mode in ("SUSPECTED_MOVING", "MOVING") and missing_time >= RESET_AFTER_MISSING_SECONDS:
        state.reference_point = None
        state.mode = "WAITING"
        state.missing_since = None
        print("Vehicle gone too long. Reset to WAITING.")

    if DEBUG:
        print(f"[DEBUG] MISSING mode={state.mode} missing_time={missing_time:.1f}s")

    return None, missing_time


def draw_overlay(
    frame,
    detection: Optional[Detection],
    state: MonitorState,
    distance: Optional[float],
    missing_time: float,
):
    output = frame.copy()

    status_text = "NOT DETECTED"
    status_color = (0, 0, 255)

    if detection is not None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        status_text = f"DETECTED {detection.confidence:.2f}"
        status_color = (0, 255, 0)

        if state.reference_point is not None:
            cv2.circle(output, state.reference_point, 5, (255, 255, 0), -1)
        cv2.circle(output, detection.center, 5, (255, 255, 255), -1)

    cv2.putText(output, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(output, f"State: {state.mode}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(
        output,
        f"Stable frames: {state.stable_frames}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 0),
        2,
    )

    if distance is not None:
        cv2.putText(
            output,
            f"Distance from parked point: {distance:.1f}px",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    if missing_time > 0:
        cv2.putText(
            output,
            f"Missing time: {missing_time:.1f}s",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

    cv2.putText(
        output,
        "Press Q to quit | Press R to reset",
        (20, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    return output


def reset_monitor(state: MonitorState) -> None:
    state.reference_point = None
    state.mode = "WAITING"
    state.stable_frames = 0
    state.moving_frames = 0
    state.missing_since = None
    print("System reset.")


def main() -> None:
    if "--test-telegram" in sys.argv:
        asyncio.run(send_telegram_message("Test alert from vehicle monitoring system."))
        return

    print_telegram_status()
    model = create_detector(MODEL_PATH)
    camera = open_camera(CAMERA_INDEX)
    state = MonitorState()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Camera frame not received. Exiting.")
                break

            result = model(frame, conf=MODEL_CONFIDENCE)[0]
            detection = select_best_detection(result, frame.shape)

            if detection is not None:
                distance, missing_time = handle_detected_vehicle(state, detection)
            else:
                distance, missing_time = handle_missing_vehicle(state)

            preview = draw_overlay(frame, detection, state, distance, missing_time)
            cv2.imshow("Vehicle Monitor", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_monitor(state)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()