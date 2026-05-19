import argparse
from pathlib import Path
import time
import threading
from typing import Any

import cv2 as _cv2
import numpy as np

cv2: Any = _cv2


WIDTH = 640
HEIGHT = 480


def build_status_text(base_index: int, overlay_index: int | None, message: str | None) -> str:
    overlay_text = str(overlay_index) if overlay_index is not None else "none"
    status = f"base={base_index} overlay={overlay_text}"
    if message:
        status = f"{status} | {message}"
    return status


def find_overlay_image_path(folder: str) -> Path:
    folder_path = Path(folder).expanduser()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Overlay image folder does not exist: {folder_path}")

    image_paths = sorted(
        path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {folder_path}")
    return image_paths[0]


def load_overlay_image(folder: str) -> tuple[Any, str]:
    image_path = find_overlay_image_path(folder)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read overlay image: {image_path}")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return image, str(image_path)


class CameraStream:
    def __init__(
        self,
        index: int,
        backend: str,
        label: str,
        warmup_frames: int,
        retries: int,
        retry_delay: float,
    ) -> None:
        self.index = index
        self.backend = backend
        self.label = label
        self.warmup_frames = warmup_frames
        self.retries = retries
        self.retry_delay = retry_delay
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: Any | None = None
        self._latest_frame: Any | None = None
        self._latest_error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def get_frame(self) -> Any | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_status(self) -> str | None:
        with self._lock:
            return self._latest_error

    def _open_capture(self) -> Any | None:
        backends_to_try = [self.backend] if self.backend == "v4l2" else ["auto", "v4l2"]
        last_reason = "unknown error"

        for candidate in backends_to_try:
            cap = configure_camera(self.index, candidate)
            if not cap.isOpened():
                last_reason = f"failed to open using backend {candidate}"
                cap.release()
                continue

            if warmup_camera(cap, frames=self.warmup_frames, retries=self.retries, retry_delay=self.retry_delay):
                if candidate != self.backend:
                    print(f"Info: {self.label} switched to backend {candidate}")
                return cap

            last_reason = f"opened with backend {candidate} but no frames were received"
            cap.release()

        with self._lock:
            self._latest_error = f"{self.label} index {self.index} is not streaming. Last attempt: {last_reason}."
        return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                cap = self._cap

            if cap is None:
                cap = self._open_capture()
                with self._lock:
                    self._cap = cap
                if cap is None:
                    time.sleep(0.5)
                    continue

            ok, frame = cap.read()
            if ok and frame is not None:
                resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                with self._lock:
                    self._latest_frame = resized
                    self._latest_error = None
                continue

            with self._lock:
                self._latest_error = f"{self.label} index {self.index} stopped delivering frames"
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
            time.sleep(max(0.0, self.retry_delay))


def configure_camera(index: int, backend: str):
    if backend == "v4l2":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def warmup_camera(cap: Any, frames: int, retries: int, retry_delay: float) -> bool:
    for _ in range(max(0, frames)):
        frame_ready = False
        for _ in range(max(1, retries)):
            ok, frame = cap.read()
            if ok and frame is not None:
                frame_ready = True
                break
            time.sleep(max(0.0, retry_delay))
        if not frame_ready:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a live camera feed or overlay one camera on another using OpenCV at 480p."
    )
    parser.add_argument("--base-cam", type=int, default=0, help="Base camera index")
    parser.add_argument("--overlay-cam", type=int, default=1, help="Overlay camera index")
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Show only the base camera feed and skip the overlay camera entirely",
    )
    parser.add_argument(
        "--overlay-image-folder",
        type=str,
        default=None,
        help="Overlay the first supported image found in this folder instead of using an overlay camera",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay strength from 0.0 to 1.0",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "v4l2"],
        default="auto",
        help="Camera backend to use",
    )
    parser.add_argument(
        "--read-retries",
        type=int,
        default=20,
        help="Read attempts per frame before giving up",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.03,
        help="Delay in seconds between read retries",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="Frames to read at startup to stabilize each camera",
    )
    parser.add_argument(
        "--overlay-max-index",
        type=int,
        default=16,
        help="Highest overlay camera index available in the GUI selector",
    )
    args = parser.parse_args()

    alpha = max(0.0, min(1.0, args.alpha))

    base_reader = CameraStream(
        index=args.base_cam,
        backend=args.backend,
        label="Base camera",
        warmup_frames=args.warmup_frames,
        retries=args.read_retries,
        retry_delay=args.retry_delay,
    )
    base_reader.start()

    window_name = "Live Camera Overlay (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    overlay_image: Any | None = None
    overlay_image_label: str | None = None
    overlay_reader: CameraStream | None = None
    current_overlay_index: int | None = None
    requested_overlay_index: int | None = None

    if args.overlay_image_folder:
        overlay_image, overlay_image_label = load_overlay_image(args.overlay_image_folder)
    elif not args.no_overlay:
        initial_overlay_index = max(0, min(args.overlay_cam, args.overlay_max_index))
        overlay_reader = CameraStream(
            index=initial_overlay_index,
            backend=args.backend,
            label="Overlay camera",
            warmup_frames=args.warmup_frames,
            retries=args.read_retries,
            retry_delay=args.retry_delay,
        )
        overlay_reader.start()
        cv2.createTrackbar("OverlayCam", window_name, initial_overlay_index, args.overlay_max_index, lambda _value: None)
        current_overlay_index = initial_overlay_index
        requested_overlay_index = initial_overlay_index

    try:
        while True:
            if overlay_reader is not None:
                selected_overlay_index = cv2.getTrackbarPos("OverlayCam", window_name)
                if selected_overlay_index != requested_overlay_index:
                    requested_overlay_index = selected_overlay_index
                    overlay_reader.stop()
                    overlay_reader = CameraStream(
                        index=selected_overlay_index,
                        backend=args.backend,
                        label="Overlay camera",
                        warmup_frames=args.warmup_frames,
                        retries=args.read_retries,
                        retry_delay=args.retry_delay,
                    )
                    overlay_reader.start()
                    current_overlay_index = selected_overlay_index

            base_frame = base_reader.get_frame()
            overlay_frame = overlay_reader.get_frame() if overlay_reader is not None else overlay_image

            if base_frame is None:
                status = build_status_text(
                    args.base_cam,
                    current_overlay_index,
                    base_reader.get_status(),
                )
                blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    status,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, blank)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if args.no_overlay or overlay_frame is None:
                blended = base_frame.copy()
            else:
                blended = cv2.addWeighted(base_frame, 1.0 - alpha, overlay_frame, alpha, 0)

            if overlay_image_label is not None:
                overlay_status = None
            else:
                overlay_status = overlay_reader.get_status() if overlay_reader is not None else None

            status = build_status_text(args.base_cam, current_overlay_index, overlay_status)
            cv2.putText(
                blended,
                status,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, blended)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if overlay_reader is not None and key == ord("["):
                new_index = max(0, selected_overlay_index - 1)
                cv2.setTrackbarPos("OverlayCam", window_name, new_index)
            if overlay_reader is not None and key == ord("]"):
                new_index = min(args.overlay_max_index, selected_overlay_index + 1)
                cv2.setTrackbarPos("OverlayCam", window_name, new_index)
    finally:
        base_reader.stop()
        if overlay_reader is not None:
            overlay_reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()