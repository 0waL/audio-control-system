"""
Audio Priority Control System — entry point.

Usage:
  python main.py                    # GUI mode (Tkinter)
  python main.py --cli              # CLI mode (headless, logs to stdout)
  python main.py --cli --verbose    # CLI mode with debug logging
  python main.py --list-devices     # List available microphone devices
  python main.py --device 2         # Use specific microphone device index
  python main.py --no-cnn           # Disable CNN, use MFCC heuristics only
  python main.py --cnn-model models/cnn_torch.pt     # Load custom PyTorch CNN
  python main.py --model models/mfcc_classifier.pkl  # Load custom MFCC model
"""

import argparse
import logging
import sys
import signal
import threading
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Audio Priority Control System"
    )
    parser.add_argument("--cli", action="store_true",
                        help="Run in headless CLI mode (no GUI)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available microphone devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone device index (default: system default)")
    parser.add_argument("--no-cnn", action="store_true",
                        help="Disable PyTorch CNN; use MFCC/heuristic classifier only")
    parser.add_argument("--cnn-model", type=str, default=None,
                        help="Path to a trained PyTorch CNN (.pt)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a trained MFCC classifier (.pkl)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Microphone sample rate in Hz (default: 16000)")
    parser.add_argument("--chunk-ms", type=int, default=500,
                        help="Audio chunk duration in ms (default: 500)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom priorities JSON config")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy numba/matplotlib font chatter at debug level
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ── List devices ────────────────────────────────────────────────
    if args.list_devices:
        from src.audio_capture import AudioCapture
        devices = AudioCapture.list_devices()
        if not devices:
            print("오디오 입력 장치를 찾을 수 없습니다.")
            sys.exit(1)
        print("\n사용 가능한 마이크 장치:")
        for d in devices:
            print(f"  [{d['index']}] {d['name']} (채널 수: {d['channels']})")
        sys.exit(0)

    # ── Initialise core components ───────────────────────────────────
    from src.priority_manager import PriorityManager
    from src.engine import AudioControlEngine
    from pathlib import Path

    config_path = Path(args.config) if args.config else None
    pm = PriorityManager(config_path=config_path)
    logger.info(f"Priority config:\n{pm}")

    engine = AudioControlEngine(
        priority_manager=pm,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
        mic_device=args.device,
        mfcc_model_path=args.model,
        cnn_model_path=args.cnn_model,
        use_cnn=not args.no_cnn,
    )

    # ── CLI mode ─────────────────────────────────────────────────────
    if args.cli:
        _run_cli(engine, pm)
        return

    # ── GUI mode ─────────────────────────────────────────────────────
    try:
        from src.ui.settings_ui import SettingsWindow
        window = SettingsWindow(priority_manager=pm, engine=engine)
        window.run()
    except ImportError as e:
        logger.error(f"Tkinter not available ({e}). Use --cli for headless mode.")
        sys.exit(1)


def _run_cli(engine, pm):
    logger = logging.getLogger("cli")

    engine.start()
    logger.info("엔진 시작됨. Ctrl+C로 종료하세요.")

    print("\n" + "─" * 60)
    print("  오디오 우선순위 제어 시스템 — CLI 모드")
    print("  Ctrl+C 로 종료")
    print("─" * 60 + "\n")

    shutdown = threading.Event()

    def _sigint(sig, frame):
        print("\n종료 중 ...")
        shutdown.set()

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not shutdown.is_set():
            status = engine.get_status()
            cat = status["top_category"]
            conf = status["confidence"]
            vol = status["volume"]
            probs = status["probabilities"]

            bar = "█" * int(vol * 20) + "░" * (20 - int(vol * 20))
            prob_str = "  ".join(
                f"{k[:4]}:{v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])
            )
            print(
                f"\r볼륨 [{bar}] {vol:.0%}  |  감지: {cat:12s} {conf:.0%}  |  {prob_str}",
                end="", flush=True,
            )
            time.sleep(0.3)
    finally:
        engine.stop()
        print("\n시스템 종료")


if __name__ == "__main__":
    main()
