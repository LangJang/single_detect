"""YOLO 目标检测系统 — Gradio UI.

Launch with:  python main.py
"""

from ui.layout import app

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
