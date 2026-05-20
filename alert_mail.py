"""Email alert via SMTP (e.g. NetEase 163 / QQ mail).

Usage::

    mailer = AlertMailer(
        smtp_server="smtp.163.com",
        smtp_port=465,
        sender="your@163.com",
        password="授权码",        # NOT login password
        receivers="target@qq.com",
    )
    mailer.send_alert("告警标题", "邮件正文")
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr


class AlertMailer:
    def __init__(
        self,
        smtp_server: str = "smtp.163.com",
        smtp_port: int = 465,
        sender: str = "",
        password: str = "",
        receivers: str = "",
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender.strip()
        self.password = password.strip()
        self.receivers = [r.strip() for r in receivers.replace(";", ",").split(",") if r.strip()]

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return bool(self.smtp_server and self.sender and self.password and self.receivers)

    # ------------------------------------------------------------------
    def send_alert(self, subject: str, body_html: str) -> bool:
        """Send an HTML email. Returns True on success."""
        if not self.is_ready():
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = formataddr(("YOLO 检测系统", self.sender))
        msg["To"] = ", ".join(self.receivers)
        msg["Subject"] = subject
        msg.attach(MIMEText(body_html, "html", "utf-8"))

        try:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=15)
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.receivers, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            # Don't crash the detection pipeline on email failure
            print(f"[AlertMailer] 发送失败: {e}")
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def try_send(
        enabled: bool,
        smtp_server: str, smtp_port: int,
        sender: str, password: str, receivers: str,
        video_name: str, video_path: str,
        frames_processed: int, total_detections: int,
        alert_frames: int, alert_threshold: float,
        alert_weights: tuple[float, float],
        class_counts: dict,
    ) -> None:
        """Conditionally send an alert email when alert_frames > 0."""
        if not enabled:
            return
        mailer = AlertMailer(smtp_server, smtp_port, sender, password, receivers)
        if not mailer.is_ready():
            return
        from datetime import datetime
        import json
        body = AlertMailer.build_body(
            video_name=video_name,
            video_path=video_path,
            processed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frames_processed=frames_processed,
            total_detections=total_detections,
            alert_frames=alert_frames,
            alert_threshold=alert_threshold,
            alert_weights=alert_weights,
            class_distribution=json.dumps(class_counts, ensure_ascii=False),
        )
        mailer.send_alert("目标检测告警", body)

    # ------------------------------------------------------------------
    @staticmethod
    def build_body(
        video_name: str,
        video_path: str,
        processed_at: str,
        frames_processed: int,
        total_detections: int,
        alert_frames: int,
        alert_threshold: float,
        alert_weights: tuple[float, float],
        class_distribution: str,
    ) -> str:
        return f"""\
<h3>目标检测告警</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; font-size:14px;">
  <tr><td><b>视频文件</b></td><td>{video_name}</td></tr>
  <tr><td><b>完整路径</b></td><td>{video_path}</td></tr>
  <tr><td><b>处理时间</b></td><td>{processed_at}</td></tr>
  <tr><td><b>处理帧数</b></td><td>{frames_processed}</td></tr>
  <tr><td><b>检测目标数</b></td><td>{total_detections}</td></tr>
  <tr><td style="color:#cc0000;"><b>告警帧数</b></td>
      <td style="color:#cc0000;">{alert_frames} / {frames_processed}
      ({alert_frames / max(frames_processed, 1) * 100:.1f}%)</td></tr>
  <tr><td><b>告警阈值</b></td><td>{alert_threshold}（数量权重 {alert_weights[0]} / 面积权重 {alert_weights[1]}）</td></tr>
  <tr><td><b>类别分布</b></td><td>{class_distribution}</td></tr>
</table>
<p><i>由 YOLO 目标检测系统自动发送</i></p>"""
