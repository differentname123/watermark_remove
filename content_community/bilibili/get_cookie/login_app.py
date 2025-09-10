import os
import time
import uuid
import io
import atexit
import threading
from queue import Queue
from flask import Flask, jsonify, request, send_file, abort
from playwright.sync_api import sync_playwright
from PIL import Image
import numpy as np

# 尝试导入 OpenCV，如果不存在则捕获异常
try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not found. QR code cropping might be less accurate.")


APP_HOST = "0.0.0.0"
APP_PORT = 5000
SESSION_TTL = 60 * 2  # **已修改: 会话生存时间设置为 120 秒**
SAVE_DIR = "./saved_sessions"
LOGIN_URL = "https://passport.bilibili.com/login"
LOGIN_COOKIE_NAMES = {"SESSDATA", "DedeUserID", "bili_jct"}

os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)


# ---- helper: 从 png bytes 自动裁切二维码区域并返回 png bytes ----
def crop_qr_from_png_bytes(png_bytes, padding_ratio=0.12):
    """
    输入：png_bytes（bytes）
    输出：裁切后的 png bytes（若无法检测则返回原始 png_bytes）
    padding_ratio: 在检测到 bbox 后的扩展比例
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return png_bytes

    arr = np.array(img)  # RGB

    # 尝试使用 OpenCV 的 QRCodeDetector（最快 & 准确）
    if cv2: # 只有当 OpenCV 成功导入时才尝试使用
        try:
            img_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            qr = cv2.QRCodeDetector()
            data, points, _ = qr.detectAndDecode(img_cv)
            if points is not None:
                pts = points.squeeze()  # (4,2)
                x_min = int(pts[:, 0].min())
                x_max = int(pts[:, 0].max())
                y_min = int(pts[:, 1].min())
                y_max = int(pts[:, 1].max())
                pad = max(10, int(max(x_max - x_min, y_max - y_min) * padding_ratio))
                x0 = max(0, x_min - pad)
                y0 = max(0, y_min - pad)
                x1 = min(arr.shape[1], x_max + pad)
                y1 = min(arr.shape[0], y_max + pad)
                cropped = img.crop((x0, y0, x1, y1))
                out = io.BytesIO()
                cropped.save(out, format="PNG")
                return out.getvalue()
        except Exception:
            # OpenCV 不可用或 detect 阶段出错 -> 继续后续策略
            pass

        # 如果 OpenCV 存在但 detect 没找到点，用轮廓法尝试找方形（也基于 OpenCV）
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # img_cv from previous block
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
            # 二值化后反色（二维码黑块更好被识别）
            th = cv2.bitwise_not(th)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            best_score = 0
            h_img, w_img = gray.shape
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 1000:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = (w / h) if h > 0 else 0
                    score = (1 - abs(aspect - 1)) * (area / (w_img * h_img))
                    if score > best_score:
                        best_score = score
                        best = (x, y, w, h)
            if best is not None:
                x, y, w, h = best
                pad = max(10, int(max(w, h) * padding_ratio))
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(arr.shape[1], x + w + pad)
                y1 = min(arr.shape[0], y + h + pad)
                cropped = img.crop((x0, y0, x1, y1))
                out = io.BytesIO()
                cropped.save(out, format="PNG")
                return out.getvalue()
        except Exception:
            pass

    # 最后回退：简单启发式裁切页面左中区域（在你当前页面布局里 QR 在左中）
    try:
        w, h = img.size
        fallback_box = (int(w * 0.05), int(h * 0.12), int(w * 0.45), int(h * 0.68))
        cropped = img.crop(fallback_box)
        out = io.BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return png_bytes


class PlaywrightWorker:
    """在专用线程里启动 Playwright + browser 并顺序执行任务（避免跨线程调用问题）。"""

    def __init__(self):
        self.task_q = Queue()
        self._responses = {}  # task_id -> (event, result_or_exception)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        # 所有 Playwright 对象都在这个线程里创建与使用
        self._play = sync_playwright().start()
        self._browser = self._play.chromium.launch(headless=True)
        self._sessions = {}  # session_id -> {'context', 'page', 'created_at', ...}
        try:
            while not self._stopped.is_set():
                try:
                    task = self.task_q.get(timeout=0.5)
                except Exception:
                    # 定期清理过期 session
                    now = time.time()
                    # 迭代副本以避免在删除元素时修改列表
                    expired_sids = [sid for sid, info in list(self._sessions.items()) if
                                    now - info.get("created_at", 0) > SESSION_TTL]
                    for sid in expired_sids:
                        try:
                            if sid in self._sessions:  # 再次检查，防止在循环中被其他操作移除
                                self._sessions[sid]["context"].close()
                                del self._sessions[sid]
                                app.logger.info(f"Session {sid} expired and cleaned up.")
                        except Exception as e:
                            app.logger.warning(f"Error closing expired session {sid}: {e}")
                    continue

                task_id, cmd, payload = task
                event = self._responses[task_id][0]
                try:
                    if cmd == "create_session":
                        sid = payload["session_id"]
                        viewport = payload.get("viewport", {"width": 1280, "height": 900})
                        ctx = self._browser.new_context(viewport=viewport)
                        page = ctx.new_page()
                        # load login page (may redirect)
                        try:
                            page.goto(payload.get("url", LOGIN_URL), timeout=60000)
                        except Exception as e:
                            # 忽略goto可能的超时，让页面的后续截图/检查来决定
                            app.logger.warning(f"Page goto for session {sid} encountered exception: {e}")
                        time.sleep(1.0)  # 给脚本一点时间渲染 QR
                        self._sessions[sid] = {
                            "context": ctx,
                            "page": page,
                            "created_at": time.time(),  # 此时间戳用于 TTL
                            "saved": False,
                            "storage_file": None
                        }
                        res = {"ok": True}
                    elif cmd == "screenshot":
                        sid = payload["session_id"]
                        info = self._sessions.get(sid)
                        if not info:
                            raise RuntimeError("session not found or expired")
                        png = info["page"].screenshot(full_page=True)
                        res = png  # bytes
                    elif cmd == "get_cookies":
                        sid = payload["session_id"]
                        info = self._sessions.get(sid)
                        if not info:
                            raise RuntimeError("session not found or expired")
                        cookies = info["context"].cookies()
                        res = cookies
                    elif cmd == "save_storage":
                        sid = payload["session_id"]
                        filename = payload.get("filename")
                        info = self._sessions.get(sid)
                        if not info:
                            raise RuntimeError("session not found or expired")
                        # 保存并关闭该 context
                        info["context"].storage_state(path=filename)
                        info["saved"] = True
                        info["storage_file"] = filename
                        try:
                            info["context"].close()
                        except Exception as e:
                            app.logger.warning(f"Error closing context after saving for session {sid}: {e}")
                        del self._sessions[sid]  # 成功保存后立即移除会话
                        app.logger.info(f"Session {sid} successfully saved and closed.")
                        res = {"saved": filename}
                    elif cmd == "list_sessions":
                        res = list(self._sessions.keys())
                    elif cmd == "close_all":
                        # 关闭所有 context
                        for sid, info in list(self._sessions.items()):
                            try:
                                info["context"].close()
                            except Exception as e:
                                app.logger.warning(f"Error closing session {sid} during close_all: {e}")
                        self._sessions.clear()
                        res = {"closed": True}
                    else:
                        raise RuntimeError("unknown cmd: " + str(cmd))
                    # send success
                    self._responses[task_id] = (event, ("ok", res))
                except Exception as e:
                    self._responses[task_id] = (event, ("err", repr(e)))
                finally:
                    # notify waiter
                    self._responses[task_id][0].set()
        finally:
            # cleanup playwright on worker thread exit
            try:
                for sid, info in list(self._sessions.items()):
                    try:
                        info["context"].close()
                    except Exception:
                        pass
                self._sessions.clear()
            except Exception:
                pass
            try:
                self._browser.close()
            except Exception:
                pass
            try:
                self._play.stop()
            except Exception:
                pass

    def _enqueue_and_wait(self, cmd, payload=None, timeout=30):
        if payload is None:
            payload = {}
        task_id = str(uuid.uuid4())
        event = threading.Event()
        self._responses[task_id] = (event, None)
        self.task_q.put((task_id, cmd, payload))
        waited = event.wait(timeout)
        _, data = self._responses.pop(task_id, (None, ("err", "no response or worker stopped")))
        if not waited:
            raise TimeoutError("playwright worker timed out while processing task or waiting for response")
        status, payload = data
        if status == "ok":
            return payload
        else:
            raise RuntimeError(payload)

    # 公共方法
    def create_session(self, session_id, url=LOGIN_URL, viewport=None):
        return self._enqueue_and_wait("create_session", {"session_id": session_id, "url": url,
                                                         "viewport": viewport or {"width": 1280, "height": 900}})

    def screenshot(self, session_id):
        return self._enqueue_and_wait("screenshot", {"session_id": session_id})

    def get_cookies(self, session_id):
        return self._enqueue_and_wait("get_cookies", {"session_id": session_id})

    def save_storage(self, session_id, filename):
        return self._enqueue_and_wait("save_storage", {"session_id": session_id, "filename": filename})

    def list_sessions(self):
        return self._enqueue_and_wait("list_sessions", {})

    def close_all(self):
        return self._enqueue_and_wait("close_all", {})

    def stop_worker(self):
        self._stopped.set()
        # put a no-op to wake up the loop
        try:
            self.task_q.put((str(uuid.uuid4()), "close_all", {}))
        except Exception:
            pass
        self._thread.join(timeout=2)


# 单例 worker
pw = PlaywrightWorker()
atexit.register(pw.stop_worker)


# --- Flask routes ---
@app.route("/start_login", methods=["POST"])
def start_login():
    session_id = str(uuid.uuid4())
    try:
        pw.create_session(session_id, url=LOGIN_URL)
    except Exception as e:
        app.logger.error(f"Failed to create session {session_id}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "session_ttl_seconds": SESSION_TTL  # 告知前端会话 TTL
    })


@app.route("/screenshot/<session_id>", methods=["GET"])
def screenshot(session_id):
    try:
        png = pw.screenshot(session_id)  # 从 Playwright worker 得到原始 png bytes
    except RuntimeError as e:  # 捕获会话过期或未找到的错误
        app.logger.warning(f"Screenshot requested for non-existent or expired session {session_id}: {e}")
        return jsonify({"ok": False, "error": "Session not found or expired. Please start a new login."}), 404
    except Exception as e:
        app.logger.error(f"Failed to get screenshot for session {session_id}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

    # 默认启用裁切：如果前端想要原图可以加 ?crop=0
    crop_flag = request.args.get("crop", "1").lower()
    if crop_flag in ("0", "false", "no"):
        return send_file(io.BytesIO(png), mimetype="image/png")

    try:
        cropped_png = crop_qr_from_png_bytes(png)
        # 如果裁切返回的是原图（因为检测失败），也会正常返回
        return send_file(io.BytesIO(cropped_png), mimetype="image/png")
    except Exception as e:
        # 出错时降级返回原始截图
        app.logger.exception(f"QR crop failed for session {session_id}. Returning original screenshot.")
        return send_file(io.BytesIO(png), mimetype="image/png")


@app.route("/check_login", methods=["GET"])
def check_login():
    session_id = request.args.get("session_id")
    account_name = request.args.get("account_name")
    if not session_id:
        abort(400, "missing session_id")
    if not account_name:
        abort(400, "missing account_name")

    try:
        cookies = pw.get_cookies(session_id)
    except RuntimeError as e:  # 捕获会话过期或未找到的错误
        app.logger.warning(f"Check login requested for non-existent or expired session {session_id}: {e}")
        return jsonify({"found": False, "error": "Session not found or expired. Please start a new login."}), 404
    except Exception as e:
        app.logger.error(f"Failed to get cookies for session {session_id}: {e}")
        return jsonify({"found": False, "error": str(e)}), 500

    cookie_names = {c["name"] for c in cookies}
    matched = LOGIN_COOKIE_NAMES.intersection(cookie_names)

    if matched:
        # 使用账号名称来保存 storage 文件
        filename = os.path.join(SAVE_DIR, f"{account_name}.json")
        try:
            pw.save_storage(session_id, filename)
            app.logger.info(f"Session {session_id} for account {account_name} successfully logged in and saved to {filename}.")
            return jsonify({"logged_in": True, "cookies_found": list(matched), "storage_file": filename})
        except RuntimeError as e:
            app.logger.warning(f"Failed to save storage for session {session_id} (might have expired right before save): {e}")
            return jsonify({"logged_in": True, "error": "Failed to save storage (session might have expired).", "cookies_found": list(matched)}), 500
        except Exception as e:
            app.logger.error(f"Failed to save storage for session {session_id} for account {account_name}: {e}")
            return jsonify({"logged_in": True, "error": "failed to save storage: " + str(e), "cookies_found": list(matched)}), 500
    else:
        return jsonify({"logged_in": False, "cookies_present": list(cookie_names)})


@app.route("/verify_status", methods=["GET"])
def verify_status():
    """验证指定账号登录状态的接口"""
    account_name = request.args.get("account_name")
    if not account_name:
        abort(400, "missing account_name")

    # 替换为真实逻辑：检查保存的 Cookie 文件是否存在
    cookie_file = os.path.join(SAVE_DIR, f"{account_name}.json")
    if os.path.exists(cookie_file):
        return jsonify({"logged_in": True, "message": f"账号 '{account_name}' 的Cookie文件存在。"}), 200
    else:
        return jsonify({"logged_in": False, "message": f"账号 '{account_name}' 未登录或Cookie文件不存在。"}), 200


@app.route("/list_sessions", methods=["GET"])
def list_sessions():
    try:
        s = pw.list_sessions()
        return jsonify({"sessions": s})
    except Exception as e:
        app.logger.error(f"Failed to list sessions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return "B站扫码登录服务 (请创建 static/index.html)"


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=False, use_reloader=False, threaded=True)