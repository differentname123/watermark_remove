# sqlite_adapter.py
import sqlite3
import json
import threading
import time
import os

# --- 配置 ---
DATABASE_FILE = 'app_data.db'
db_lock = threading.Lock()  # 锁对于多线程下的SQLite写入操作仍然至关重要


def get_db_conn():
    """获取数据库连接。连接是线程安全的，但游标不是，所以每个线程/函数中都应获取自己的连接。"""
    conn = sqlite3.connect(DATABASE_FILE, timeout=10)  # 设置超时
    conn.row_factory = sqlite3.Row  # 让查询结果可以像字典一样通过列名访问
    return conn


def init_db():
    """
    初始化数据库。如果表不存在，则创建它。
    这个函数应该在应用启动时只调用一次。
    """
    with db_lock:
        conn = get_db_conn()
        cursor = conn.cursor()

        # 使用 "IF NOT EXISTS" 避免在表已存在时出错
        # 核心字段：video_id, status, user_name, filename, timestamp
        # 口袋字段：extra_data (存储JSON)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                video_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                user_name TEXT,
                filename TEXT,
                timestamp REAL NOT NULL,
                extra_data TEXT
            )
        ''')

        # 为常用查询字段创建索引，极大提升查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON tasks (status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON tasks (filename)')

        conn.commit()
        conn.close()
        print("数据库初始化完成。")


def _row_to_dict(row):
    """辅助函数：将数据库行对象转换为我们期望的完整任务字典。"""
    if not row:
        return None
    task = dict(row)
    extra_data = json.loads(task.pop('extra_data', '{}') or '{}')  # 弹出并解析JSON
    task.update(extra_data)  # 将JSON内容合并回主字典
    return task


# --- 核心CRUD函数 ---

def upsert_task(task_data):
    """
    插入或更新一个任务。
    :param task_data: 包含任务所有信息的字典。
    """
    video_id = task_data.get('video_id')
    if not video_id:
        raise ValueError("task_data 必须包含一个 'video_id' 键。")

    # 准备核心字段和口袋字段
    core_data = {
        'video_id': video_id,
        'status': task_data.get('status', 'queued'),
        'user_name': task_data.get('user_name') or task_data.get('userName'),  # 兼容旧字段
        'filename': task_data.get('filename'),
        'timestamp': task_data.get('timestamp', time.time()),
    }

    # 将所有不在核心字段列表中的键都放入 extra_data
    core_keys = set(core_data.keys())
    extra_data = {k: v for k, v in task_data.items() if k not in core_keys}

    # 将extra_data转为JSON字符串
    core_data['extra_data'] = json.dumps(extra_data, ensure_ascii=False)

    with db_lock:
        conn = get_db_conn()
        cursor = conn.cursor()
        # 使用 "INSERT OR REPLACE" (SQLite特有) 来实现 Upsert
        columns = ', '.join(core_data.keys())
        placeholders = ', '.join(['?'] * len(core_data))
        sql = f"INSERT OR REPLACE INTO tasks ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(core_data.values()))
        conn.commit()
        conn.close()
    # print(f"数据库操作[Upsert]: video_id={video_id}")


def get_task(video_id):
    """根据 video_id 获取单个任务的完整信息。"""
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks WHERE video_id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


def update_task_fields(video_id, updates_dict):
    """
    一次性更新一个任务的多个字段。
    这个函数会智能地区分核心字段和口袋字段。
    """
    if not updates_dict:
        return

    core_updates = {}
    extra_updates = {}

    # 自动更新时间戳
    updates_dict['timestamp'] = time.time()

    # 定义核心字段列表
    core_keys = ['status', 'user_name', 'filename', 'timestamp']

    # 分离核心更新和额外更新
    for key, value in updates_dict.items():
        if key in core_keys:
            core_updates[key] = value
        else:
            extra_updates[key] = value

    with db_lock:
        conn = get_db_conn()
        cursor = conn.cursor()

        # 步骤1: 更新核心字段 (如果需要)
        if core_updates:
            set_clause = ', '.join([f"{key} = ?" for key in core_updates.keys()])
            sql = f"UPDATE tasks SET {set_clause} WHERE video_id = ?"
            params = list(core_updates.values()) + [video_id]
            cursor.execute(sql, tuple(params))

        # 步骤2: 更新口袋字段 (如果需要)
        if extra_updates:
            # SQLite 3.38.0+ 支持 JSON 函数，但为保证兼容性，我们采用读-改-写模式
            cursor.execute("SELECT extra_data FROM tasks WHERE video_id = ?", (video_id,))
            row = cursor.fetchone()
            if row:
                current_extra = json.loads(row[0] or '{}')
                current_extra.update(extra_updates)
                new_extra_json = json.dumps(current_extra, ensure_ascii=False)
                cursor.execute("UPDATE tasks SET extra_data = ? WHERE video_id = ?", (new_extra_json, video_id))

        conn.commit()
        conn.close()
    # print(f"数据库操作[Update Fields]: video_id={video_id}, 更新了 {list(updates_dict.keys())} 字段")


def get_task_by_filename(filename):
    """根据文件名查找任务。"""
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


def get_tasks_by_status(status_list):
    """根据一个或多个状态，获取所有匹配的任务。"""
    conn = get_db_conn()
    cursor = conn.cursor()
    placeholders = ', '.join(['?'] * len(status_list))
    sql = f"SELECT * FROM tasks WHERE status IN ({placeholders})"
    cursor.execute(sql, tuple(status_list))
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def does_task_exist(video_id):
    """快速检查一个任务是否存在于数据库中。"""
    conn = get_db_conn()
    cursor = conn.cursor()
    # 使用 COUNT(*) 比 SELECT * 更高效
    cursor.execute("SELECT COUNT(*) FROM tasks WHERE video_id = ?", (video_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0