import json
import requests
from bs4 import BeautifulSoup

# --- 通用配置 ---
# 你的 Cookie 字符串，两个函数都会使用这个
# 在这里粘贴你从浏览器开发者工具中复制的完整 Cookie 字符串
cookie_string = "SESSIONID=ccTe39hgbMoqGAG6x7yxOaIqVyP2jOhuQeCVBUxI21a; JOID=WlsVAU8Ma8C4QhOkbQbMUFkN6KhzYgK1zQRK9gh6CvHMFS7mDUPjv9pFG6VpKx_6astCmjjXfQxa9mBe9z5RU0c=; osd=W1AWBU4NYMO8QxKvbgLNUVIO7KlyaQGxzAVB9Qx7C_rPES_nBkDnvttOGKFoKhT5bspDkTvTfA1R9WRf9jVSV0Y=; _xsrf=EGbVA6NHTaM3dXlCMEiWj9aRBvWl4inW; _zap=34c3bc6c-ebae-4e0d-9a06-5bfb7b8a9548; d_c0=APARO_5-wBmPTvibmi_6NacNH42miN-ERZY=|1735194921; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1752912686; HMACCOUNT=14EFD85132347319; SESSIONID=ejwsJRQgzj2vQ1IcUUFxLKc7JOFE4QuwL8X3i8xxblT; DATE=1752912688356; crystal=U2FsdGVkX19HZsSWXHhQvCmTy0IhrzSuQUWjAmBNS9sgZFRn8/yuh3qnHWWPtqhs6Fx+lpP4dATHFISdyemFNM4nXvmwy0g7ICamPZ7CB/IU+sI4WwbiZS83jULObdwxNXvxwM1lrWOA06h6rgMPFiBn+qiMLtZXpkprUsDW2FNS69MOpODrf6kUIyRL/QXg9IJ08veQmhzEr8G7YQHsdz07b0Wmj/50nQPJAtdug/kh+RlGpYjjx8ALS6jGD2Gq; JOID=VFwUAk3oLjXoL-1wde-Mqg1jG3lkj0FCnGW9JBKSRgKeedw7E954SYct6nVyiBeaIe5OPz1cPqJPMgrHfa5epQs=; osd=U10QBkvvLzHsKepxceuKrQxnH39jjkVGmmK8IBaUQQOafdo8Etp8T4As7nF0jxaeJehJPjlYOKVONg7Beq9aoQ0=; __snaker__id=RQbe8vWX2MrZo7OW; cmci9xde=U2FsdGVkX184glWsyX1mezYXZa3qDzIhjRBe9TQQVEY+LLFTFMDXP9nSs9RkgMzxPSZtU4lOrgeYZefNb02KEA==; pmck9xge=U2FsdGVkX1/T3lTFMBSW1MNHTt55yQ7uWCI2fzMhaBs=; assva6=U2FsdGVkX1803CTSuJgB3mtcX8vMBRJ4mrNXHxoktsA=; assva5=U2FsdGVkX18s7ilkblADh/oGQosbZLgtP7rzonbhC4T0Gf8YWm1GZf8atIJSu1QVH69xU8U+rNeMHgJRf8dEEQ==; vmce9xdq=U2FsdGVkX1+9L5kRV9p5NPBm2ZFxevxsXB601UTW1lO8oB1sywbxX35uCVgDtPFrCRoAhGz6Qw95IDt5HxZKgRgHcwII5jsliZ9AEeEMx0QyMg0NlFbFg2/No39rs8FcREB1wxx4Hg7ZLGq+HBSZ6UbnPSt1xTw3YsTv3I27GXM=; z_c0=2|1:0|10:1752912935|4:z_c0|92:Mi4xemV3ZkR3QUFBQUFBOEJFN19uN0FHU1lBQUFCZ0FsVk5KNkpvYVFCQlUzLUlBR3dQNGcwQkhEOWNXcmsyZ0VFZnJB|55035f8a3b883e94b09b952267d56c00a1a59a98cc43712feb238eb4056739b8; q_c1=e3e3832501fe4a17ba8bf7b5b47f6e60|1752912935000|1752912935000; __zse_ck=004_9njDuYcKusVXiMuD5SZ7LomPNAudnA3idCp0Ig/GY0=5gYSDmi0TNqJ7FjyXoqYArffzF1BhUy=GuO4ecVqJEGHl8QUnfJi3U5WYI/Y1QFhwF7Gz7I7xqjnm05LC1vY2-1G/2qpHnazSH76s+360GqWHjozS9rp18hQPVk+DOgjdq/n5leUgxJ+237tPuguqC5x1a1EjhuQ/RyoAp0+8lSPskVcy16jac+kELW9lwJayh1jscDZfo7NsPnlZfJUWL; gdxidpyhxdE=tdTqBn8olmH5j1Mvzgb2xuBP1JV%5CEOlNCeWjTWLag7PU1kAgwZc0iYoJNXHP4pesZ9c6pt6sCy9XvWRvcCih4H3wlMve85vPDsuwZGH0niB0eBEbOdwyCMBakjZYqhkOUjVurq3zUjP5bbW9MM0av5%2Fc9lNsShTacGcfNJ14hrRJYUyE%3A1752924069740; tst=h; BEC=92a0fca0e2e4d1109c446d0a990ad863; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1752937082"


def fetch_question_and_answers():
    """
    函数一：抓取知乎问题页面，解析问题标题、详情和首个回答。
    """
    print("--- 正在执行：抓取问题和回答 ---")

    # 1. 目标 URL
    url = "https://www.zhihu.com/question/1929927890734145793"

    # 2. 从 fetch 调用中复制过来的 Headers
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        # 自动添加下面这行
        "Cookie": cookie_string
    }

    # --- 开始执行 ---
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        html_content = response.text

        soup = BeautifulSoup(html_content, 'lxml')
        script_tag = soup.find('script', id='js-initialData')

        if not script_tag:
            raise Exception("未找到包含初始数据的 <script> 标签 (id='js-initialData')")

        json_data = json.loads(script_tag.string)
        initial_state = json_data['initialState']

        question_id = list(initial_state['entities']['questions'].keys())[0]
        question_data = initial_state['entities']['questions'][question_id]

        print("=" * 30)
        print("问题标题:", question_data['title'])
        print("问题创建时间:", question_data['created'])
        print("回答数量:", question_data['answerCount'])
        print("=" * 30)

        answers = initial_state['entities']['answers']
        print(f"\n找到了 {len(answers)} 个回答，展示第一个：\n")

        # 只遍历第一个回答进行演示
        if answers:
            first_answer_id = list(answers.keys())[0]
            answer_data = answers[first_answer_id]
            author_name = answer_data['author']['name']
            voteup_count = answer_data['voteupCount']
            answer_html_content = answer_data['content']

            answer_soup = BeautifulSoup(answer_html_content, 'lxml')
            answer_text = answer_soup.get_text(separator='\n', strip=True)

            print(f"--- 回答作者: {author_name} ({voteup_count} 赞) ---")
            print(answer_text[:500] + "...")
            print("\n" + "-" * 20 + "\n")

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except Exception as e:
        print(f"处理数据时发生错误: {e}")


def fetch_answer_comments():
    """
    函数二：抓取指定回答下的评论。
    """
    print("--- 正在执行：抓取回答的评论 ---")

    # 1. 目标 API URL
    url = "https://www.zhihu.com/api/v4/comment_v5/answers/1929939005723620631/root_comment?order_by=score&limit=20&offset="

    # 2. 请求头 (Headers)
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "fetch",
        "x-zse-93": "101_3_3.0",
        "x-zse-96": "2.0_7thZ74Fp8Sj8Oo1APe3bkg4ktxOT6PwcsajKibAmZzzTOBnJGAFYT0kwrgVmc2wN",
        "Referer": "https://www.zhihu.com/question/1929927890734145793",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        # 自动添加下面这行
        "Cookie": cookie_string
    }

    # 3. 发送 GET 请求
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        print("成功获取评论 API 数据，内容如下：")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 错误: {http_err}")
        print(f"响应内容: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"请求发生错误: {req_err}")
    except json.JSONDecodeError:
        print("无法解析响应内容为 JSON 格式，请检查响应内容：")
        print(response.text)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 调用第一个函数
    # fetch_question_and_answers()

    print("\n" * 3)  # 打印一些空行来分隔输出

    # 调用第二个函数
    fetch_answer_comments()