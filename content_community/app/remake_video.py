# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/7/10 1:46
:last_date:
    2025/7/10 1:46
:description:
    重制视频
"""
import os
import shutil
import time
import traceback

import cv2

from LLM.gemini import get_llm_content_gemini_flash_video
from common_utils.common_utils import string_to_object, optimize_subtitle_timing, merge_time_segments, read_json, \
    save_json, fill_time_gaps, time_to_ms, find_file_by_name, merge_time_intervals, map_and_adjust_scenes
from common_utils.ocr.paddle_ocr_utils import find_overall_subtitle_box, find_overall_subtitle_box_target_number
from common_utils.split_audio import separate_with_cli
from common_utils.split_scenes import find_and_split_scenes
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.tts.paddle_speech_demo import synthesize_and_get_duration
from common_utils.video_utils import cover_video_area_gently, add_subtitles_to_video, cover_video_area_simple, \
    re_edit_video_ffmpeg, extract_audio_from_video, cut_audio_segment, cover_video_area_blur, get_video_duration_seconds
from paddlespeech.cli.tts.infer import TTSExecutor

import json

from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video


def get_owner_speech(video_path):
    """
    获取视频中的主人公语音片段。以及相应的语音
    """
    prompt = """
    你是一名专业的音频处理AI和**视频内容策划专家**。你的任务是进行说话人识别、语音转写、文案优化，**并为最终的文稿推荐最合适的背景音乐（BGM）和旁白人声。**
    # 任务背景
    
    -   我是视频创作者，我将为你提供一份带有时间戳的语音转写初稿或音频文件。
    -   内容中混合了我的旁白、其他人的声音、以及外部视频片段的声音。
    
    **# 可选资源 (Reference Materials)**
    
    **你进行推荐决策时，必须从以下列表中进行选择。**
    
    **## 可选背景音乐 (BGM)**
    ```json
    [
      {
        "id": "3e0758d4ab8d5666f0c0e0f4c7f0c9a6.mp4",
        "title": "Улетали птицами гордыми",
        "artist": "NUANHAO/MINGYANG",
        "description": "这是一首节奏舒缓、带有忧郁和怀旧情绪的Lo-fi Hip Hop纯音乐。乐曲以慵懒的鼓点为基础，融合了柔和的电钢琴和合成器音色，营造出一种宁静、放松且略带伤感的氛围。",
        "tags": ["Lo-fi", "节奏舒缓", "氛围感强", "旋律简单重复", "复古", "朦胧感"],
        "suggested_use": "适合用作学习、工作、放松或深夜驾驶时的背景音乐。也常用于Vlog、情感类短视频、或需要营造安静、内省氛围的影视片段中。"
      },
      {
        "id": "6a16b2c0ffb12981709db7cd9dd23557.mp4",
        "title": "Rise",
        "artist": "Epic Music",
        "description": "这是一首情感丰富、极具戏剧性的史诗管弦乐。乐曲以轻柔的钢琴独奏开始，情感细腻而悲伤，随后弦乐和唱诗班的加入逐渐将情绪推向高潮，营造出一种在灾难和悲剧中透露着希望和力量的感觉。",
        "tags": ["史诗", "管弦乐", "情感递进", "旋律优美", "氛围宏大", "电影感", "悲壮", "充满希望"],
        "suggested_use": "广泛用于电影、电视剧中的感人或悲壮场景，尤其适合灾难、战争、历史等题材。也常用于游戏CG、纪录片、励志和致敬类的视频剪辑中，能强烈地烘托气氛，震撼人心。"
      },
      {
        "id": "8dfb680196265fcafe4cc19ce6e75ffe.mp4",
        "title": "Камин (Fireplace)",
        "artist": "Unknown",
        "description": "一首充满情感爆发力的俄语流行歌曲。歌曲由男声演唱，旋律激昂，节奏感强。从一开始的娓娓道来到副歌部分的撕心裂肺，情感层层递进，表达了强烈的失落、痛苦和不甘的情绪。",
        "tags": ["俄语流行", "情感浓烈", "节奏感强", "旋律抓耳", "叙事性", "画面感"],
        "suggested_use": "常用于情感类、故事类的视频剪辑，尤其是在表达人物内心矛盾、分手、决裂等激烈情绪的场景中。在动漫混剪(AMV)、游戏剪辑等二次创作中也颇受欢迎。"
      },
      {
        "id": "428eaba81088bd92cbc5a6a273dbf873.mp4",
        "title": "Wake",
        "artist": "Elzio",
        "description": "一首充满力量和史诗感的电子音乐。乐曲融合了管弦乐元素和强劲的电子节拍，节奏由慢到快，旋律激昂，充满了前进的动力和觉醒的力量感，给人带来振奋和鼓舞的感觉。",
        "tags": ["电子音乐", "史诗感", "节奏强劲", "激昂", "振奋人心", "电子与管弦乐结合"],
        "suggested_use": "非常适合用于游戏集锦、体育赛事、极限运动等高燃场面。也常用于科幻、战争题材的预告片或视频剪辑，以及需要展现宏大场面和科技感的航拍视频中。"
      },
      {
        "id": "86b4723373ab92747961ff027365dbca.mp4",
        "title": "Immortal",
        "artist": "Two Steps From Hell / Thomas Bergersen",
        "description": "这是一首典型的史诗级战争配乐，来自著名的音乐团体Two Steps From Hell。音乐气势磅礴，雄壮有力，以铜管乐和打击乐为主导，弦乐作为铺垫，营造出一种宏大、悲壮、充满史诗感的战争场面，展现了不屈不挠、视死如归的战斗精神。",
        "tags": ["史诗", "战争配乐", "气势磅礴", "悲壮", "震撼人心"],
        "suggested_use": "主要应用于战争题材的电影、电视剧、游戏和纪录片中，尤其适合表现大规模战斗、冲锋、史诗对决等宏大场面，能够极大地增强画面的冲击力和感染力。"
      },
      {
        "id": "0671d099e221faf1b77922fa08ade356.mp4",
        "title": "Call of Silence",
        "artist": "Unknown",
        "description": "一首极其悲伤和忧郁的纯音乐。乐曲主要由钢琴演奏，旋律缓慢而沉重，充满了无尽的遗憾、孤独和压抑感。背景中的人声吟唱和环境音效加深了这种悲凉的氛围，仿佛是对逝去之事的无声呐喊。",
        "tags": ["纯音乐", "氛围忧郁", "情感悲伤", "旋律舒缓", "充满遗憾", "无力感"],
        "suggested_use": "常用于致郁、悲伤、反思等主题的视频中，尤其适合作为动漫《进击的巨人》相关剪辑的背景音乐。能够深刻地表达角色内心的痛苦、绝望和对命运的无奈。"
      },
      {
        "id": "4f7ed367245a6ba525d07f21d4790a25.mp4",
        "title": "Last Reunion",
        "artist": "Peter Roe",
        "description": "这是一首空灵、唯美且充满希望的纯音乐。乐曲以钢琴和弦乐为主，旋律悠扬动听，情感细腻，从开始的平静舒缓，到后半段的逐渐激昂，描绘了一幅宏伟而又美好的画卷，给人一种豁然开朗、涤荡心灵的感觉。",
        "tags": ["纯音乐", "空灵唯美", "情感细腻", "旋律悠扬", "充满希望", "治愈感"],
        "suggested_use": "适合用于风景、旅行、自然风光等视频的背景音乐。也常用于情感独白、励志故事、唯美动漫剪辑等场景，能够营造一种宁静、美好、感人至深的氛围。"
      },
      {
        "id": "9d34a87ec50e5bf577f1405f1475ec7f.mp4",
        "title": "Underground",
        "artist": "Linds, Irling",
        "description": "这是一首节奏感极强、充满力量的Phonk（或Trap）风格电子音乐。乐曲以其标志性的牛铃声、重低音和快速的鼓点为特点，间奏中加入了小提琴旋律，形成一种独特又富有攻击性的听感，充满逆风翻盘的张力。",
        "tags": ["Phonk", "Trap", "节奏感强", "重低音突出", "高能", "富有攻击性", "氛围紧张刺激"],
        "suggested_use": "广泛应用于汽车漂移、极限运动、游戏高能时刻、打斗场景等视频剪辑中。其强烈的节奏和“前奏一响，逆风登场”的属性，使其成为短视频平台中展现技术、力量和反转场面的热门BGM。"
      },
      {
        "id": "6891e8ab04a6a17e2a471017f1642a67.mp4",
        "title": "Lifestyle",
        "artist": "Qora",
        "description": "这是一首充满活力的高能量电子舞曲（EDM）。它具有强烈的驱动节拍、合成器主导的旋律和脉动的低音，营造出一种激动人心和紧张的氛围。音乐的节奏快，结构上包含典型的EDM元素，如逐步增强的构建（build-ups）和激烈的高潮（drops）。",
        "tags": ["EDM", "快节奏", "强劲的贝斯", "合成器主旋律", "高能量", "纯器乐"],
        "suggested_use": "非常适合用于需要强烈动感和活力的视频内容，例如：游戏集锦（特别是射击或赛车类游戏）、极限运动视频、科技产品或大型活动的宣传片、健身训练背景音乐以及派对场景。"
      },
      {
        "id": "3572fb71c23f80e2ce66bc7e4903789f.mp4",
        "title": "Epic Trap Instrumental",
        "artist": "Unknown",
        "description": "这是一首融合了史诗感和现代感的Trap/Hip-Hop器乐。其特点是标志性的Trap鼓点（如快速的踩镲和深沉的808贝斯）与庄严、宏大的旋律相结合。整体情绪严肃、充满力量感，并带有一丝紧张的戏剧性。",
        "tags": ["Trap", "Hip-Hop", "中慢速节奏", "沉重的808贝斯", "史诗感", "严肃", "有冲击力"],
        "suggested_use": "适用于需要营造戏剧性或史诗氛围的视频，例如：军事或重工业展示、电影预告片、具有深刻主题的纪录片、企业宣传片以及需要表现庄重或力量感的场景。"
      },
      {
        "id": "60548e5322831f5b12dd5a00c04c1f7a.mp4",
        "title": "Titan",
        "artist": "Unknown",
        "description": "一首宏伟的电影配乐风格的管弦乐。音乐中大量使用了弦乐、雄壮的铜管乐和雷鸣般的打击乐。整首曲子气势磅礴，逐步将情绪推向高潮，创造出一种壮丽、冒险和英雄主义的感觉。",
        "tags": ["电影配乐", "管弦乐", "戏剧性强", "充满力量感", "情绪层层递进", "史诗感"],
        "suggested_use": "理想的用途包括电影预告片（尤其是动作、奇幻或科幻类型）、游戏宣传视频、体育赛事精彩集锦、企业历史回顾以及任何旨在激发敬畏、英雄气概或宏大场面的视频内容。"
      },
      {
        "id": "1212a7cf29e09ef63e689cb23b1b6fed.mp4",
        "title": "Groovin' King",
        "artist": "Taqumi",
        "description": "这是一首轻快、时髦的放克/爵士风格器乐。突出的萨克斯旋律、富有弹性的贝斯线条和活泼的鼓点共同营造出一种愉快、积极和有趣的氛围。音乐让人感觉轻松愉悦，充满动感。",
        "tags": ["Funk", "爵士", "节奏明快", "萨克斯", "积极", "阳光", "时髦"],
        "suggested_use": "非常适合用于生活方式的Vlog、旅行记录、美食节目、轻松愉快的广告、产品发布会以及任何需要营造时髦、有趣、积极向上氛围的视频。"
      },
      {
        "id": "635687d4416ae6fbb400a09356454347.mp4",
        "title": "MyWay (Lofi)",
        "artist": "Unknown",
        "description": "一首典型的Lo-fi Hip-hop（低保真嘻哈）风格音乐。它的特点是舒缓、稳定的节拍，柔和的合成器音色，以及简单循环的旋律。整体情绪放松、怀旧，带有一点点忧郁感，非常适合营造一个平静、引人思考的氛围。",
        "tags": ["Lo-fi", "Hip-hop", "慢速", "舒缓节奏", "氛围感强", "怀旧", "平静"],
        "suggested_use": "广泛用于学习、工作或放松时的背景音乐播放列表。也常用于Vlog的叙事或谈话片段、城市夜景或自然风光的延时摄影、动画循环背景以及任何追求“Chill”或放松感觉的内容。"
      },
      {
        "id": "b72b5cf111b0bf0c02ef0cfd70f1843d.mp4",
        "title": "Unknown Title",
        "artist": "Unknown",
        "description": "一首充满力量且激昂的电子舞曲。该曲目节奏飞快，鼓点强劲有力，营造出一种紧张、刺激的氛围。通过层层递进的合成器音效和不断加速的节拍，在顶点迎来爆发，非常适合用于展现激烈对抗或追求极致速度的场景。",
        "tags": ["电子舞曲(EDM)", "Hardstyle", "激昂", "高能", "紧张", "富有冲击力", "快节奏"],
        "suggested_use": "适用于游戏高光时刻、赛车视频、极限运动、战斗场景或需要快节奏和强烈冲击感的商业广告。"
      },
      {
        "id": "f6c1b5fca6a5e47ed52485aca4afd5fc.mp4",
        "title": "Unknown Title",
        "artist": "Unknown",
        "description": "这是一首带有怀旧和忧郁色彩的电子音乐。它以稳定的节拍和循环的旋律为基础，营造出一种沉静而引人入胜的氛围。虽然节奏平稳，但其中蕴含的情感使其不仅仅是单调的背景音，而是能引导观众情绪的音乐。",
        "tags": ["电子音乐", "Lo-fi House", "怀旧", "忧郁", "冷静", "放松", "中等节奏"],
        "suggested_use": "非常适合用作城市夜景、个人独白、情感故事讲述、咖啡馆或酒吧环境的背景音乐，以及需要营造特定情境氛围的Vlog。"
      },
      {
        "id": "6660081394559d9a7f2a03c6b0c512ab.mp4",
        "title": "Unknown Title",
        "artist": "Unknown",
        "description": "一首典型的后摇滚风格音乐。乐曲从安静、简约的吉他旋律开始，逐步加入鼓、贝斯和其他乐器，音量和复杂性随之增加，最终形成宏大而富有感染力的音墙。整首曲子充满了情感的张力，从沉思到爆发，具有强烈的叙事感。",
        "tags": ["后摇滚(Post-Rock)", "史诗感", "大气", "情感丰富", "忧郁", "充满希望"],
        "suggested_use": "完美适用于电影预告片、纪录片、自然风光延时摄影、情感浓厚的短片以及需要营造史诗感和宏大叙事氛围的各种视频作品。"
      },
      {
        "id": "4511342007fc8700b5f76e24c5955140.mp4",
        "title": "We Don't Talk Anymore (8D Remix)",
        "artist": "Unknown",
        "description": "这是流行歌曲《We Don't Talk Anymore》的8D环绕音效版本。通过特殊处理，音乐听起来像是在听者周围移动，创造出一种独特的空间感和沉浸感。歌曲本身是一首关于分手的流行情歌，旋律流畅，情感细腻。",
        "tags": ["流行(Pop)", "8D Audio", "沉浸式", "伤感", "浪漫", "现代"],
        "suggested_use": "主要为佩戴耳机个人聆听设计，以体验其独特的环绕效果。可用于需要营造梦幻、内省或情感氛围的视频中，如回忆Vlog或情感短片。"
      },
      {
        "id": "d424af7c445e066e606d98ce0d1534cf.mp4",
        "title": "Unknown Title",
        "artist": "Unknown",
        "description": "这是一首具有攻击性和驾驶感的Phonk音乐。它融合了嘻哈的节奏、沉重的低音和独特的牛铃旋律，创造出一种黑暗、充满力量的街头感。节奏紧凑，充满动感，让人不由自主地跟着摇摆。",
        "tags": ["Phonk", "Trap", "黑暗", "攻击性", "有节奏感", "酷", "快节奏"],
        "suggested_use": "非常适合汽车漂移、街头文化相关的视频、健身房训练、游戏剪辑以及需要营造前卫、 edgy风格的时尚或产品视频。"
      },
      {
        "id": "dd7044ebed9454f08936edb33ae01788.mp4",
        "title": "Unknown Title",
        "artist": "Unknown",
        "description": "一首宏伟、鼓舞人心的管弦乐。它以宽广的弦乐和雄壮的铜管乐为主导，营造出一种史诗般的电影配乐感。音乐充满了积极向上的力量，能激发听众的敬畏感和对未知探索的渴望。",
        "tags": ["电影配乐", "史诗音乐", "鼓舞人心", "宏伟", "大气", "充满希望", "中等节奏"],
        "suggested_use": "适用于电影预告片、壮丽的自然风光航拍、励志演讲、探险纪录片以及任何希望传达宏大、积极和激励人心信息的场合。"
      },
      {
        "id": "db2f73a1b8d7eeb7c4ef70ba6611e463.mp4",
        "title": "Counting Stars (Remix)",
        "artist": "Unknown",
        "description": "流行摇滚歌曲《Counting Stars》的电子混音版。这个版本加快了原曲的节奏，加入了强烈的电子鼓点和合成器效果，使其转变为一首充满活力的舞曲。歌词本身关于梦想和人生，搭配动感的节奏，充满了积极和励志的能量。",
        "tags": ["电子舞曲(EDM)", "流行舞曲(Dance-Pop)", "励志", "充满活力", "乐观", "振奋人心", "快节奏"],
        "suggested_use": "非常适合用于派对场景、体育集锦、健身视频、旅行Vlog以及各种需要营造欢乐、积极向上和充满活力的氛围的场合。"
      },
      {
        "id": "fb2ea8cb19bf45dbbcdb64a7f0b90d77.mp4",
        "title": "Unknown Title",
        "artist": "Unknown",
        "description": "一首带有复古未来感的合成器流行曲。音乐节奏较慢，以深沉的贝斯线条和梦幻般的合成器旋律为特点，营造出一种既忧郁又时尚的氛围。整体感觉很适合在夜晚的城市背景下聆听，带有一种反思和孤独感。",
        "tags": ["合成器流行(Synth-pop)", "Dream Pop", "忧郁", "复古", "梦幻", "冷静", "慢节奏"],
        "suggested_use": "适用于展现城市夜景、角色内心独白、复古风格的短片、时尚秀场背景音乐，或任何需要营造冷静、深沉且略带忧郁情绪的内容。"
      }
    ]
    ```
    
    **## 可选人声 (Voiceover)**
    ```json
        {
          "voice_name": "zh-CN-XiaoxiaoNeural",
          "voice_id_cn": "晓晓",
          "gender": "Female",
          "locale": "zh-CN",
          "style_list": [
            "assistant", "chat", "customerservice", "newscast", "affectionate", "angry",
            "calm", "cheerful", "disgruntled", "fearful", "gentle", "lyrical", "sad", "serious"
          ],
          "voice_description": {
            "vocal_age_cn": "青年女声",
            "timbre_description_cn": "清亮、圆润、字正腔圆、穿透力强。",
            "summary_cn": "标准的AI助手音色，专业且通用。音色明亮清晰，适合各种场景，尤其在需要情感表达时，其丰富的风格库是巨大优势。"
          },
          "application_scenarios": "全能型场景，如新闻播报、智能助手、客服、有声书、广告配音、各类情感对话。",
          "special_notes": "功能最全面的女声，支持多种情感和风格，适用性极广。",
          "rate_info": { "default_rate_description_cn": "适中" }
        },
        {
          "voice_name": "zh-CN-XiaoyiNeural",
          "voice_id_cn": "晓伊",
          "gender": "Female",
          "locale": "zh-CN",
          "style_list": [],
          "voice_description": {
            "vocal_age_cn": "儿童女声",
            "timbre_description_cn": "天真、清脆、稚气、音调偏高。",
            "summary_cn": "一个非常典型的童声，声音纯净无杂质，带有天真可爱的感觉，发音清晰。"
          },
          "application_scenarios": "儿童故事、少儿教育内容、卡通角色配音。",
          "special_notes": "这是一个儿童声音，非常独特，不适用于常规成人内容。",
          "rate_info": { "default_rate_description_cn": "适中" }
        },
        {
          "voice_name": "zh-CN-YunjianNeural",
          "voice_id_cn": "云健",
          "gender": "Male",
          "locale": "zh-CN",
          "style_list": ["narration-professional", "newscast-casual", "sports-commentary"],
          "voice_description": {
            "vocal_age_cn": "中年男声",
            "timbre_description_cn": "浑厚、磁性、共鸣感强、沉稳。",
            "summary_cn": "经典的纪录片或广告旁白音色。声音厚实且富有磁性，给人一种权威、可靠和专业的信赖感。"
          },
          "application_scenarios": "纪录片解说、新闻朗读、企业宣传片、体育赛事评论、正式演讲。",
          "special_notes": "专业的旁白型男声，特别适合正式和需要信赖感的场合。",
          "rate_info": { "default_rate_description_cn": "适中偏慢" }
        },
        {
          "voice_name": "zh-CN-YunxiNeural",
          "voice_id_cn": "云希",
          "gender": "Male",
          "locale": "zh-CN",
          "style_list": [
            "narration-relaxed", "embarrassed", "fearful", "cheerful", "disgruntled",
            "serious", "angry", "sad", "chat"
          ],
          "voice_description": {
            "vocal_age_cn": "青年男声",
            "timbre_description_cn": "清朗、阳光、略带磁性、有活力。",
            "summary_cn": "年轻化的男声，充满活力和亲和力。音色比云健更明亮，比云扬更具情感色彩，适合社交和娱乐场景。"
          },
          "application_scenarios": "聊天机器人、短视频配音、青少年内容、品牌宣传、情感对话。",
          "special_notes": "非常适合需要年轻、活泼感觉的场景，聊天风格自然。",
          "rate_info": { "default_rate_description_cn": "适中偏快" }
        },
        {
          "voice_name": "zh-CN-YunxiaNeural",
          "voice_id_cn": "云霞",
          "gender": "Female",
          "locale": "zh-CN",
          "style_list": [],
          "voice_description": {
            "vocal_age_cn": "青年女声（偏成熟）",
            "timbre_description_cn": "沉稳、柔和、端庄、清晰。",
            "summary_cn": "比晓晓更成熟、正式的播音女声。声音柔和但稳定，没有过多花哨的情感，传递出一种可靠、专业的播报员形象。"
          },
          "application_scenarios": "新闻摘要、专业解说、通知公告、在线教育。",
          "special_notes": "情感风格较少，专注于标准播报场景。",
          "rate_info": { "default_rate_description_cn": "适中" }
        },
        {
          "voice_name": "zh-CN-YunyangNeural",
          "voice_id_cn": "云扬",
          "gender": "Male",
          "locale": "zh-CN",
          "style_list": ["customerservice", "narration-professional", "newscast-casual"],
          "voice_description": {
            "vocal_age_cn": "青年男声",
            "timbre_description_cn": "洪亮、清晰、富有说服力、标准。",
            "summary_cn": "标准的青年播音男声。音色介于云健的浑厚和云希的阳光之间，非常平衡，具有很强的通用性，听感专业且友好。"
          },
          "application_scenarios": "新闻播报、客服指南、在线课程、产品介绍。",
          "special_notes": "一个通用的专业男声，比“云健”稍显年轻和亲切。",
          "rate_info": { "default_rate_description_cn": "适中" }
        },
        {
          "voice_name": "zh-CN-liaoning-XiaobeiNeural",
          "voice_id_cn": "辽宁-晓北",
          "gender": "Female",
          "locale": "zh-CN-liaoning",
          "style_list": [],
          "voice_description": {
            "vocal_age_cn": "青年女声",
            "timbre_description_cn": "爽朗、直接、音调偏高、带有东北口音。",
            "summary_cn": "带有鲜明东北（辽宁）方言特色的女声，语调直率、爽朗，富有生活气息和喜剧感。"
          },
          "application_scenarios": "地方特色内容、搞笑短视频、特定角色配音。",
          "special_notes": "这是一个方言声音，非标准普通话，选择时需特别注意应用场景。",
          "rate_info": { "default_rate_description_cn": "偏快" }
        },
        {
          "voice_name": "zh-CN-shaanxi-XiaoniNeural",
          "voice_id_cn": "陕西-晓妮",
          "gender": "Female",
          "locale": "zh-CN-shaanxi",
          "style_list": [],
          "voice_description": {
            "vocal_age_cn": "青年女声",
            "timbre_description_cn": "质朴、亲切、语调平缓、带有陕西口音。",
            "summary_cn": "带有浓厚陕西（关中）方言特色的女声，音色质朴，语调温和，能唤起西北地区的地域风情。"
          },
          "application_scenarios": "地方文化宣传、特色旅游介绍、方言短剧配音。",
          "special_notes": "这是一个方言声音，非标准普通话，具有很强的地域特色。",
          "rate_info": { "default_rate_description_cn": "适中" }
        }
    ```
    
    # 任务目标
    
    1.  **识别主体**：在所有声音中，只识别并提取出属于“我”（视频创作者）的旁白部分。
    2.  **内容筛选**：完全忽略所有其他人声、背景音、以及非我本人说出的语句。
    3.  **精准对齐**：将我说的每一句旁白，都切分成一个符合自然语义的完整短句。每一句都必须带有精确到毫秒的起始和结束时间戳。
    4.  **验证校准**：如果给出了时间区间，请验证该区间是否准确对应我的声音，并进行必要的校准。
    5.  **文案润色**：
        -   在完成上述步骤后，针对每一句识别出的原始旁白 (`text` 字段)，你需要生成一句新的文本。
        -   **润色要求**：
            -   **保持原意**: 新句子的核心含义必须与原句完全一致。
            -   **长度严格一致**: 新句子的长度（字数）**尽最大可能**与原句保持一致。这是为了确保优化后的文案能精准匹配原视频的时间轴和口型，因此请严格遵守此项规则。
            -   **整体通顺**: 所有润色后的新句子按顺序串联起来，也应能形成一篇通顺、连贯的文稿。
    **6.  智能配乐与配音推荐 (新增目标):**
        -   **在完成所有旁白处理后，综合分析润色后的文稿（`optimizedText` 的集合）的整体主题、情感和风格。**
        -   **根据分析结果，从上方提供的「可选背景音乐」和「可选人声」列表中，分别为这个视频推荐最匹配的一款BGM和一款人声。**
        -   **你需要为你的每一个推荐提供简明扼要的理由。**
    
    # 输出要求
    
    -   **格式**：最终结果必须是一个纯净、合法的 **JSON 对象**。
    -   **内容**：你的回答**必须且只能是**这个 JSON 对象本身，绝对不能包含任何解释性文字、注释、Markdown 标记（例如 ```json）或任何非 JSON 内容。
    -   **结构**：JSON 对象包含**两个**顶级字段：
        -   `transcription`: (Array of Objects) 包含所有旁白句子的数组。数组中每个对象代表我的一句旁白，包含以下**五个**字段：
            -   `id`: (Number) 序号，从 1 开始递增。
            -   `startTime`: (String) 开始时间，格式为 `HH:MM:SS.mmm`。
            -   `endTime`: (String) 结束时间，格式为 `HH:MM:SS.mmm`。
            -   `text`: (String) 旁白**原始**文本内容。
            -   `optimizedText`: (String) 经过润色后的新旁白文本，与原句意义相同、长度相近。
        -   **`recommendations`**: (Object) 包含配乐和配音的推荐及理由。包含以下**两个**字段：
            -   `bgm`: (Object) 背景音乐推荐，包含 `id`, `title`, 和 `reason` 三个字段。
            -   `voice`: (Object) 人声推荐，包含 `voice_name`, `voice_id_cn`, `style` (如果适用), 和 `reason` 四个字段。
    
    注意时间戳一定要是精确到毫秒的格式，且必须严格遵守 `HH:MM:SS.mmm` 的格式。
    
    # JSON 格式示例
    
    ```json
    {
      "transcription": [
        {
          "id": 1,
          "startTime": "00:00:03.125",
          "endTime": "00:00:05.890",
          "text": "今天，我们来回顾一段沉重的历史。",
          "optimizedText": "今天，我们来回望一段沉重的历史。"
        },
        {
          "id": 2,
          "startTime": "00:00:07.500",
          "endTime": "00:00:11.200",
          "text": "在那场巨大的灾难中，人们失去了家园，但从未放弃希望。",
          "optimizedText": "那场巨大的灾难里，人们虽失去家园，却从未放弃希望。"
        }
      ],
      "recommendations": {
        "bgm": {
          "id": "6a16b2c0ffb12981709db7cd9dd23557.mp4",
          "title": "Rise",
          "reason": "文稿整体基调悲壮且蕴含力量，与 'Rise' 从悲伤到充满希望的情感递进和宏大氛围高度契合，能有效烘托主题。"
        },
        "voice": {
          "voice_name": "zh-CN-YunjianNeural",
          "voice_id_cn": "云健",
          "style": "narration-professional",
          "reason": "文稿内容严肃、深刻，需要权威且沉稳的声音来讲述。'云健' 浑厚磁性的音色和专业的旁白风格是最佳选择。"
        }
      }
    }
    """
    base_name = os.path.basename(video_path)
    count = 0
    while True:
        count += 1
        if count > 3:
            print("重试次数超过3次，退出程序。")
            return []
        print("正在生成和优化字幕...")
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)

        # 步骤 3: 优化字幕计时
        optimized_subtitles = optimize_subtitle_timing(result['transcription'])
        result['transcription'] = optimized_subtitles

        # 检查是否存在不正常的字幕时长（小于0或大于10秒）
        if any(not (0 <= subtitle.get('duration', 0) <= 20) for subtitle in optimized_subtitles):
            print(f"检测到无效的字幕时长（小于0或大于20秒）：{optimized_subtitles}，将在2秒后重试...")
            continue  # 如果存在异常值，则跳过本次循环的剩余部分，重新开始

        else:
            break  # 成功，跳出 while 循环

    # 步骤 6: 返回最终的、验证过的结果
    return result


def get_owner_speech_pure(video_path):
    """
    获取视频中的主人公语音片段。以及相应的语音
    """
    prompt = """
    你是一名专业的音频处理AI，任务是进行说话人识别、语音转写**和文案优化**。

    # 任务背景
    - 我是视频创作者，我将为你提供一份带有时间戳的语音转写初稿或音频文件。
    - 内容中混合了我的旁白、其他人的声音、以及外部视频片段的声音。

    # 任务目标
    1.  **识别主体**：在所有声音中，只识别并提取出属于“我”（视频创作者）的旁白部分。
    2.  **内容筛选**：完全忽略所有其他人声、背景音、以及非我本人说出的语句。
    3.  **精准对齐**：将我说的每一句旁白，都切分成一个符合自然语义的完整短句。每一句都必须带有精确到毫秒的起始和结束时间戳。
    4.  **验证校准**：如果给出了时间区间，请验证该区间是否准确对应我的声音，并进行必要的校准。
    **5.  文案润色 (新增目标):**
        -   **在完成上述步骤后，针对每一句识别出的原始旁白 (`text` 字段)，你需要生成一句新的文本。**
        -   **润色要求：**
            -   **保持原意**: 新句子的核心含义必须与原句完全一致。
            -   **长度严格一致**: **此为关键要求。** 新句子的长度（字数）**必须尽最大可能**与原句保持一致,或者少于原句子，最不希望大于原句子。这是为了确保优化后的文案能精准匹配原视频的时间轴和口型，因此请严格遵守此项规则。
            -   **整体通顺**: 所有润色后的新句子按顺序串联起来，也应能形成一篇通顺、连贯的文稿。

    # 输出要求
    - **格式**：最终结果必须是一个纯净、合法的 JSON 数组 (`Array of Objects`)。
    - **内容**：你的回答**必须且只能是**这个 JSON 数组本身，绝对不能包含任何解释性文字、注释、Markdown 标记（例如 ```json）或任何非 JSON 内容。
    - **结构**：数组中的每个对象代表我的一句旁白，包含以下**五个**字段：
        - `id`: (Number) 序号，从 1 开始递增。
        - `startTime`: (String) 开始时间，格式为 `HH:MM:SS.mmm`。
        - `endTime`: (String) 结束时间，格式为 `HH:MM:SS.mmm`。
        - `text`: (String) 旁白**原始**文本内容。
        - **`optimizedText`: (String) 经过润色后的新旁白文本，与原句意义相同、长度相近。**
    注意时间戳一定要是精确到毫秒的格式，且必须严格遵守 `HH:MM:SS.mmm` 的格式。
    # JSON 格式示例
    ```json
    [
      {
        "id": 1,
        "startTime": "00:00:03.125",
        "endTime": "00:00:05.890",
        "text": "欢迎来到我的视频。",
        "optimizedText": "欢迎来到我的频道。"
      },
      {
        "id": 2,
        "startTime": "00:00:07.500",
        "endTime": "00:00:10.000",
        "text": "今天我们来聊一个重要话题。",
        "optimizedText": "这次我们要谈一个核心要点。"
      }
    ]
    """
    count = 0
    while True:
        count += 1
        if count > 3:
            print("重试次数超过3次，退出程序。")
            return []
        print("正在生成和优化字幕...")
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)

        # 步骤 3: 优化字幕计时
        optimized_subtitles = optimize_subtitle_timing(result)
        result = optimized_subtitles

        # 检查是否存在不正常的字幕时长（小于0或大于10秒）
        if any(not (0 <= subtitle.get('duration', 0) <= 20) for subtitle in optimized_subtitles):
            print(f"检测到无效的字幕时长（小于0或大于20秒）：{optimized_subtitles}，将在2秒后重试...")
            continue  # 如果存在异常值，则跳过本次循环的剩余部分，重新开始

        else:
            break  # 成功，跳出 while 循环

    # 步骤 6: 返回最终的、验证过的结果
    return result

def cover_subtitle(video_path, output_path, top_left, bottom_right):
    """
    覆盖视频中的字幕
    """
    start_time = time.time()
    try:

        cover_video_area_blur(
            video_path=video_path,
            output_path=output_path,
            top_left=top_left,
            bottom_right=bottom_right
        )
    except Exception as e:
        print(f"覆盖字幕区域失败: {e} 尝试使用备用方法...")
        cover_video_area_simple(
            video_path=video_path,
            output_path=output_path,
            top_left=top_left,
            bottom_right=bottom_right
        )
        return
    print(f"覆盖字幕区域完成，输出文件: {output_path} 耗时: {time.time() - start_time:.2f} 秒")

def gen_new_audio(optimized_subtitles,voice_name="zh-CN-YunjianNeural",output_dir='output_audio'):
    """
    生成语音文件，并更新字幕信息。

    默认使用第二种方式（如Azure TTS）进行语音合成。如果合成失败（返回时长为0.0），
    则自动切换到第一种备用方式（如本地TTS）重试。

    Args:
        optimized_subtitles (list): 包含字幕信息的列表，每个元素是一个字典。

    Returns:
        list: 更新了 'outputPath' 和 'trimmedDuration' 键的字幕列表。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tts_engine_backup = TTSExecutor()

    for subtitle in optimized_subtitles:
        output_file = os.path.join(output_dir, f"{subtitle['id']}.wav")
        text_to_speak = subtitle['optimizedText']

        print(f"\n--- [字幕 {subtitle['id']}] 正在处理: '{text_to_speak}' ---")

        audio_length = generate_audio_and_get_duration_sync(
            text=text_to_speak,
            output_filename=output_file,
            voice_name=voice_name
        )

        # 检查方式二是否成功，如果不成功 (audio_length为0)，则切换到方式一
        if audio_length == 0.0:
            print(f"    [!] 默认方式生成失败，返回时长为 0.0。")
            print(f"    --> 切换到备用方式 (方式一) 重试...")

            audio_length = synthesize_and_get_duration(
                tts_executor=tts_engine_backup,
                text=text_to_speak,
                output_path=output_file
            )

        # 更新字幕信息
        subtitle['outputPath'] = output_file
        subtitle['trimmedDuration'] = audio_length

        # 打印最终结果
        if audio_length > 0.0:
            print(f"<-- [字幕 {subtitle['id']}] 生成成功！最终音频时长: {audio_length:.2f} 秒")
        else:
            print(f"<-- [字幕 {subtitle['id']}] 生成失败！两种方式都无法生成有效音频。")

    # 保存优化并更新后的字幕到文件
    print("\n所有音频处理完成，正在保存结果到 'optimized_subtitles.json'...")
    with open('optimized_subtitles.json', 'w', encoding='utf-8') as f:
        json.dump(optimized_subtitles, f, ensure_ascii=False, indent=4)
    print("文件保存成功！")

    # 直接返回内存中已更新的列表，无需重新读取文件
    return optimized_subtitles

def add_subtitle(input_video, subtitle_data, output_with_subtitles, bottom_margin, font_size, fixed_rect):
    try:
        # 尝试查找一个常见的系统字体
        font_file_path = ""
        if os.name == 'nt':  # Windows
            font_file_path = 'C:/Windows/Fonts/simhei.ttf'
            if not os.path.exists(font_file_path):
                font_file_path = 'C:/Windows/Fonts/msyh.ttc'
        elif os.name == 'posix':  # macOS or Linux
            if os.path.exists('/System/Library/Fonts/PingFang.ttc'):
                font_file_path = '/System/Library/Fonts/PingFang.ttc'  # macOS
            else:
                # 简单的Linux字体查找
                common_linux_fonts = [
                    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                ]
                for font in common_linux_fonts:
                    if os.path.exists(font):
                        font_file_path = font
                        break

        if not font_file_path or not os.path.exists(font_file_path):
            raise FileNotFoundError("未能自动找到合适的系统字体。")

        print(f"自动检测到字体: {font_file_path}")

        # 4. 调用函数
        add_subtitles_to_video(
            video_path=input_video,
            subtitles_info=subtitle_data,
            output_path=output_with_subtitles,
            font_path=font_file_path,
            font_size=font_size,
            bottom_margin=bottom_margin,
            fixed_rect=fixed_rect
        )

    except (FileNotFoundError, ValueError) as err:
        print(f"[主程序错误] 操作失败: {err}")
        print("\n[提示] 请确保：")
        print("1. `test.mp4` 文件存在于脚本相同目录下。")
        print("2. 你的系统中安装了 ffmpeg 并已添加到环境变量(PATH)。")
        print("3. 如果自动字体检测失败，请在代码中手动指定一个有效的中文字体路径。")


def adjust_subtitle_box(video_path: str, final_box: list[list[int, int]]):
    """
    将字幕框左右边距至少设为视频宽度的 10%，
    但如果原框更宽就不再缩窄它。

    参数:
        video_path: 视频文件路径
        final_box: 原始字幕框，格式 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

    返回:
        (top_left, bottom_right)：调整后的左上角和右下角坐标
    """
    # 1. 打开视频，获取分辨率
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2. 阈值：左右各保留 10%
    thresh_left  = int(width * 0.1)
    thresh_right = int(width * 0.9)

    # 3. 原始框的最小/最大 x，和最小/最大 y
    xs = [pt[0] for pt in final_box]
    ys = [pt[1] for pt in final_box]
    orig_x_min = min(xs)
    orig_x_max = max(xs)
    y_top      = min(ys)
    y_bottom   = max(ys)

    # 4. 如果原框比 阈值 宽度小则扩张，否则保留
    new_x_left  = min(orig_x_min, thresh_left)
    new_x_right = max(orig_x_max, thresh_right)

    # 5. 构造返回值
    top_left     = [new_x_left,  y_top]
    bottom_right = [new_x_right, y_bottom]

    return top_left, bottom_right, width, height



def gen_cut_suggestion(video_path):
    """
    生成剪辑的建议，交换场景顺序或者删除场景。
    """
    try:
        scene_info_dict = find_and_split_scenes(video_path)
        if not scene_info_dict:
            print("未能成功获取视频场景信息。", scene_info_dict)
            return
        prompt = """# 角色
                    你是一位拥有十年以上经验的**资深视频剪辑总监**和**首席社交媒体内容策略师**。你不仅精通抖音、Bilibili、YouTube Shorts的算法和用户心理，更重要的是，你是一位**务实的创作者**，深刻理解每一次剪辑都意味着时间成本。你的决策冷静、精准，始终追求“投入产出比”最高的神级剪辑。
                    
                    # 核心原则
                    在开始任何分析之前，请将以下原则作为你思考的基石：
                    
                    1.  **叙事逻辑优先 (Narrative First)**：视频的流畅性和逻辑连贯性是基础。任何调整都不能破坏故事的内在逻辑或观众的理解流畅度。
                    2.  **保留是默认选项 (Keep is the Default)**：尊重原始素材的创作意图。不要为了调整而调整。如果一个场景没有严重问题，就应该保留。
                    3.  **高门槛调整原则 (High Bar for Changes)**：
                        *   **删除 (Delete)**：必须有充分理由，例如内容完全冗余、质量严重低下或明显偏离主题。
                        *   **重排 (Reorder)**：这是最高成本的操作，必须慎之又慎。**只有当重排能带来压倒性的优势时（例如，创造出无法替代的“黄金三秒”钩子，或解决了致命的叙事缺陷），才予以考虑。** 你的理由必须极具说服力。
                    4.  **效果是唯一标准 (Impact is Everything)**：所有决策的唯一目标是让最终成片在**观众吸引力、叙事流畅性、信息价值和传播潜力**上获得**显著提升**。微小的、可有可无的优化不是你的追求。
                    5.  如果该场景中有不相关的内容（如广告 推广 甚至是关于我的声明或者介绍）不管是不是我推荐的，那么这个场景都应该被删除
                    6. 而且要尽量不要产生太多的剪切点，因为会导致难度大大提示
                    
                    # 任务指令
                    1.  **整体理解与诊断 (Holistic Understanding & Diagnosis)**：
                        *   首先，快速看完所有场景描述，总结出视频的**核心价值主张**（即“观众为什么要看这个视频？”）。
                        *   识别出整个视频中最具潜力的**“黄金时刻”或“高光片段”**。这是你后续决策的关键锚点。
                    
                    2.  **逐一评估 (Scene-by-Scene Evaluation)**：
                        *   结合你对整体的理解，独立评估每个原始场景。评估维度包括：
                            *   **信息密度**：是否传递了关键信息？
                            *   **视觉冲击力**：画面是否吸引人？
                            *   **情绪价值**：能否引发观众的情绪（好奇、共鸣、兴奋、爽感等）？
                            *   **叙事功能**：在故事中扮演什么角色（开端、发展、高潮、结尾、铺垫、转折）？
                            *   **冗余性**：是否拖沓、重复或可被更好的场景替代？
                    
                    3.  **制定剪辑策略 (Formulate the Editing Strategy)**：
                        *   严格遵循上述**【核心原则】**，结合你的评估，构建最终剪辑方案。
                        *   对于每一个决策（保留、重排、删除），在`reasoning`中清晰阐述你的思考过程，特别是要体现你的**审慎和对效果的追求**。例如，解释为什么保留是当前最佳选择，或者阐述一个重排建议为何能带来“压倒性优势”。
                    
                    4.  **生成最终方案 (Generate Final Plan)**：
                        *   将你的决策结果以纯JSON格式输出。
                    
                    # 输出要求
                    *   **严格的JSON格式**：你的输出必须是**一个完整且格式正确的JSON对象**，不能包含任何JSON格式之外的标记、注释、代码块标识（如 ```json ... ```）或任何解释性文本。
                    *   **内容结构**：JSON对象必须包含以下三个顶级键：`overall_strategy`, `final_cut_sequence`, `deleted_scenes`。
                    
                    ---
                    ### **JSON输出格式定义与示例**
                    
                    ```json
                    {
                      "overall_strategy": "（这里是你基于【核心原则】和【整体诊断】得出的顶层策略。例如：原始顺序的叙事逻辑清晰，核心价值突出，仅需删除一个冗余场景来加快节奏，无需进行高成本的重排。）",
                      "final_cut_sequence": [
                        {
                          "scene_id": "场景1",
                          "scene_desc": "（场景的简短描述）",
                          "reasoning": "（你的决策理由。例如：作为视频的自然开端，有效建立情境，逻辑清晰，是最佳的起始点，无需调整。）"
                        },
                        {
                          "scene_id": "场景3",
                          "scene_desc": "（场景的简短描述）",
                          "reasoning": "（例如：这是视频的‘高光时刻’，情绪价值最高，紧随场景1能快速抓住用户，保留其在故事发展中的位置可确保叙事连贯性。）"
                        }
                      ],
                      "deleted_scenes": [
                        {
                          "scene_id": "场景2",
                          "scene_desc": "（场景的简短描述）",
                          "reasoning": "（你的决策理由。例如：此场景与场景3内容高度重叠，且信息密度较低，属于明显冗余。删除后能让叙事流直接从情境建立进入高光时刻，节奏更紧凑。）"
                        }
                      ]
                    }
                    ```
                    
                    **原始场景分割信息如下**:

        """
        prompt = f"{prompt}\n{scene_info_dict}"
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)
        # 增加原始的时间段到result
        final_cut_sequence = result.get('final_cut_sequence', [])
        deleted_scenes = result.get('deleted_scenes', [])
        for scene in final_cut_sequence:
            scene_id = scene.get('scene_id')
            if scene_id:
                # 在场景信息中添加原始时间段
                time_list = scene_info_dict.get(scene_id, [])
                scene['original_start_time'] = time_list[0]
                scene['original_end_time'] = time_list[1]
        for scene in deleted_scenes:
            scene_id = scene.get('scene_id')
            if scene_id:
                # 在删除的场景信息中添加原始时间段
                time_list = scene_info_dict.get(scene_id, [])
                scene['original_start_time'] = time_list[0]
                scene['original_end_time'] = time_list[1]
        return result
    except Exception as e:
        traceback.print_exc()
        return None


def gen_cut_suggestion_with_scene(video_path, scene_info_dict):
    """
    生成剪辑的建议，交换场景顺序或者删除场景。
    """
    try:
        prompt = """**角色:**
                    你是一个专业的视频剪辑师和内容优化专家。你擅长通过分析视频画面和文案，提出能显著提升视频质量的剪辑和文案优化方案。
                    
                    **任务:**
                    将我提供一个视频文件以及一份详细的场景信息JSON。你的任务是严格按照以下步骤，对我提供的视频进行深度分析，并以JSON格式返回一个完整的优化方案。
                    
                    **工作流程与要求:**
                    
                    **第一步：深度场景理解 (Foundation of Everything)**
                    这是所有后续操作的基础，请务C必深入执行。
                    *   **综合分析:** 针对我提供的每一个场景（`scene`），你必须将视频内容（在指定的 `time_range` 内）与我提供的文案（`full_texts`）紧密结合起来进行理解。
                    *   **画面解读:** 仔细观察每个场景的视觉元素、动态变化、情感基调和核心信息。
                    *   **文案关联:** 将画面内容与`full_texts`进行关联分析。思考文案是如何解释、补充或升华画面的。
                    *   **无文案场景:** 如果一个场景的 `full_texts` 为空字符串，这不代表它没有内容或不重要。你必须更加仔细地分析其视觉语言，理解它可能扮演的角色，例如：作为情绪过渡、信息停顿、视觉冲击，或是为后续内容铺垫。请务必在你的分析中体现出对这类场景的理解。
                    
                    **第二步：移除个人信息场景 (Privacy Guard)**
                    在完成深度理解后，审查所有场景。
                    *   **识别并删除:** 立即识别并删除任何包含我的个人身份信息（如Logo、个人头像、姓名、联系方式等）的场景。这是强制性要求。
                    
                    **第三步：智能剪辑建议 (Smart Re-cutting)**
                    基于你对内容的深刻理解，提出以“提升视频质量”为唯一目标的剪辑建议。
                    *   **评估叙事流:** 分析当前场景顺序的逻辑性、节奏感和情感曲线。
                    *   **提出优化方案:** 你只被允许进行两种操作：**删除场景**或**调整场景顺序**。
                        *   **删除:** 只有当某个场景内容空洞、冗余，或与主题关联性弱，删除后能使视频更紧凑、主题更突出时，才建议删除。
                        *   **重排:** 只有当调整顺序能创造更强的逻辑递进、戏剧冲突或情感共鸣，从而显著提升观看体验时，才建议重排。
                    *   **避免无意义的改动:** 我非常反感为了改而改的微小调整。如果原始顺序已经足够好，或者任何改动带来的提升都微不足道，请明确指出应保留原顺序，并解释原因。减少不必要的剪辑工作量是一个重要的考量点。
                    
                    **第四步：文案优化 (Copywriting Enhancement)**
                    根据你最终确定的场景顺序 (`final_cut_sequence`)，对每个保留下来的场景的文案进行优化。
                    *   **目标:** 新文案的目标是最大化地提升视频的吸引力、清晰度和影响力。它可以是让表达更精炼、更具感染力，或是更好地与画面配合。
                    *   **约束条件:**
                        1.  **内容焕新:** 调整后的文案（`adjusted_texts`）**不能**与原始文案（`full_texts`）一模一样。
                        2.  **字数对齐:** 新文案的字数必须与原文案的字数**大致相等**。这是为了确保优化后的文案能够匹配场景固定的时长，避免音画不同步的问题。不允许大幅增加或缩减字数，字数差值范围必须在20%以内。
                    
                    **第五步：格式化输出 (Strict JSON Output)**
                    你的最终输出**必须**是一个完整的、格式正确的JSON对象，不能包含任何额外的解释性文字或标记。请严格遵循我提供的示例结构。
                    
                    ### **期望的输出 `JSON` 格式 (必须严格遵守):**
                    
                    ```json
                    {
                      "overall_strategy": "在这里对你最终剪辑策略进行一个高度概括的说明。解释为什么你选择保留/调整顺序，以及这个策略如何服务于“提升视频质量”的最终目标。",
                      "final_cut_sequence": [
                        {
                          "scene_id": "场景1",
                          "scene_desc": "在这里用一句话精准描述这个场景的核心内容或作用。",
                          "reasoning": "解释为什么这个场景被保留在当前位置。它在新的叙事结构中扮演什么角色。",
                          "original_start_time": "00:00:00.000",
                          "original_end_time": "00:00:05.311",
                          "full_texts": "中国劳动力至少会出现1亿到3亿人失业。未来的出路必将是自由职业和回农村。",
                          "adjusted_texts": "未来，我国或将面临上亿规模的失业挑战。返乡创业与成为自由职业者，是两大破局之路。"
                        }
                      ],
                      "deleted_scenes": [
                        {
                          "scene_id": "场景3",
                          "scene_desc": "描述被删除场景的内容。",
                          "reasoning": "清晰地解释删除这个场景的原因（例如：包含个人信息、内容冗余、节奏拖沓等）。",
                          "original_start_time": "00:00:10.621",
                          "original_end_time": "00:00:15.646"
                        }
                      ]
                    }
                    ```
                    场景信息如下:
        """
        prompt = f"{prompt}\n{scene_info_dict}"
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)
        # 增加原始的时间段到result
        final_cut_sequence = result.get('final_cut_sequence', [])
        deleted_scenes = result.get('deleted_scenes', [])
        for scene in final_cut_sequence:
            scene_id = scene.get('scene_id')
            # if scene_id:
            #     # 在场景信息中添加原始时间段
            #     time_list = scene_info_dict.get(scene_id, [])
            #     scene['original_start_time'] = time_list[0]
            #     scene['original_end_time'] = time_list[1]
        for scene in deleted_scenes:
            scene_id = scene.get('scene_id')
            # if scene_id:
            #     # 在删除的场景信息中添加原始时间段
            #     time_list = scene_info_dict.get(scene_id, [])
            #     scene['original_start_time'] = time_list[0]
            #     scene['original_end_time'] = time_list[1]
        return result
    except Exception as e:
        traceback.print_exc()
        return None


def gen_cut_suggestion_with_speech(video_path, owner_speech_list):
    """
    生成剪辑的建议，交换场景顺序或者删除场景。
    """
    try:
        scene_info_dict = find_and_split_scenes(video_path)
        if not scene_info_dict:
            print("未能成功获取视频场景信息。", scene_info_dict)
            return
        prompt = """# 角色
                    你是一位拥有十年以上经验的**资深视频剪辑总监**和**首席社交媒体内容策略师**。你不仅精通抖音、Bilibili、YouTube Shorts的算法和用户心理，更重要的是，你是一位**务实的创作者**，深刻理解每一次剪辑都意味着时间成本。你的决策冷静、精准，始终追求“投入产出比”最高的神级剪辑。

                    # 核心原则
                    在开始任何分析之前，请将以下原则作为你思考的基石：

                    1.  **叙事逻辑优先 (Narrative First)**：视频的流畅性和逻辑连贯性是基础。任何调整都不能破坏故事的内在逻辑或观众的理解流畅度。
                    2.  **保留是默认选项 (Keep is the Default)**：尊重原始素材的创作意图。不要为了调整而调整。如果一个场景没有严重问题，就应该保留。
                    3.  **高门槛调整原则 (High Bar for Changes)**：
                        *   **删除 (Delete)**：必须有充分理由，例如内容完全冗余、质量严重低下或明显偏离主题。
                        *   **重排 (Reorder)**：这是最高成本的操作，必须慎之又慎。**只有当重排能带来压倒性的优势时（例如，创造出无法替代的“黄金三秒”钩子，或解决了致命的叙事缺陷），才予以考虑。** 你的理由必须极具说服力。
                    4.  **效果是唯一标准 (Impact is Everything)**：所有决策的唯一目标是让最终成片在**观众吸引力、叙事流畅性、信息价值和传播潜力**上获得**显著提升**。微小的、可有可无的优化不是你的追求。
                    5.  如果该场景中有不相关的内容（如广告 推广 甚至是关于我的声明或者介绍）不管是不是我推荐的，那么这个场景都应该被删除
                    6. 而且要尽量不要产生太多的剪切点，因为会导致难度大大提示

                    # 任务指令
                    1.  **整体理解与诊断 (Holistic Understanding & Diagnosis)**：
                        *   首先，快速看完所有场景描述，总结出视频的**核心价值主张**（即“观众为什么要看这个视频？”）。
                        *   识别出整个视频中最具潜力的**“黄金时刻”或“高光片段”**。这是你后续决策的关键锚点。

                    2.  **逐一评估 (Scene-by-Scene Evaluation)**：
                        *   结合你对整体的理解，独立评估每个原始场景。评估维度包括：
                            *   **信息密度**：是否传递了关键信息？
                            *   **视觉冲击力**：画面是否吸引人？
                            *   **情绪价值**：能否引发观众的情绪（好奇、共鸣、兴奋、爽感等）？
                            *   **叙事功能**：在故事中扮演什么角色（开端、发展、高潮、结尾、铺垫、转折）？
                            *   **冗余性**：是否拖沓、重复或可被更好的场景替代？

                    3.  **制定剪辑策略 (Formulate the Editing Strategy)**：
                        *   严格遵循上述**【核心原则】**，结合你的评估，构建最终剪辑方案。
                        *   对于每一个决策（保留、重排、删除），在`reasoning`中清晰阐述你的思考过程，特别是要体现你的**审慎和对效果的追求**。例如，解释为什么保留是当前最佳选择，或者阐述一个重排建议为何能带来“压倒性优势”。

                    4.  **生成最终方案 (Generate Final Plan)**：
                        *   将你的决策结果以纯JSON格式输出。

                    # 输出要求
                    *   **严格的JSON格式**：你的输出必须是**一个完整且格式正确的JSON对象**，不能包含任何JSON格式之外的标记、注释、代码块标识（如 ```json ... ```）或任何解释性文本。
                    *   **内容结构**：JSON对象必须包含以下三个顶级键：`overall_strategy`, `final_cut_sequence`, `deleted_scenes`。

                    ---
                    ### **JSON输出格式定义与示例**

                    ```json
                    {
                      "overall_strategy": "（这里是你基于【核心原则】和【整体诊断】得出的顶层策略。例如：原始顺序的叙事逻辑清晰，核心价值突出，仅需删除一个冗余场景来加快节奏，无需进行高成本的重排。）",
                      "final_cut_sequence": [
                        {
                          "scene_id": "场景1",
                          "scene_desc": "（场景的简短描述）",
                          "reasoning": "（你的决策理由。例如：作为视频的自然开端，有效建立情境，逻辑清晰，是最佳的起始点，无需调整。）"
                        },
                        {
                          "scene_id": "场景3",
                          "scene_desc": "（场景的简短描述）",
                          "reasoning": "（例如：这是视频的‘高光时刻’，情绪价值最高，紧随场景1能快速抓住用户，保留其在故事发展中的位置可确保叙事连贯性。）"
                        }
                      ],
                      "deleted_scenes": [
                        {
                          "scene_id": "场景2",
                          "scene_desc": "（场景的简短描述）",
                          "reasoning": "（你的决策理由。例如：此场景与场景3内容高度重叠，且信息密度较低，属于明显冗余。删除后能让叙事流直接从情境建立进入高光时刻，节奏更紧凑。）"
                        }
                      ]
                    }
                    ```

                    **原始场景分割信息如下**:

        """
        prompt = f"{prompt}\n{scene_info_dict}"
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)
        # 增加原始的时间段到result
        final_cut_sequence = result.get('final_cut_sequence', [])
        deleted_scenes = result.get('deleted_scenes', [])
        for scene in final_cut_sequence:
            scene_id = scene.get('scene_id')
            if scene_id:
                # 在场景信息中添加原始时间段
                time_list = scene_info_dict.get(scene_id, [])
                scene['original_start_time'] = time_list[0]
                scene['original_end_time'] = time_list[1]
        for scene in deleted_scenes:
            scene_id = scene.get('scene_id')
            if scene_id:
                # 在删除的场景信息中添加原始时间段
                time_list = scene_info_dict.get(scene_id, [])
                scene['original_start_time'] = time_list[0]
                scene['original_end_time'] = time_list[1]
        return result
    except Exception as e:
        traceback.print_exc()
        return None

def auto_cut(video_path, all_info, output_path):
    """
    尝试根据场景建议对视频进行剪辑。
    如果剪辑过程失败，则将原始视频文件复制到输出路径。

    参数:
    video_path (str): 原始视频的文件路径。
    all_info (dict): 包含视频信息的字典，可能会有 'cut_suggestion_info'。
    output_path (str): 处理后视频的输出路径。

    返回:
    dict: 返回剪辑建议信息；如果不存在则返回空字典。
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(video_path):
        print(f"ERROR: 视频文件未找到: {video_path}")
        return {}

    cut_suggestion_info = all_info.get('cut_suggestion_info')

    # 2. 如果没有现成的剪辑建议，则生成新的
    if not cut_suggestion_info:
        print("INFO: 未找到剪辑建议，正在生成新的建议...")
        try:
            # 假设 gen_cut_suggestion 是一个可能耗时或失败的函数
            cut_suggestion_info = gen_cut_suggestion(video_path)
            all_info['cut_suggestion_info'] = cut_suggestion_info
            print(f"INFO: 成功生成剪辑建议: {cut_suggestion_info}")
        except Exception as e:
            print(f"ERROR: 生成剪辑建议时发生错误: {e}")
            cut_suggestion_info = {} # 确保 cut_suggestion_info 依然是一个字典

    # 3. 检查剪辑建议是否有效，无效则直接复制文件
    if not cut_suggestion_info or not cut_suggestion_info.get('final_cut_sequence'):
        print("WARNING: 没有可用的剪辑序列，将直接复制原始视频。")
        try:
            shutil.copy(video_path, output_path)
            print(f"INFO: 已将原始视频成功复制到: {output_path}")
        except Exception as e:
            print(f"ERROR: 复制文件时发生错误: {e}")
        return cut_suggestion_info

    # 4. 核心处理逻辑：尝试剪辑视频
    try:
        final_cut_sequence = cut_suggestion_info['final_cut_sequence']
        print(f"INFO: 检测到 {len(final_cut_sequence)} 个剪辑片段，准备进行处理。")

        merged_list = merge_time_segments(final_cut_sequence)

        print("INFO: 开始使用 FFmpeg 对视频进行重新剪辑...")
        re_edit_video_ffmpeg(video_path, merged_list, output_path=output_path)
        print(f"INFO: 视频已成功剪辑并保存到: {output_path}")

    except Exception as e:
        # 5. 失败回退：如果 try 块中任何地方出错，执行这里的复制操作
        print(f"ERROR: 视频剪辑处理失败: {e}")
        print("INFO: 执行回退操作：复制原始视频文件。")
        try:
            shutil.copy(video_path, output_path)
            print(f"INFO: 已将原始视频成功复制到: {output_path}")
        except Exception as copy_e:
            print(f"ERROR: 复制原始视频文件时也发生错误: {copy_e}")

    return cut_suggestion_info

def fix_speech_list_by_scene(video_path, owner_speech_list):
    scene_info_dict = find_and_split_scenes(video_path)
    if not scene_info_dict:
        print("未能成功获取视频场景信息。", scene_info_dict)
        return scene_info_dict, owner_speech_list
    new_scenes, adjusted_texts = map_and_adjust_scenes(scene_info_dict, owner_speech_list)
    return new_scenes, adjusted_texts


def auto_cut_with_speech(video_path, all_info, output_path, owner_speech_list):
    """
    尝试根据场景建议对视频进行剪辑。
    如果剪辑过程失败，则将原始视频文件复制到输出路径。

    参数:
    video_path (str): 原始视频的文件路径。
    all_info (dict): 包含视频信息的字典，可能会有 'cut_suggestion_info'。
    output_path (str): 处理后视频的输出路径。

    返回:
    dict: 返回剪辑建议信息；如果不存在则返回空字典。
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(video_path):
        print(f"ERROR: 视频文件未找到: {video_path}")
        return {}
    new_scenes, adjusted_texts = fix_speech_list_by_scene(video_path, owner_speech_list)
    pure_scenes = {key: {"time_range": value.get("time_range"), "full_texts": value.get("full_text")} for key, value in
     new_scenes.items()}

    all_info['new_scenes'] = new_scenes
    all_info['adjusted_texts'] = adjusted_texts

    if not new_scenes:
        # 没有场景就直接复制文件
        shutil.copy(video_path, output_path)
        return {}
    cut_suggestion_info = gen_cut_suggestion_with_scene(video_path, pure_scenes)
    all_info['cut_suggestion_info'] = cut_suggestion_info

    # 2. 如果没有现成的剪辑建议，则生成新的
    if not cut_suggestion_info:
        print("INFO: 未找到剪辑建议，正在生成新的建议...")
        try:
            # 假设 gen_cut_suggestion 是一个可能耗时或失败的函数
            cut_suggestion_info = gen_cut_suggestion(video_path)
            all_info['cut_suggestion_info'] = cut_suggestion_info
            print(f"INFO: 成功生成剪辑建议: {cut_suggestion_info}")
        except Exception as e:
            print(f"ERROR: 生成剪辑建议时发生错误: {e}")
            cut_suggestion_info = {} # 确保 cut_suggestion_info 依然是一个字典

    # 3. 检查剪辑建议是否有效，无效则直接复制文件
    if not cut_suggestion_info or not cut_suggestion_info.get('final_cut_sequence'):
        print("WARNING: 没有可用的剪辑序列，将直接复制原始视频。")
        try:
            shutil.copy(video_path, output_path)
            print(f"INFO: 已将原始视频成功复制到: {output_path}")
        except Exception as e:
            print(f"ERROR: 复制文件时发生错误: {e}")
        return cut_suggestion_info

    # 4. 核心处理逻辑：尝试剪辑视频
    try:
        final_cut_sequence = cut_suggestion_info['final_cut_sequence']
        print(f"INFO: 检测到 {len(final_cut_sequence)} 个剪辑片段，准备进行处理。")

        merged_list = merge_time_segments(final_cut_sequence)

        print("INFO: 开始使用 FFmpeg 对视频进行重新剪辑...")
        re_edit_video_ffmpeg(video_path, merged_list, output_path=output_path)
        print(f"INFO: 视频已成功剪辑并保存到: {output_path}")

    except Exception as e:
        # 5. 失败回退：如果 try 块中任何地方出错，执行这里的复制操作
        print(f"ERROR: 视频剪辑处理失败: {e}")
        print("INFO: 执行回退操作：复制原始视频文件。")
        try:
            shutil.copy(video_path, output_path)
            print(f"INFO: 已将原始视频成功复制到: {output_path}")
        except Exception as copy_e:
            print(f"ERROR: 复制原始视频文件时也发生错误: {copy_e}")

    return cut_suggestion_info

def add_origin_audio(video_path, owner_speech_with_audio_list, voice_output_dir, video_duration):
    """
    补充原来的声音，因为有些时候视频中引用了其他人的声音，现在需要保留下来
    """
    base_name = os.path.basename(video_path)
    new_owner_speech_with_audio_list = fill_time_gaps(owner_speech_with_audio_list, video_duration)
    if len(new_owner_speech_with_audio_list) > len(owner_speech_with_audio_list):
        origin_audio_path = base_name.replace('.mp4', '_origin_audio.wav')
        # 说明新增了片段，需要进行处理
        extract_audio_from_video(video_path, f'{voice_output_dir}/{origin_audio_path}')
        print(f"已提取原始音频到: {voice_output_dir}/{origin_audio_path}")
        separate_with_cli(f'{voice_output_dir}/{origin_audio_path}', output_dir=voice_output_dir, two_stems=True)
        vocals_path = find_file_by_name(voice_output_dir, 'vocals.wav')
        if not vocals_path:
            vocals_path = origin_audio_path
            print("未找到分离的原始音频，使用原始音频作为补充。")

        for speech in new_owner_speech_with_audio_list:
            text = speech['text']
            if "[无声]" == text:
                speech_id = speech['id']
                audio_path = f'{voice_output_dir}/{speech_id}_origin.wav'
                startTime = speech['startTime']
                endTime = speech['endTime']
                start_time_s = time_to_ms(startTime) / 1000
                end_time_s = time_to_ms(endTime) / 1000
                cut_audio_segment(vocals_path, start_time_s, end_time_s, audio_path)
                if os.path.exists(audio_path):
                    speech['outputPath'] = audio_path
                print(f"已将无声片段 {speech_id} 的音频补充为原始音频: {audio_path} {startTime} {endTime}")


    return new_owner_speech_with_audio_list

def remake_video_op(
        video_path: str,
        output_root='./',
        bgm_library_path: str = 'bgm_audio',
        force_regenerate: bool = False
) -> str | None:
    """
    重制视频的健壮版本。

    Args:
        video_path (str): 原始视频的绝对或相对路径.
        output_root (str): 所有输出文件（包括中间文件和最终结果）存放的根目录.
        bgm_library_path (str): BGM 音频库的路径.
        force_regenerate (bool): 如果为 True, 将忽略缓存, 重新生成所有数据.

    Returns:
        str | None: 成功则返回最终视频的路径, 失败则返回 None.
    """
    # --- 1. 初始化和路径设置 (核心改进) ---
    try:
        # 验证输入路径
        if not os.path.isfile(video_path):
            print(f"输入视频文件不存在: {video_path}")
            return None

        # 创建独立的输出目录，避免污染根目录
        base_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(base_name)[0]
        # 所有输出都将在这个目录里
        processing_dir = os.path.join(output_root, f"{video_name_without_ext}_remake_files")
        os.makedirs(processing_dir, exist_ok=True)
        print(f"所有文件将输出到: {processing_dir}")

        # 使用字典统一管理所有路径，清晰且不易出错
        paths = {
            'original': video_path,
            'info_json': os.path.join(processing_dir, 'all_info.json'),
            'covered': os.path.join(processing_dir, f"{video_name_without_ext}_covered.mp4"),
            'with_subtitles': os.path.join(processing_dir, f"{video_name_without_ext}_with_subtitles.mp4"),
            'redubbed': os.path.join(processing_dir, f"{video_name_without_ext}_redub.mp4"),
            'auto_cut': os.path.join(processing_dir, f"{video_name_without_ext}_auto_cut.mp4"),
            'final_video': os.path.join(processing_dir, f"{video_name_without_ext}_final.mp4"),
        }

        # --- 2. 加载或创建核心信息文件 ---
        if os.path.exists(paths['info_json']) and not force_regenerate:
            print(f"从 {paths['info_json']} 加载缓存信息...")
            all_info = read_json(paths['info_json'])
        else:
            all_info = {}

    except Exception as e:
        print(f"初始化失败: {e}", exc_info=True)
        return None
    owner_speech_list = all_info.get('owner_speech_list', [])
    # --- 3. 核心处理流程 (每一步都有错误处理和数据校验) ---
    try:
        video_duration = get_video_duration_seconds(paths['original'])  # 检查视频是否有效
        # 步骤 3.1: 获取主人公语音片段
        if 'owner_speech_list' not in all_info or force_regenerate:
            print("缓存未命中或强制刷新，正在提取主人公语音...")
            owner_speech_list = get_owner_speech_pure(paths['original'])
            if not owner_speech_list:
                raise ValueError("提取主人公语音失败或返回格式不正确")
            all_info['owner_speech_list'] = owner_speech_list
            save_json(paths['info_json'], all_info)
        if not owner_speech_list:
            print('无主人说话可直接返回原视频')
            return paths['original']

        auto_cut_with_speech(paths['original'], all_info, paths['auto_cut'], owner_speech_list)


        suggestion_speech = all_info['suggestion_speech']
        owner_speech_list = suggestion_speech.get('transcription', [])
        # 安全地获取嵌套数据
        recommendations = suggestion_speech.get('recommendations', {})
        voice_name = recommendations.get('voice', {}).get('voice_name')
        bgm_id = recommendations.get('bgm', {}).get('id')
        if not owner_speech_list:
            print('无主人说话可直接返回原视频')
            return paths['original']
        if not all([voice_name, bgm_id]):
            raise ValueError("从 suggestion_speech 中获取 'transcription', 'voice_name' 或 'bgm_id' 失败")

        # 构建BGM的绝对路径，更可靠
        bgm_filename = bgm_id.replace(".mp4", ".wav")
        bgm_path = os.path.join(bgm_library_path, bgm_filename)
        if not os.path.isfile(bgm_path):
            raise FileNotFoundError(f"BGM文件不存在: {bgm_path}")
        print(f"使用 BGM: {bgm_path}")
        merged_timerange_list = merge_time_intervals(owner_speech_list)  # 确保时间片段合并
        # 步骤 3.2: 获取字幕框
        if 'final_subtitle_box' not in all_info or force_regenerate:
            print("正在计算字幕框...")
            final_box = find_overall_subtitle_box_target_number(paths['original'], merged_timerange_list=merged_timerange_list)
            if not final_box:
                raise ValueError("寻找字幕框失败")
            all_info['final_subtitle_box'] = final_box
            save_json(paths['info_json'], all_info)


        top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(paths['original'], all_info['final_subtitle_box'])

        # 步骤 3.3: 覆盖字幕区域
        print(f"正在覆盖字幕区域...")
        cover_subtitle(paths['original'], paths['covered'], top_left, bottom_right)
        if not os.path.exists(paths['covered']):
            raise RuntimeError("覆盖字幕后，输出文件未生成")

        # 步骤 3.4: 增加新的文案和字幕
        font_size = int((bottom_right[1] - top_left[1]) * 0.8)
        bottom_margin = vid_h - bottom_right[1] + int((bottom_right[1] - top_left[1]) * 0.1)
        print("正在添加新字幕...")
        add_subtitle(
            paths['covered'],
            owner_speech_list,
            paths['with_subtitles'],
            bottom_margin=bottom_margin,
            font_size=font_size,
            fixed_rect=[top_left, bottom_right]
        )
        if not os.path.exists(paths['with_subtitles']):
            raise RuntimeError("添加新字幕后，输出文件未生成")

        # 步骤 3.5: 生成新音频并配音
        if 'new_owner_speech_with_audio_list' not in all_info or force_regenerate:
            print("正在生成新音频...")
            voice_output_dir = f'{processing_dir}/{voice_name}'

            owner_speech_with_audio_list = gen_new_audio(owner_speech_list, voice_name, voice_output_dir)
            new_owner_speech_with_audio_list = add_origin_audio(paths['original'], owner_speech_with_audio_list, voice_output_dir, video_duration)
            if not new_owner_speech_with_audio_list:
                raise ValueError("生成新音频或混合原音频失败")
            all_info['new_owner_speech_with_audio_list'] = new_owner_speech_with_audio_list
            save_json(paths['info_json'], all_info)

        new_owner_speech_with_audio_list = all_info['new_owner_speech_with_audio_list']
        print("正在为视频重配音...")
        redub_video_with_ffmpeg(paths['with_subtitles'], new_owner_speech_with_audio_list,
                                output_path=paths['redubbed'])
        if not os.path.exists(paths['redubbed']):
            raise RuntimeError("重配音后，输出文件未生成")

        # 步骤 3.6: 自动剪辑
        print("正在自动剪辑...")
        auto_cut(paths['redubbed'], all_info, paths['auto_cut'])
        save_json(paths['info_json'], all_info)  # 保存可能在auto_cut中更新的信息
        if not os.path.exists(paths['auto_cut']):
            raise RuntimeError("自动剪辑后，输出文件未生成")

        # 步骤 3.7: 添加背景音乐
        print("正在添加背景音乐...")
        add_bgm_to_video(paths['auto_cut'], bgm_path, paths['final_video'])
        if not os.path.exists(paths['final_video']):
            raise RuntimeError("添加BGM后，最终文件未生成")

        print(f"视频重制成功！最终文件位于: {paths['final_video']}")

        # final_name = os.path.basename(paths['final_video'])
        # for entry in os.listdir(processing_dir):
        #     full_path = os.path.join(processing_dir, entry)
        #     if entry == final_name or entry == 'all_info.json':
        #         continue
        #     try:
        #         if os.path.isdir(full_path):
        #             shutil.rmtree(full_path)
        #         else:
        #             os.remove(full_path)
        #     except Exception as cleanup_err:
        #         print(f"清理时出错，无法删除 {full_path}: {cleanup_err}")

        return paths['final_video']

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        traceback.print_exc()
        print(f"处理流程中断: {e}")
        return None
    except Exception as e:
        traceback.print_exc()
        return None


def remake_video_robust(
        video_path: str,
        output_root='./',
        bgm_library_path: str = 'bgm_audio',
        force_regenerate: bool = False
) -> str | None:
    """
    重制视频的健壮版本。

    Args:
        video_path (str): 原始视频的绝对或相对路径.
        output_root (str): 所有输出文件（包括中间文件和最终结果）存放的根目录.
        bgm_library_path (str): BGM 音频库的路径.
        force_regenerate (bool): 如果为 True, 将忽略缓存, 重新生成所有数据.

    Returns:
        str | None: 成功则返回最终视频的路径, 失败则返回 None.
    """
    # --- 1. 初始化和路径设置 (核心改进) ---
    try:
        # 验证输入路径
        if not os.path.isfile(video_path):
            print(f"输入视频文件不存在: {video_path}")
            return None

        # 创建独立的输出目录，避免污染根目录
        base_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(base_name)[0]
        # 所有输出都将在这个目录里
        processing_dir = os.path.join(output_root, f"{video_name_without_ext}_remake_files")
        os.makedirs(processing_dir, exist_ok=True)
        print(f"所有文件将输出到: {processing_dir}")

        # 使用字典统一管理所有路径，清晰且不易出错
        paths = {
            'original': video_path,
            'info_json': os.path.join(processing_dir, 'all_info.json'),
            'covered': os.path.join(processing_dir, f"{video_name_without_ext}_covered.mp4"),
            'with_subtitles': os.path.join(processing_dir, f"{video_name_without_ext}_with_subtitles.mp4"),
            'redubbed': os.path.join(processing_dir, f"{video_name_without_ext}_redub.mp4"),
            'auto_cut': os.path.join(processing_dir, f"{video_name_without_ext}_auto_cut.mp4"),
            'final_video': os.path.join(processing_dir, f"{video_name_without_ext}_final.mp4"),
        }

        # --- 2. 加载或创建核心信息文件 ---
        if os.path.exists(paths['info_json']) and not force_regenerate:
            print(f"从 {paths['info_json']} 加载缓存信息...")
            all_info = read_json(paths['info_json'])
        else:
            all_info = {}

    except Exception as e:
        print(f"初始化失败: {e}", exc_info=True)
        return None

    # --- 3. 核心处理流程 (每一步都有错误处理和数据校验) ---
    try:
        video_duration = get_video_duration_seconds(paths['original'])  # 检查视频是否有效
        # 步骤 3.1: 获取主人公语音片段
        if 'suggestion_speech' not in all_info or force_regenerate:
            print("缓存未命中或强制刷新，正在提取主人公语音...")
            suggestion_speech = get_owner_speech(paths['original'])
            if not suggestion_speech or 'transcription' not in suggestion_speech:
                raise ValueError("提取主人公语音失败或返回格式不正确")
            all_info['suggestion_speech'] = suggestion_speech
            save_json(paths['info_json'], all_info)

        suggestion_speech = all_info['suggestion_speech']
        owner_speech_list = suggestion_speech.get('transcription', [])
        # 安全地获取嵌套数据
        recommendations = suggestion_speech.get('recommendations', {})
        voice_name = recommendations.get('voice', {}).get('voice_name')
        bgm_id = recommendations.get('bgm', {}).get('id')
        if not owner_speech_list:
            print('无主人说话可直接返回原视频')
            return paths['original']
        if not all([voice_name, bgm_id]):
            raise ValueError("从 suggestion_speech 中获取 'transcription', 'voice_name' 或 'bgm_id' 失败")

        # 构建BGM的绝对路径，更可靠
        bgm_filename = bgm_id.replace(".mp4", ".wav")
        bgm_path = os.path.join(bgm_library_path, bgm_filename)
        if not os.path.isfile(bgm_path):
            raise FileNotFoundError(f"BGM文件不存在: {bgm_path}")
        print(f"使用 BGM: {bgm_path}")
        merged_timerange_list = merge_time_intervals(owner_speech_list)  # 确保时间片段合并
        # 步骤 3.2: 获取字幕框
        if 'final_subtitle_box' not in all_info or force_regenerate:
            print("正在计算字幕框...")
            final_box = find_overall_subtitle_box_target_number(paths['original'], merged_timerange_list=merged_timerange_list)
            if not final_box:
                raise ValueError("寻找字幕框失败")
            all_info['final_subtitle_box'] = final_box
            save_json(paths['info_json'], all_info)


        top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(paths['original'], all_info['final_subtitle_box'])

        # 步骤 3.3: 覆盖字幕区域
        print(f"正在覆盖字幕区域...")
        cover_subtitle(paths['original'], paths['covered'], top_left, bottom_right)
        if not os.path.exists(paths['covered']):
            raise RuntimeError("覆盖字幕后，输出文件未生成")

        # 步骤 3.4: 增加新的文案和字幕
        font_size = int((bottom_right[1] - top_left[1]) * 0.8)
        bottom_margin = vid_h - bottom_right[1] + int((bottom_right[1] - top_left[1]) * 0.1)
        print("正在添加新字幕...")
        add_subtitle(
            paths['covered'],
            owner_speech_list,
            paths['with_subtitles'],
            bottom_margin=bottom_margin,
            font_size=font_size,
            fixed_rect=[top_left, bottom_right]
        )
        if not os.path.exists(paths['with_subtitles']):
            raise RuntimeError("添加新字幕后，输出文件未生成")

        # 步骤 3.5: 生成新音频并配音
        if 'new_owner_speech_with_audio_list' not in all_info or force_regenerate:
            print("正在生成新音频...")
            voice_output_dir = f'{processing_dir}/{voice_name}'

            owner_speech_with_audio_list = gen_new_audio(owner_speech_list, voice_name, voice_output_dir)
            new_owner_speech_with_audio_list = add_origin_audio(paths['original'], owner_speech_with_audio_list, voice_output_dir, video_duration)
            if not new_owner_speech_with_audio_list:
                raise ValueError("生成新音频或混合原音频失败")
            all_info['new_owner_speech_with_audio_list'] = new_owner_speech_with_audio_list
            save_json(paths['info_json'], all_info)

        new_owner_speech_with_audio_list = all_info['new_owner_speech_with_audio_list']
        print("正在为视频重配音...")
        redub_video_with_ffmpeg(paths['with_subtitles'], new_owner_speech_with_audio_list,
                                output_path=paths['redubbed'])
        if not os.path.exists(paths['redubbed']):
            raise RuntimeError("重配音后，输出文件未生成")

        # 步骤 3.6: 自动剪辑
        print("正在自动剪辑...")
        auto_cut(paths['redubbed'], all_info, paths['auto_cut'])
        save_json(paths['info_json'], all_info)  # 保存可能在auto_cut中更新的信息
        if not os.path.exists(paths['auto_cut']):
            raise RuntimeError("自动剪辑后，输出文件未生成")

        # 步骤 3.7: 添加背景音乐
        print("正在添加背景音乐...")
        add_bgm_to_video(paths['auto_cut'], bgm_path, paths['final_video'])
        if not os.path.exists(paths['final_video']):
            raise RuntimeError("添加BGM后，最终文件未生成")

        print(f"视频重制成功！最终文件位于: {paths['final_video']}")

        # final_name = os.path.basename(paths['final_video'])
        # for entry in os.listdir(processing_dir):
        #     full_path = os.path.join(processing_dir, entry)
        #     if entry == final_name or entry == 'all_info.json':
        #         continue
        #     try:
        #         if os.path.isdir(full_path):
        #             shutil.rmtree(full_path)
        #         else:
        #             os.remove(full_path)
        #     except Exception as cleanup_err:
        #         print(f"清理时出错，无法删除 {full_path}: {cleanup_err}")

        return paths['final_video']

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        traceback.print_exc()
        print(f"处理流程中断: {e}")
        return None
    except Exception as e:
        traceback.print_exc()
        return None




if __name__ == '__main__':
    video_path = r"W:\project\python_project\watermark_remove\LLM\TikTokDownloader\downloads\2025-07-14 11.36.47-视频-餐饮小诸葛-外卖乱战，谁是受害者？ #餐饮 #外卖战 #餐饮外卖 #餐饮人 @抖音小助手.mp4"
    # remake_video_robust(video_path)
    remake_video_op(video_path)



