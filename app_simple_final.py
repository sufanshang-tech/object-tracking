import streamlit as st
import numpy as np
from collections import deque
import tempfile
import os
from PIL import Image, ImageDraw
import pandas as pd

# 頁面配置
st.set_page_config(
    page_title="物件追蹤系統",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 物件追蹤系統")
st.markdown("簡單的物件追蹤應用程式")

# 側邊欄
with st.sidebar:
    st.header("⚙️ 設定")
    uploaded_file = st.file_uploader("上傳影片", type=['mp4', 'avi', 'mov', 'mkv'])
    
    st.subheader("偵測參數")
    brightness_threshold = st.slider("亮度閾值", 0, 255, 150)
    min_area = st.slider("最小物體面積（像素）", 100, 10000, 500)

def detect_bright_objects(frame_pil, threshold, min_area):
    """偵測影像中的亮色物體"""
    frame_np = np.array(frame_pil)
    
    # 轉換為灰度
    if len(frame_np.shape) == 3:
        gray = np.mean(frame_np, axis=2)
    else:
        gray = frame_np
    
    # 二值化：找出亮度高於閾值的像素
    binary = gray > threshold
    
    # 找出連通區域（簡單的物體偵測）
    objects = []
    visited = np.zeros_like(binary, dtype=bool)
    
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] and not visited[i, j]:
                # 使用簡單的泛洪填充找出物體
                object_pixels = []
                stack = [(i, j)]
                
                while stack and len(object_pixels) < min_area * 2:
                    y, x = stack.pop()
                    
                    if y < 0 or y >= binary.shape[0] or x < 0 or x >= binary.shape[1]:
                        continue
                    
                    if visited[y, x] or not binary[y, x]:
                        continue
                    
                    visited[y, x] = True
                    object_pixels.append((x, y))
                    
                    # 檢查相鄰像素
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((y + dy, x + dx))
                
                # 如果物體足夠大，記錄它
                if len(object_pixels) >= min_area:
                    xs = [p[0] for p in object_pixels]
                    ys = [p[1] for p in object_pixels]
                    
                    center_x = np.mean(xs)
                    center_y = np.mean(ys)
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    objects.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'area': len(object_pixels)
                    })
    
    return objects

def match_objects(current_objects, track_history, max_distance=50):
    """將當前幀的物體與追蹤歷史匹配"""
    if not track_history:
        # 第一幀，為每個物體分配新的 ID
        for i, obj in enumerate(current_objects):
            track_history[i] = deque(maxlen=30)
            track_history[i].append((obj['center_x'], obj['center_y']))
        return track_history
    
    # 簡單的匹配策略：找最近的追蹤
    used_objects = set()
    
    for track_id, history in track_history.items():
        if history:
            last_x, last_y = history[-1]
            
            # 找最近的物體
            best_obj_idx = -1
            best_distance = max_distance
            
            for obj_idx, obj in enumerate(current_objects):
                if obj_idx in used_objects:
                    continue
                
                dist = np.sqrt((obj['center_x'] - last_x)**2 + (obj['center_y'] - last_y)**2)
                
                if dist < best_distance:
                    best_distance = dist
                    best_obj_idx = obj_idx
            
            if best_obj_idx >= 0:
                used_objects.add(best_obj_idx)
                history.append((current_objects[best_obj_idx]['center_x'], 
                               current_objects[best_obj_idx]['center_y']))
    
    # 為未匹配的物體分配新 ID
    next_id = max(track_history.keys()) + 1 if track_history else 0
    for obj_idx, obj in enumerate(current_objects):
        if obj_idx not in used_objects:
            track_history[next_id] = deque(maxlen=30)
            track_history[next_id].append((obj['center_x'], obj['center_y']))
            next_id += 1
    
    return track_history

def draw_tracking_results(frame_pil, current_objects, track_history):
    """在影像上繪製追蹤結果"""
    draw = ImageDraw.Draw(frame_pil)
    
    for track_id, history in track_history.items():
        if not history:
            continue
        
        # 繪製軌跡
        points = list(history)
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                draw.line([pt1, pt2], fill='blue', width=2)
        
        # 繪製當前位置
        if points:
            last_x, last_y = points[-1]
            draw.ellipse([last_x - 5, last_y - 5, last_x + 5, last_y + 5], 
                        fill='red', outline='red')
            draw.text((last_x + 10, last_y), f"ID: {track_id}", fill='green')
    
    return frame_pil

if uploaded_file is not None:
    # 保存上傳的檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.success("✅ 檔案已上傳")
    
    try:
        # 使用 moviepy 處理影片
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(tmp_path)
        frame_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frames = []
        track_history = {}
        coordinates_data = []
        
        for frame_array in video.iter_frames():
            # 轉換為 PIL Image
            frame_pil = Image.fromarray(frame_array.astype('uint8'))
            
            # 調整大小
            frame_pil = frame_pil.resize((640, 480))
            
            # 偵測物體
            current_objects = detect_bright_objects(frame_pil, brightness_threshold, min_area)
            
            # 匹配追蹤
            track_history = match_objects(current_objects, track_history)
            
            # 繪製結果
            frame_with_tracking = draw_tracking_results(frame_pil.copy(), current_objects, track_history)
            
            # 記錄座標
            for track_id, history in track_history.items():
                if history:
                    last_x, last_y = history[-1]
                    coordinates_data.append({
                        'frame': frame_count,
                        'track_id': track_id,
                        'x': float(last_x),
                        'y': float(last_y),
                        'trajectory_length': len(history)
                    })
            
            frames.append(frame_with_tracking)
            frame_count += 1
            
            progress = min(frame_count / 150, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"已處理 {frame_count} 幀...")
            
            if frame_count >= 150:
                break
        
        video.close()
        
        # 顯示結果
        st.subheader("📹 追蹤結果")
        
        if frames:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(frames[0], use_column_width=True)
                st.caption("第一幀")
            
            with col2:
                st.image(frames[-1], use_column_width=True)
                st.caption("最後一幀")
            
            # 統計資訊
            st.subheader("📊 統計資訊")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("總幀數", frame_count)
            
            with col2:
                st.metric("追蹤物件數", len(track_history))
            
            with col3:
                if track_history:
                    avg_length = int(np.mean([len(h) for h in track_history.values()]))
                    st.metric("平均軌跡長度", avg_length)
            
            # 顯示座標數據
            st.subheader("📊 座標數據")
            if coordinates_data:
                df = pd.DataFrame(coordinates_data)
                st.dataframe(df, use_container_width=True)
                
                # 下載 CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下載座標數據 (CSV)",
                    data=csv,
                    file_name="tracking_coordinates.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ 處理影片時出錯: {str(e)}")
        st.info("💡 請確保上傳的是有效的影片檔案")
    
    finally:
        # 清理臨時檔案
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 請在左側上傳影片檔案開始追蹤")
