from flask import Flask, request, render_template, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import demo  # 引入您的 demo.py 文件

app = Flask(__name__)

# 配置上传文件夹和允许的文件类型
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# 确保上传和处理文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'video' not in request.files or 'face' not in request.files or 'mosaic' not in request.files:
        return "Missing files. Please upload a video, target face image, and mosaic image.", 400

    # 获取文件
    video = request.files['video']
    face = request.files['face']
    mosaic = request.files['mosaic']

    if video and allowed_file(video.filename) and face and allowed_file(face.filename) and mosaic and allowed_file(mosaic.filename):
        # 保存文件
        video_filename = secure_filename(video.filename)
        face_filename = secure_filename(face.filename)
        mosaic_filename = secure_filename(mosaic.filename)

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        mosaic_path = os.path.join(app.config['UPLOAD_FOLDER'], mosaic_filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{video_filename}")
        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'short.mp3')  # 提取出的音频，处理后删掉
        silent_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_short.mp4')  # 打码后的无声视频，最后删掉

        video.save(video_path)
        face.save(face_path)
        mosaic.save(mosaic_path)

        try:
            # 调用 demo.py 中的函数进行视频处理
            demo.process_video(video_path, face_path, mosaic_path, output_path)

            # 删除临时文件 short.mp3
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                # 删除无声视频 processed_short.mp4
                if os.path.exists(silent_video_path):
                    os.remove(silent_video_path)

        except Exception as e:
            return f"Error processing video: {str(e)}", 500

        # 返回 JSON 格式响应，包含处理后视频的文件名
        return jsonify({"filename": f"processed_{video_filename.split('.')[0]}-final.mp4"})

    return "Invalid file format. Please upload valid files.", 400

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found.", 404

if __name__ == '__main__':
    app.run(debug=True)

