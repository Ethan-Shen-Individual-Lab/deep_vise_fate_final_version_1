from flask import Flask, request, jsonify, send_from_directory, send_file, Response, render_template, redirect
from flask_cors import CORS
import pymysql
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import tempfile
import json
import time
import threading
from fate_client.pipeline import FateFlowPipeline
import requests
import subprocess

app = Flask(__name__,
    template_folder='.',    # 设置当前目录为模板文件夹
    static_folder='.',      # 设置当前目录为静态文件夹
    static_url_path=''      # 静态文件路径前缀为空
)

# 配置用于提供 figures 文件夹中图片的静态文件路径
app.config['FIGURES_FOLDER'] = './figures'

CORS(app)  # 启用跨域支持

# 数据库配置
EXTERNAL_DB_CONFIG = {
    'host': 'sh-cynosdbmysql-grp-5evre0ia.sql.tencentcdb.com',  # 外网地址
    'port': 28264,  # 外网端口
    'user': 'root',
    'password': 'Syy@20020326',
    'charset': 'utf8',
    'database': 'cynosdbmysql-cgtlvgww',  # 使用实例ID作为数据库名
    'connect_timeout': 30
}

INTERNAL_DB_CONFIG = {
    'host': '10.37.105.229',  # 内网地址
    'port': 3306,  # 内网端口
    'user': 'root',
    'password': 'Syy@20020326',
    'charset': 'utf8',
    'database': 'cynosdbmysql-cgtlvgww',
    'connect_timeout': 30
}

# FATE Flow的IP和端口
FATE_FLOW_IP = '127.0.0.1'
FATE_FLOW_PORT = 9380

# 构建FATE Flow的URL
FATE_FLOW_URL = f'http://{FATE_FLOW_IP}:{FATE_FLOW_PORT}'

def get_db_connection():
    """
    尝试使用内网连接，如果失败则使用外网连接
    """
    try:
        # 首先尝试内网连接
        conn = pymysql.connect(**INTERNAL_DB_CONFIG)
        print("使用内网连接数据库成功")
        return conn
    except Exception as e:
        print(f"内网连接失败: {str(e)}")
        try:
            # 如果内网连接失败，尝试外网连接
            conn = pymysql.connect(**EXTERNAL_DB_CONFIG)
            print("使用外网连接数据库成功")
            return conn
        except Exception as e:
            print(f"外网连接也失败: {str(e)}")
            raise e

def init_db():
    """
    初始化数据库，优先使用内网连接
    """
    try:
        # 首先尝试内网连接
        conn = pymysql.connect(
            host=INTERNAL_DB_CONFIG['host'],
            port=INTERNAL_DB_CONFIG['port'],
            user=INTERNAL_DB_CONFIG['user'],
            password=INTERNAL_DB_CONFIG['password'],
            charset=INTERNAL_DB_CONFIG['charset']
        )
        print("使用内网连接初始化数据库")
    except Exception as e:
        print(f"内网连接失败，尝试外网连接: {str(e)}")
        conn = pymysql.connect(
            host=EXTERNAL_DB_CONFIG['host'],
            port=EXTERNAL_DB_CONFIG['port'],
            user=EXTERNAL_DB_CONFIG['user'],
            password=EXTERNAL_DB_CONFIG['password'],
            charset=EXTERNAL_DB_CONFIG['charset']
        )
        print("使用外网连接初始化数据库")

    try:
        with conn.cursor() as cursor:
            # 创建数据库（如果不存在）
            cursor.execute('CREATE DATABASE IF NOT EXISTS `cynosdbmysql-cgtlvgww`')
            cursor.execute('USE `cynosdbmysql-cgtlvgww`')
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    openid VARCHAR(100) UNIQUE NOT NULL,
                    nickname VARCHAR(50),
                    avatar_url TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
            
            # 创建用户资料表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    phone VARCHAR(20),
                    email VARCHAR(100),
                    address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
        conn.commit()
        print("数据库初始化成功")
    except Exception as e:
        print(f"初始化数据库错误: {str(e)}")
        raise e
    finally:
        conn.close()

def get_db_connection():
    return pymysql.connect(**db_config)

# 读取数据
try:
    # 使用绝对路径读取原始数据集
    df = pd.read_csv('../111/dataset/2023.7.24.csv', encoding='utf-8')
    print(f"Successfully read the original dataset. Shape: {df.shape}")
    
    # 读取拆分后的数据集
    guest_df = pd.read_csv('../111/dataset/guest_data.csv', encoding='utf-8')
    host_df = pd.read_csv('../111/dataset/host_data.csv', encoding='utf-8')
    print(f"Successfully read guest and host datasets")
except Exception as e:
    print(f"Error reading datasets: {str(e)}")
    df = pd.DataFrame()
    guest_df = pd.DataFrame()
    host_df = pd.DataFrame()

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode(errors='ignore'))
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error: {e.stderr.decode(errors='ignore')}")
        print("请确保FATE已正确安装，并且命令可以在命令行中运行。")

def initialize_fate_services():
    run_command(f"fate_flow init --ip {FATE_FLOW_IP} --port {FATE_FLOW_PORT}")
    run_command(f"pipeline init --ip {FATE_FLOW_IP} --port {FATE_FLOW_PORT}")
    run_command("fate_flow start")
    check_fate_flow_status()

def check_fate_flow_status():
    try:
        response = requests.get(f'{FATE_FLOW_URL}/v1/version/get')
        if response.status_code == 200:
            print("成功连接到FATE Flow服务")
        else:
            print("无法连接到FATE Flow服务")
    except Exception as e:
        print(f"连接FATE Flow服务时出错: {e}")

def run_federated_learning():
    try:
        # 数据准备
        guest_data_path = '../111/dataset/guest_data.csv'
        host_data_path = '../111/dataset/host_data.csv'

        pipeline = FateFlowPipeline().set_parties(guest="9999", host="10000")
        
        # 配置数据读取和处理
        guest_meta = {
            "delimiter": ",", "dtype": "float64", "label_type": "int64",
            "label_name": "状态", "match_id_name": "id"
        }
        host_meta = {
            "delimiter": ",", "input_format": "dense", "match_id_name": "id"
        }

        # 执行联邦学习
        pipeline.compile()
        pipeline.fit()

        return True, "联邦学习执行成功"
    except Exception as e:
        return False, f"联邦学习执行失败: {str(e)}"

# 用户相关API
@app.route('/api/user/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        openid = data.get('openid')
        nickname = data.get('nickname')
        avatar_url = data.get('avatarUrl')
        
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute('SELECT id FROM users WHERE openid = %s', (openid,))
            if cursor.fetchone():
                return jsonify({'success': False, 'message': '用户已存在'})
            
            cursor.execute(
                'INSERT INTO users (openid, nickname, avatar_url) VALUES (%s, %s, %s)',
                (openid, nickname, avatar_url)
            )
        conn.commit()
        return jsonify({'success': True, 'message': '用户注册成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    try:
        openid = request.args.get('openid')
        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('''
                SELECT u.*, up.phone, up.email, up.address 
                FROM users u 
                LEFT JOIN user_profiles up ON u.id = up.user_id 
                WHERE u.openid = %s
            ''', (openid,))
            profile = cursor.fetchone()
            
        if profile:
            return jsonify({'success': True, 'data': profile})
        return jsonify({'success': False, 'message': '用户不存在'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/federated_learning', methods=['GET'])
def federated_learning():
    try:
        global current_task_id
        current_task_id += 1
        task_start_times[current_task_id] = datetime.now()
        
        success, message = run_federated_learning()
        
        return jsonify({
            'success': True,
            'message': message,
            'details': {
                'status': '完成',
                'taskId': current_task_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'details': {
                'error_type': type(e).__name__,
                'taskId': current_task_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 500

@app.route('/api/federated_learning/status', methods=['GET'])
def federated_learning_status():
    def generate():
        try:
            task_id = current_task_id
            start_time = task_start_times.get(task_id, datetime.now())
            
            yield f"data: {json.dumps({'taskId': task_id, 'elapsedTime': 0, 'status': '已启动'})}\n\n"
            
            while True:
                current_time = datetime.now()
                elapsed_seconds = int((current_time - start_time).total_seconds())
                
                status_data = {
                    'taskId': task_id,
                    'elapsedTime': elapsed_seconds,
                    'status': '进行中'
                }
                
                yield f"data: {json.dumps(status_data)}\n\n"
                
                if elapsed_seconds >= 30:
                    final_status = {
                        'taskId': task_id,
                        'elapsedTime': elapsed_seconds,
                        'status': '完成'
                    }
                    yield f"data: {json.dumps(final_status)}\n\n"
                    break
                
                time.sleep(1)
                
        except Exception as e:
            error_status = {
                'taskId': task_id,
                'error': str(e),
                'status': '执行出错'
            }
            yield f"data: {json.dumps(error_status)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/host-status', methods=['GET'])
def host_status():
    node_counts = host_df['id'].nunique()
    data_counts = host_df.shape[0]
    return jsonify({'node_counts': node_counts, 'data_counts': data_counts})

@app.route('/api/guest-nodes-count', methods=['GET'])
def guest_nodes_count():
    num_guest_nodes = guest_df['id'].nunique()
    num_guest_data_counts = guest_df.shape[0]
    return jsonify({'count': num_guest_nodes, 'data_counts': num_guest_data_counts})

# 路由定义
@app.route('/')
def default_redirect():
    return redirect('/index.html')

@app.route('/index.html')
def index_page():
    return render_template('index.html')

@app.route('/intro.html')
def intro_page():
    return render_template('intro.html')

@app.route('/health_monitor.html')
def health_monitor_page():
    return render_template('health_monitor.html')

@app.route('/maintenance_optimization.html')
def maintenance_optimization_page():
    return render_template('maintenance_optimization.html')

@app.route('/quality_control_defect_prediction.html')
def quality_control_defect_prediction_page():
    return render_template('quality_control_defect_prediction.html')

@app.route('/federal_learning.html')
def federal_learning_page():
    return render_template('federal_learning.html')

@app.route('/LLM.html')
def llm_page():
    return render_template('LLM.html')

@app.route('/figures/<filename>')
def serve_figure(filename):
    return send_from_directory(app.config['FIGURES_FOLDER'], filename)

@app.route('/frontend/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # 初始化数据库
    try:
        init_db()
        print("数据库初始化成功")
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
    
    # 初始化FATE服务
    try:
        initialize_fate_services()
        print("FATE服务初始化成功")
    except Exception as e:
        print(f"FATE服务初始化失败: {str(e)}")
    
    print("正在启动服务器...")
    print("请访问: http://localhost:5000")
    # 启动服务器，使用 5000 端口，允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

