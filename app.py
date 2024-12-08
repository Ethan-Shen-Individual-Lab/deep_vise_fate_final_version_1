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
import os

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import (
    HeteroSecureBoost,
    Reader,
    Evaluation,
    PSI
)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import (
    HeteroSecureBoost,
    Reader,
    Evaluation,
    PSI
)
import numpy as np




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
    'connect_timeout': 10000
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
    # 读取原始数据集
    df = pd.read_csv("dataset/Companies_Data12.csv", encoding='utf-8')
    print(f"Successfully read the original dataset. Shape: {df.shape}")
    df['operation_status'] = df['operation_status'].replace({'Fault': 1, 'Warnin': 1, 'Normal': 0})
    
    print(f"Successfully split and saved datasets:")
    
except Exception as e:
    print(f"Error in reading or splitting dataset: {str(e)}")
    df = pd.DataFrame()


def calculate_health_index(subset):
    """
    计算设备健康指标
    """
    try:
        # 使用所有数值列作为特征
        numeric_columns = subset.select_dtypes(include=['float64', 'int64']).columns
        features = [col for col in numeric_columns if col not in ['company_id', 'device_id', 'time', 'operation_status']]
        
        # 计算基本统计量
        stats = {}
        for feature in features:
            feature_data = subset[feature].values
            stats[f'{feature}_mean'] = np.mean(feature_data)
            stats[f'{feature}_std'] = np.std(feature_data)
            stats[f'{feature}_max'] = np.max(feature_data)
            stats[f'{feature}_min'] = np.min(feature_data)
        
        return stats
        
    except Exception as e:
        print(f"计算健康指标时出错: {str(e)}")
        return {}

def process_and_train_data(df, use_local_only=False, num_nodes=None):
    """
    处理数据并训练模型
    Args:
        df: 数据框
        use_local_only: 是否只使用本地数据（company_id=1）
        num_nodes: 当use_local_only=False时，需要使用的外部节点数量
    """
    try:
        results = []  # 用于存储每个计算结果
        
        # 根据训练模式选择数据
        if use_local_only:
            # 只使用本地数据(company_id=1)
            selected_companies = [1]
        else:
            # 使用本地数据加上随机选择的外部节点数据
            available_companies = sorted(list(df['company_id'].unique()))
            available_companies.remove(1)  # 移除本地公司ID
            if num_nodes and num_nodes > 0:
                # 随机选择指定数量的外部公司
                selected_external = np.random.choice(available_companies, 
                                                   size=min(num_nodes, len(available_companies)), 
                                                   replace=False)
                selected_companies = [1] + list(selected_external)  # 确保包含本地数据
            else:
                selected_companies = [1]  # 如果未指定节点数，默认只用本地数据
        
        # 遍历选中的公司和它们的设备
        for i in selected_companies:
            company_devices = df[df['company_id'] == i]['device_id'].unique()
            for j in company_devices:
                subset = df.loc[(df['company_id'] == i) & (df['device_id'] == j), :]
                if not subset.empty:
                    result = calculate_health_index(subset)
                    if result:  # 只有当计算成功时才添加结果
                        result['time'] = subset['time'].iloc[0]
                        result['company_id'] = i
                        result['device_id'] = j
                        result['target'] = subset['operation_status'].iloc[0]
                        results.append(result)

        if not results:
            raise Exception("没有可用的处理结果")

        final_df = pd.DataFrame(results)
        
        if final_df.empty:
            raise Exception("没有成功处理的数据")

        # 准备训练数据
        X = final_df.drop(columns=['target','company_id','device_id','time'])
        y = final_df['target']

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 准备XGBoost数据格式
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)

        # 设置模型参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 12,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'colsample_bytree': 0.8,
            'subsample': 0.8
        }

        # 训练模型
        model = xgb.train(params, dtrain, num_boost_round=100)

        # 模型评估
        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_rep)

        # 预测所有设备的状态
        X_all_scaled = scaler.transform(X)
        dall = xgb.DMatrix(X_all_scaled)
        all_predictions = model.predict(dall)
        
        # 找出预警设备
        warning_devices = final_df[all_predictions > 0.5]['device_id'].unique().tolist()

        # 添加使用的数据来源信息
        data_source_info = "本地数据" if use_local_only else f"联邦学习（使用{len(selected_companies)}个节点）"

        return {
            'success': True,
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'warning_devices': warning_devices,
            'message': f'训练完成（使用{data_source_info}）',
            'data_source': data_source_info
        }

    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        return {
            'success': False,
            'message': str(e),
            'warning_devices': []
        }














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

def run_federated_learning(features=None, label=None):
    try:
        # 数据准备
        guest_data_path = '../111/dataset/guest_data.csv'
        host_data_path = '../111/dataset/host_data.csv'

        pipeline = FateFlowPipeline().set_parties(guest="9999", host="10000")
        
        # 配置数据读取和处理
        guest_meta = {
            "delimiter": ",", 
            "dtype": "float64", 
            "label_type": "int64",
            "label_name": label if label else "operation_status", 
            "match_id_name": "id",
            "selected_features": features if features else None
        }
        host_meta = {
            "delimiter": ",", 
            "input_format": "dense", 
            "match_id_name": "id",
            "selected_features": features if features else None
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

#可选fate模型线上训练
def heteroxgboost:
    reader = Reader()
    train_data = reader.read(data_path)
    
    # 2. 使用 FATE 的 XGBoost 模型
    model = HeteroXGBoost()
    
    # 3. 配置 pipeline，添加步骤
    pipeline = Pipeline()
    pipeline.add_component(reader, name='reader', input_data=train_data)
    pipeline.add_component(model, name='hetero_xgb')
    
    # 4. 运行 pipeline
    pipeline.fit()
    

@app.route('/federal_learning/host-status', methods=['GET'])
def host_status():
    try:
        # 获取本地公司(company_id=1)的数据
        local_data = df[df['company_id'] == 1]
        node_counts = len(local_data['device_id'].unique())  # 使用device_id作为节点
        data_counts = len(local_data)
        return jsonify({'node_counts': node_counts, 'data_counts': data_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/federal_learning/guest-nodes-count', methods=['GET'])
def guest_nodes_count():
    try:
        # 获取除了company_id=1以外的公司数量（因为1是本地公司）
        num_guest_nodes = len(df[df['company_id'] != 1]['company_id'].unique())
        num_guest_data_counts = len(df[df['company_id'] != 1])
        return jsonify({'count': num_guest_nodes, 'data_counts': num_guest_data_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/federal_learning/features', methods=['GET'])
def get_features():
    try:
        # 获取DataFrame的所有列名
        features = df.columns.tolist()
        # 排除不需要的列（如id等）
        exclude_columns = ['company_id', 'device_id', 'time', 'operation_status']
        features = [col for col in features if col not in exclude_columns]
        return jsonify({
            'success': True, 
            'features': features,
            'label_features': ['operation_status']  # 标签变量固定为operation_status
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/federal_learning/connect_fate', methods=['POST'])
def connect_fate():
    try:
        data = request.get_json()
        mode = data.get('mode')
        if mode == 'federated':
            initialize_fate_services()
            return jsonify({'success': True, 'message': '成功连接到FATE服务器'})
        return jsonify({'success': False, 'message': '不支持的模式'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/federal_learning/run_federal_learning', methods=['POST'])
def run_federal_learning_api():
    try:
        data = request.get_json()
        task_id = data.get('taskId')
        is_standard_model = data.get('isStandardModel', False)
        
        if is_standard_model:
            # 使用标准模型的预设特征
            features = [
                'temperature', 'pressure', 'vibration', 'noise', 
                'current', 'voltage', 'speed', 'load'
            ]
            label = 'operation_status'
        else:
            # 使用用户选择的特征
            features = data.get('features', [])
            label = data.get('label')
        
        success, message = run_federated_learning(features=features, label=label)
        return jsonify({
            'success': success,
            'message': message,
            'taskId': task_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'taskId': data.get('taskId')
        }), 500

@app.route('/federal_learning/process_and_train', methods=['POST'])
def process_and_train_api():
    try:
        data = request.get_json()
        
        # 从请求中获取训练模式和节点数
        use_local_only = data.get('use_local_only', False)
        num_nodes = data.get('num_nodes', 0) if not use_local_only else 0
        
        # 调用训练函数
        result = process_and_train_data(df, use_local_only=use_local_only, num_nodes=num_nodes)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'accuracy': result['accuracy'],
                'warning_devices': result['warning_devices'],
                'data_source': result['data_source']
            })
        else:
            return jsonify({
                'success': False,
                'message': result['message'],
                'warning_devices': []
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'warning_devices': []
        }), 500

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






def calculate_health_index(df, variables, accel_column, weight_column):
    """
    计算健康检测指数，包括光谱峰度、加权归一化平方包络之和、光谱负熵、光谱基尼指数等
    
    :param df: 数据集（Pandas DataFrame）
    :param variables: 要计算的变量列表
    :param accel_column: 加速度信号的列名，用于归一化计算
    :param weight_column: 权重信号的列名，用于计算加权结果
    :return: 每个变量的��果字典
    """
    results = {}  # 用于存储每个变量的结果
    
    for variable in variables:
        # 获取当前变量的信号数据
        signal = df[variable]
        
        # 计算二阶矩（m_2）和四阶矩（m_4）
        m_2 = np.mean(np.abs(signal)**2)  # 二阶矩，信号平方的均值
        m_4 = np.mean(np.abs(signal)**4)  # 四阶矩，信号的四次方的均值
        
        # 计算光谱峰度
        spectral_kurtosis = (m_4 / m_2**2) - 2
        results[f'{variable}_spectral_kurtosis'] = spectral_kurtosis
        
        # 计算加权归一化平方包络之和
        accel_signal = df[accel_column]  # 使用指定的加速度信号列
        accel_signal_norm = np.abs(accel_signal) / np.sum(np.abs(accel_signal))  # 归一化
        weighted_squared_sum = np.sum(accel_signal_norm * accel_signal_norm) - 2
        results[f'{variable}_weighted_squared_sum'] = weighted_squared_sum
        
        # 计算光谱负熵
        spectral_entropy = np.sum(
            accel_signal_norm * np.log(accel_signal_norm + 1e-10)  # 避免 log(0)
        )
        results[f'{variable}_spectral_entropy'] = -spectral_entropy
        
        # 计算光谱基尼指数
        N = len(accel_signal_norm)
        sorted_norm = np.sort(accel_signal_norm)
        gini_index = 1 - 2 * np.sum(
            (np.arange(1, N + 1) / N) * sorted_norm
        )
        results[f'{variable}_spectral_gini_index'] = gini_index
        
        # 计算光谱平滑度指数的倒数
        smoothness = np.linalg.norm(accel_signal_norm, ord=1) / (
            np.sqrt(N) * np.prod(accel_signal_norm + 1e-10)**(1/N)
        )
        spectral_smoothness_reciprocal = 1 / smoothness
        results[f'{variable}_spectral_smoothness_reciprocal'] = spectral_smoothness_reciprocal
        
        # 计算加权结果
        weight_signal = df[weight_column]  # 使用指定的权重信号列
        omega_1 = np.abs(weight_signal) / np.sum(np.abs(weight_signal))  # 权重归一化
        weighted_result = np.sum(omega_1 * weight_signal) - 2
        results[f'{variable}_weighted_result'] = weighted_result
        
        # 打印中间结果验证
        print(f"Variable: {variable}")
        print(f"  m_2: {m_2}, m_4: {m_4}")
        print(f"  Spectral Kurtosis: {spectral_kurtosis}")
        print(f"  Weighted Squared Sum: {weighted_squared_sum}")
        print(f"  Spectral Entropy: {spectral_entropy}")
        print(f"  Spectral Gini Index: {gini_index}")
        print(f"  Spectral Smoothness Reciprocal: {spectral_smoothness_reciprocal}")
        print(f"  Weighted Result: {weighted_result}")
        print("-----")
    
    return results

# 使用示例
variables = ['motor_temp', 'motor_speed_act',
       'motor_speed_actual', 'bearing_temp', 'rear_bearing_temp',
       'bearing_accel_value', 'bearing_freq_value', 'bearing_accel_peak',
       'bearing_accel_value2', 'bearing_freq_value2', 'bearing_temp_value',
       'motor_bearing_front_temp', 'motor_bearing_front_freq',
       'motor_bearing_front_accel', 'motor_bearing_front_accel_peak',
       'motor_bearing_rear_temp', 'motor_bearing_rear_freq',
       'motor_bearing_rear_accel', 'motor_bearing_rear_accel_peak']  # 要计算的多个变量
accel_column = 'bearing_accel_value'  # 加速度信号列
weight_column = 'motor_bearing_front_temp'  # 权重信号列
