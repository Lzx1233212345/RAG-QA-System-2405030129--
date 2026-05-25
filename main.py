"""
RAG智能问答系统 - 启动脚本
用于PyInstaller打包
"""
import sys
import subprocess
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHERUSAGESTATS'] = 'false'
    
    app_path = os.path.join(current_dir, 'app.py')
    command = [
        sys.executable, '-m', 'streamlit', 'run', 
        app_path, 
        '--server.headless=true',
        '--server.port=8501'
    ]
    
    subprocess.run(command)

if __name__ == '__main__':
    main()