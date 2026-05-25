"""
Ollama API测试脚本
验证Ollama服务和大模型是否正常工作
AI辅助生成
"""
import requests
import sys
from config import OLLAMA_BASE_URL, DEFAULT_MODEL


def test_ollama_connection():
    """测试Ollama服务连接"""
    print(f"正在测试Ollama连接: {OLLAMA_BASE_URL}")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama服务连接成功!")
            print(f"  已安装的模型数量: {len(models)}")
            for model in models:
                print(f"    - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"✗ Ollama服务返回异常状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ 无法连接到Ollama服务，请确保Ollama正在运行")
        print(f"  提示: 在终端运行 'ollama serve' 启动服务")
        return False
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {str(e)}")
        return False


def test_model_inference(model_name: str = DEFAULT_MODEL):
    """测试模型推理"""
    print(f"\n正在测试模型推理: {model_name}")
    try:
        payload = {
            "model": model_name,
            "prompt": "请用一句话介绍一下自己",
            "stream": False
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 模型推理测试成功!")
            print(f"  回答: {result.get('response', '')[:200]}...")
            return True
        else:
            print(f"✗ 模型推理返回异常状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 模型推理测试失败: {str(e)}")
        return False


def check_model_downloaded(model_name: str = DEFAULT_MODEL):
    """检查模型是否已下载"""
    print(f"\n检查模型 '{model_name}' 是否已下载...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if any(model_name in name for name in model_names):
                print(f"✓ 模型 '{model_name}' 已下载")
                return True
            else:
                print(f"✗ 模型 '{model_name}' 未找到")
                print(f"  请运行: ollama pull {model_name}")
                return False
    except Exception as e:
        print(f"✗ 检查模型时发生错误: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Ollama API 测试脚本")
    print("=" * 50)

    if not test_ollama_connection():
        sys.exit(1)

    model_name = DEFAULT_MODEL
    check_model_downloaded(model_name)

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)