import json
import re
import chardet
import os
from tqdm import tqdm  # 添加进度条

def clean_text(text):
    """清理文本"""
    # 移除特殊字符
    text = text.replace('\u3000', ' ')  # 移除全角空格
    
    # 处理换行符
    # 1. 将连续的换行符替换为单个换行符
    text = re.sub(r'\n+', '\n', text)
    # 2. 移除行首和行尾的空白字符
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # 3. 移除只包含空白字符的行
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    
    # 清理其他特殊字符，保留必要的标点符号
    # text = re.sub(r'[^\u4e00-\u9fff\w\s.,?!;:，。？！；：""''《》「」\-\n]', '', text)
    
    # 处理可能出现的多余空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def process_novel_text(text, max_length=2048):
    """处理小说文本，生成训练样本"""
    # 清理文本
    text = clean_text(text)
    
    # 按章节分割
    chapters = re.split(r'\n第\d+章\s*', text)
    
    training_samples = []
    for chapter in chapters:
        if not chapter.strip():
            continue
        
        # 将章节内容按句子分割
        sentences = re.split(r'([。！？])', chapter)
        current_text = ""
        
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                # 将句子和标点符号组合
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                if not sentence.strip():
                    continue
                
                current_text += sentence
                
                # 当累积的文本长度接近max_length时，创建一个样本
                if len(current_text) >= max_length:
                    training_samples.append({
                        "text": current_text[:max_length]
                    })
                    # 保留最后一句作为下一个样本的开始，保持连贯性
                    current_text = sentence
        
        # 处理剩余的文本
        if current_text and len(current_text.strip()) > 50:  # 确保样本长度足够
            training_samples.append({
                "text": current_text
            })
    
    return training_samples

def read_file_with_encoding(file_path):
    """使用正确的编码读取文件"""
    # 首先尝试检测编码
    encoding = detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果检测失败，尝试其他常见编码
        encodings = ['gbk', 'gb2312', 'gb18030', 'big5', 'utf-8']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                print(f"成功使用 {enc} 编码读取文件: {os.path.basename(file_path)}")
                return content
            except UnicodeDecodeError:
                continue
    
    print(f"警告：无法读取文件 {file_path}")
    return None

def main():
    folder_path = '小说训练素材/style_1'  # 文件夹路径
    all_training_samples = []
    
    # 获取文件夹中所有的txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 使用tqdm显示处理进度
    for file_name in tqdm(txt_files, desc="处理文件"):
        file_path = os.path.join(folder_path, file_name)
        
        # 读取文件内容
        novel_text = read_file_with_encoding(file_path)
        if novel_text is None:
            continue
            
        # 处理文本并生成训练数据
        samples = process_novel_text(novel_text)
        all_training_samples.extend(samples)
        
        print(f"从 {file_name} 生成了 {len(samples)} 个训练样本")
    
    # 保存所有训练数据
    with open('training_data.json', 'w', encoding='utf-8') as f:
        for sample in all_training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"总共生成了 {len(all_training_samples)} 个训练样本")

if __name__ == "__main__":
    main() 