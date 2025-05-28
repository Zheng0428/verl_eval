import os
import json
import random
import datasets
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import argparse

def get_difficulty_level(solve_rate):
    """
    根据llama8b_solve_rate返回对应的难度等级
    
    参数:
        solve_rate (float): 0-100之间的解题成功率
        
    返回:
        int: 难度等级数字(1-8)
    """
    # 确保solve_rate在有效范围内
    solve_rate = solve_rate * 100
    solve_rate = max(0, min(100, solve_rate))
    
    if solve_rate >= 85:
        return 1
    elif solve_rate >= 70:
        return 2
    elif solve_rate >= 55:
        return 3
    elif solve_rate >= 40:
        return 4
    elif solve_rate >= 25:
        return 5
    elif solve_rate >= 15:
        return 6
    elif solve_rate >= 5:
        return 7
    else:
        return 8

def process_dataset(dataset, test_size=0.1, random_state=42):
    """处理数据集并按难度分级"""
    
    # 将数据集转换为DataFrame以便处理
    df = pd.DataFrame(dataset)
    
    # 添加难度等级列
    df['level'] = df['llama8b_solve_rate'].apply(get_difficulty_level)
    
    # 初始化结果存储
    train_data = []
    test_data = []
    
    # 按难度等级分组处理
    for level in range(1, 9):
        level_df = df[df['level'] == level]
        
        if len(level_df) > 0:
            # 按9:1分割训练集和测试集
            level_train, level_test = train_test_split(
                level_df, test_size=test_size, random_state=random_state
            )
            
            train_data.append(level_train)
            test_data.append(level_test)
            
            print(f"难度等级 {level}: 总数 {len(level_df)}, 训练集 {len(level_train)}, 测试集 {len(level_test)}")
    
    # 合并所有难度等级的数据
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    # 转换回数据集格式
    train_dataset = datasets.Dataset.from_pandas(train_df)
    test_dataset = datasets.Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset

def prepare_for_training(example, idx, split, data_source="big_math"):
    """准备最终训练格式"""
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    
    question_raw = example['problem']
    question = question_raw + ' ' + instruction_following
    answer = example['answer']
    question_level = example['level']
    llama8b_solve_rate = example['llama8b_solve_rate']
    
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer,
            "question": question_raw,
            'level': question_level,
            'llama8b_solve_rate': llama8b_solve_rate,
            'domain': example.get('domain', ''),
            'source': example.get('source', '')
        }
    }
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='处理Big-Math-RL-Verified数据集')
    parser.add_argument('--local_dir', type=str, default='./big_math_processed', 
                        help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.1, 
                        help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='随机种子')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.local_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified")['train']
    
    # 处理数据集，划分难度等级和训练/测试集
    print("处理数据集...")
    train_dataset, test_dataset = process_dataset(
        dataset, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    # 整理数据格式
    print("整理训练格式...")
    train_dataset = train_dataset.map(
        lambda example, idx: prepare_for_training(example, idx, 'train'), 
        with_indices=True
    )
    test_dataset = test_dataset.map(
        lambda example, idx: prepare_for_training(example, idx, 'test'), 
        with_indices=True
    )
    
    # 保存到本地
    print("保存数据...")
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    
    # 按难度等级分别保存
    for level in range(1, 9):
        # 训练集
        level_train = train_dataset.filter(lambda x: x['extra_info']['level'] == level)
        if len(level_train) > 0:
            level_train.to_parquet(os.path.join(args.local_dir, f'train_level_{level}.parquet'))
        
        # 测试集
        level_test = test_dataset.filter(lambda x: x['extra_info']['level'] == level)
        if len(level_test) > 0:
            level_test.to_parquet(os.path.join(args.local_dir, f'test_level_{level}.parquet'))
    
    # 输出统计信息
    print(f"处理完成！")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"数据已保存至: {args.local_dir}")
    
    # 输出各难度等级的分布统计
    level_stats = {}
    for level in range(1, 9):
        train_count = len(train_dataset.filter(lambda x: x['extra_info']['level'] == level))
        test_count = len(test_dataset.filter(lambda x: x['extra_info']['level'] == level))
        total = train_count + test_count
        level_stats[level] = {
            'train': train_count,
            'test': test_count,
            'total': total,
            'percentage': round(total / (len(train_dataset) + len(test_dataset)) * 100, 2)
        }
    
    # 输出难度分布表格
    print("\n难度等级分布:")
    print("-" * 60)
    print(f"{'难度等级':<10}{'训练集数量':<12}{'测试集数量':<12}{'总数':<10}{'百分比 (%)':<12}")
    print("-" * 60)
    for level, stats in level_stats.items():
        print(f"{level:<10}{stats['train']:<12}{stats['test']:<12}{stats['total']:<10}{stats['percentage']:<12}")
    print("-" * 60)