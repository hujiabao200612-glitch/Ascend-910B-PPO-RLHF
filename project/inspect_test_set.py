# -*- coding: utf-8 -*-
"""inspect_test_set.py — 快速探查 200 题独立测试集 (test.parquet) 结构与样本"""
import os
import sys
import pandas as pd

def main():
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/rlvr/test.parquet"
    if not os.path.exists(parquet_path):
        print(f"[ERROR] 找不到测试集文件: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print("=" * 70)
    print(f"📊 【测试集总览】数据文件: {parquet_path}")
    print(f"  - 题目总数: {len(df)} 题")
    print(f"  - 列字段列表: {list(df.columns)}")
    print("=" * 70)

    prompt_lens = []
    test_lens = []
    assert_counts = []

    for _, row in df.iterrows():
        p = row['prompt']
        p_str = p[0]['content'] if isinstance(p, (list, tuple)) and len(p) > 0 and isinstance(p[0], dict) else str(p)
        prompt_lens.append(len(p_str))

        rm = row.get('reward_model', {})
        t_str = rm.get('ground_truth', '') if isinstance(rm, dict) else str(rm)
        test_lens.append(len(t_str))
        assert_counts.append(t_str.count("assert "))

    print("\n📈 【数据集特征分布】")
    print(f"  - Prompt 字符长度:  平均 {sum(prompt_lens)/len(prompt_lens):.1f} | 最小 {min(prompt_lens)} | 最大 {max(prompt_lens)}")
    print(f"  - 单元测试代码长度: 平均 {sum(test_lens)/len(test_lens):.1f} | 最小 {min(test_lens)} | 最大 {max(test_lens)}")
    print(f"  - 每题断言数 (assert): 平均 {sum(assert_counts)/len(assert_counts):.1f} 条 | 范围 [{min(assert_counts)}, {max(assert_counts)}]")

    print("\n🔍 【题目真实样本抽样探查】")
    for i in range(min(2, len(df))):
        row = df.iloc[i]
        p = row['prompt']
        p_str = p[0]['content'] if isinstance(p, (list, tuple)) and len(p) > 0 and isinstance(p[0], dict) else str(p)
        rm = row.get('reward_model', {})
        t_str = rm.get('ground_truth', '') if isinstance(rm, dict) else str(rm)
        extra = row.get('extra_info', {})
        qid = extra.get('question_id', f'sample_{i}')
        sol = extra.get('solution', '')

        print("-" * 70)
        print(f"📌 [样例 {i+1}] 题目唯一标识 (ID): {qid}")
        print("【用户提示词 (Prompt)】:")
        print(p_str.strip()[:400] + ("\n... [省略后续内容]" if len(p_str) > 400 else ""))
        print("\n【沙箱测试用例 (Ground Truth Tests)】:")
        print(t_str.strip()[:400] + ("\n... [省略后续内容]" if len(t_str) > 400 else ""))
        if sol:
            print("\n【参考标准解答 (Reference Solution)】:")
            print(sol.strip()[:300] + ("\n... [省略后续内容]" if len(sol) > 300 else ""))
    print("=" * 70)

if __name__ == "__main__":
    main()
