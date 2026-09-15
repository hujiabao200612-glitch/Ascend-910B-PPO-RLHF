# -*- coding: utf-8 -*-
"""eval_mbpp.py — Google 官方 MBPP (427 题) 标准 Pass@1 评测器（严格遵循官方 Prompt 规范）"""
import os, sys, json, time, argparse, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--data", type=str, default="mbpp_sanitized_official.json", help="MBPP 数据路径")
    parser.add_argument("--output", type=str, default="samples_mbpp.jsonl")
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model)
    model_name = os.path.basename(os.path.normpath(model_dir))
    
    with open(args.data, "r", encoding="utf-8") as f:
        problems = json.load(f)
        
    print(f"\n=================================================================")
    print(f"[MBPP Eval Engine] 启动 427 题 Google MBPP 官方标准评测")
    print(f"  - 评测模型: {model_dir}")
    print(f"  - 题目总数: {len(problems)} 题")
    print(f"=================================================================")
    
    # 1. 构造标准 Prompt（遵循 Google 官方基准：题目描述 + 第 1 条测试用例作为函数签名规范）
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompts = []
    tests_list = []
    
    for item in problems:
        p = item["prompt"]
        tests = item.get("test_list", [])
        imports = item.get("test_imports", [])
        
        # 组装完整的 pytest 测试代码（必须包含 from solution import *）
        full_test_code = "from solution import *\n" + "\n".join(imports) + "\n"
        for idx, assertion in enumerate(tests):
            full_test_code += f"\ndef test_{idx}():\n    {assertion}\n"
        tests_list.append(full_test_code)
        
        # 官方标准 Prompt 构造
        first_test = tests[0] if len(tests) > 0 else ""
        prompt_content = f"Write a Python function to solve the following problem:\n{p}\nYour code should satisfy the following assertion:\n```python\n{first_test}\n```\nEnclose your completed code in ```python ```."
        dialog = [{"role": "user", "content": prompt_content}]
        formatted = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
        
    # 2. vLLM 批量贪心推理
    print(f"\n>>> [1/2] 启动 vLLM 批量推理生成 ({len(prompts)} 题) ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir, tensor_parallel_size=1, gpu_memory_utilization=0.5, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=512, top_p=1.0)
    outputs = llm.generate(prompts, sp)
    generated_texts = [o.outputs[0].text for o in outputs]
    print(f"    vLLM 推理完成，耗时: {time.time()-t0:.2f}s")
    
    # 保存结果
    records = []
    for i in range(len(problems)):
        records.append({
            "task_id": problems[i]["task_id"],
            "prompt": problems[i]["prompt"],
            "response": generated_texts[i],
            "test": tests_list[i]
        })
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"\n>>> [2/2] 启动隔离子进程进行 64 线程沙箱并发判分 ...")
    eval_cmd = [
        sys.executable, "-c",
        f"""
import json, sys, os
from concurrent.futures import ThreadPoolExecutor

cur_dir = os.path.abspath('.')
sys.path.insert(0, os.path.join(cur_dir, 'pipeline'))
sys.path.insert(0, os.path.join(cur_dir, 'rewards'))
from code_rlvr import compute_score

with open('{args.output}', 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]

def _eval(item):
    s = compute_score(data_source='kodcode', solution_str=item['response'], ground_truth=item['test'])
    return s >= 1.0

with ThreadPoolExecutor(max_workers={args.workers}) as ex:
    passed = list(ex.map(_eval, items))

total = len(passed)
succ = sum(passed)
print('=' * 65)
print('            Google MBPP 官方评测最终成绩 (Pass@1 Report)')
print('=' * 65)
print(f'  模型名称:         {model_name}')
print(f'  测试题目总数:     {{total}}')
print(f'  Pass@1 满分通过率: {{succ/total*100:.2f}}% ({{succ}}/{{total}} 题)')
print('=' * 65)
"""
    ]
    subprocess.run(eval_cmd, check=True)

if __name__ == "__main__":
    main()
