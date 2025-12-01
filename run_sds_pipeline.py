import os
import json
import argparse
import tqdm
from dotenv import load_dotenv
from datasets import Dataset

# Import local modules
from examples.sds.sds_task import SDSTask
from shinka.annotation.trace_generator import TraceGenerator
from shinka.core.evolution import Evolution  # Adjust based on exact shinka import path
from shinka.llm.llm_client import LLMClient # Adjust based on Shinka's LLM wrapper

# Import your generation script (assuming it's in the same folder or pythonpath)
try:
    from gen_sds_dataset import sds_sample, sds_render_prompt, _instance_to_problem
    from gen_sds_dataset import make_dense_instance, make_tree_instance # explicit imports
except ImportError:
    print("❌ Could not import gen_sds_dataset.py. Make sure it is in the path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of SFT samples to generate")
    parser.add_argument("--generations", type=int, default=10, help="Evolution generations per problem")
    parser.add_argument("--pop_size", type=int, default=10, help="Evolution population size")
    parser.add_argument("--push_to", type=str, help="HuggingFace repo ID (e.g., 'Org/Dataset')")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key (optional if in env)")
    args = parser.parse_args()
    
    load_dotenv()
    
    # 1. Initialize Components
    # Use Shinka's internal LLM client for the Evolution part (uses local LLM or API)
    # We assume Shinka is configured via its standard config.yaml, or we pass basic params
    shinka_llm = LLMClient(model_name="Qwen2.5-Coder-7B-Instruct") 
    
    # Use OpenAI for the high-quality trace generation
    tracer = TraceGenerator(api_key=args.api_key)
    
    dataset_records = []
    
    print(f"🚀 Starting Pipeline: {args.samples} samples")
    
    # 2. Loop through desired samples
    for i in tqdm.tqdm(range(args.samples)):
        
        # A. Generate a Problem Instance
        # Rotate problem types to ensure diversity
        ptype = "dense" if i % 2 == 0 else "tree"
        if ptype == "dense":
            inst = make_dense_instance(n=random.randint(15, 25), seed=i)
        else:
            inst = make_tree_instance(n=random.randint(15, 25), seed=i)
            
        problem_dict = _instance_to_problem(inst, ptype, i)
        prompt_data = sds_render_prompt(problem_dict)
        
        # B. Run Shinka Evolution
        task = SDSTask(problem_dict)
        
        evolution = Evolution(
            task=task,
            llm=shinka_llm,
            population_size=args.pop_size,
            generations=args.generations,
            mutation_rate=0.5
        )
        
        # Run evolution loop
        best_code, best_fitness = evolution.run()
        
        # If evolution failed to find a valid solution, skip
        if best_fitness <= 0.01:
            continue
            
        # C. Generate Thinking Trace
        trace = tracer.generate(prompt_data["problem"], best_code)
        
        # D. Format Entry
        full_content = f"<think>\n{trace}\n</think>\n\n<code>\n{best_code}\n</code>"
        
        dataset_records.append({
            "problem_id": prompt_data["uuid"],
            "messages": [
                {"role": "user", "content": prompt_data["problem"]},
                {"role": "assistant", "content": full_content}
            ],
            "score": best_fitness,
            "type": ptype,
            # Store raw mission as string to avoid HF Arrow errors
            "mission": json.dumps(problem_dict["mission"]) 
        })
        
    print(f"✅ Generated {len(dataset_records)} valid samples.")
    
    # 3. Push to HuggingFace
    if args.push_to and len(dataset_records) > 0:
        print(f"out Pushing to {args.push_to}...")
        ds = Dataset.from_list(dataset_records)
        ds.push_to_hub(args.push_to, private=True)
        print("Done!")
    else:
        # Save locally if no repo specified
        with open("sft_dataset_local.jsonl", "w") as f:
            for r in dataset_records:
                f.write(json.dumps(r) + "\n")
        print("Saved to sft_dataset_local.jsonl")

if __name__ == "__main__":
    import random # needed for loop
    main()