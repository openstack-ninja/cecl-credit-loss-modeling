##
# python run_cecl_pipeline.py --data_dir ./data/fannie --gse fannie --cores 64 --output fannie_results.parquet
# python run_cecl_pipeline.py --data_dir ./data/freddie --gse freddie --cores 64 --output freddie_results.parquet
##

import os
import glob
import argparse
import time
import psutil
import cudf
import pandas as pd
from joblib import Parallel, delayed

# --- CONFIGURATION & MAPPINGS ---
MAPPINGS = {
    'fannie': {
        'acq': {0: 'loan_id', 2: 'seller', 4: 'orig_upb', 10: 'oltv'},
        'perf': {0: 'loan_id', 1: 'period', 3: 'current_upb', 8: 'delq_status'},
        'date_fmt': '%m%Y',
        'prefix': 'Acquisition'
    },
    'freddie': {
        'acq': {0: 'loan_id', 4: 'orig_upb', 11: 'ocltv'},
        'perf': {0: 'loan_id', 1: 'period', 3: 'current_upb', 8: 'zero_balance_code'},
        'date_fmt': '%Y%m',
        'prefix': 'historical_data'
    }
}

def get_args():
    parser = argparse.ArgumentParser(description="DGX Spark: High-Performance CECL Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .txt files")
    parser.add_argument("--gse", type=str, choices=['fannie', 'freddie'], required=True)
    parser.add_argument("--cores", type=int, default=64, help="CPU cores for Grace (default: 64)")
    parser.add_argument("--output", type=str, default="cecl_results.parquet")
    parser.add_argument("--log", type=str, default="pipeline_metrics.csv")
    return parser.parse_args()

def log_metrics(log_file, quarter, duration, mem_usage):
    """Logs performance metrics for audit and optimization."""
    df_log = pd.DataFrame([{
        'timestamp': pd.Timestamp.now(),
        'quarter': quarter,
        'duration_sec': round(duration, 2),
        'max_mem_gb': round(mem_usage / (1024**3), 2)
    }])
    df_log.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

def cecl_logic_wrapper(df_chunk):
    """
    Apply the wikihifi model logic here.
    Note: Ensure 'from cecl_model import ...' is available in your path.
    """
    # Placeholder: Current UPB * 1.5% static reserve (replace with model.predict)
    df_chunk['expected_loss'] = df_chunk['current_upb'] * 0.015
    return df_chunk

def process_quarter(acq_p, perf_p, gse_type, cores):
    start_time = time.time()
    m = MAPPINGS[gse_type]
    
    # 1. GPU Load & Join (Blackwell Unified Memory)
    acq = cudf.read_csv(acq_p, sep='|', header=None, usecols=list(m['acq'].keys()))
    perf = cudf.read_csv(perf_p, sep='|', header=None, usecols=list(m['perf'].keys()))
    
    acq.columns = [m['acq'][c] for c in acq.columns]
    perf.columns = [m['perf'][c] for c in perf.columns]
    
    merged = acq.merge(perf, on='loan_id')
    merged['period'] = cudf.to_datetime(merged['period'], format=m['date_fmt'])
    
    # 2. Parallel Model Execution (Grace CPU)
    df_pd = merged.to_pandas()
    shards = [df_pd[i::cores] for i in range(cores)]
    results = Parallel(n_jobs=cores)(delayed(cecl_logic_wrapper)(s) for s in shards)
    
    duration = time.time() - start_time
    mem = psutil.Process().memory_info().rss
    return pd.concat(results), duration, mem

def main():
    args = get_args()
    m = MAPPINGS[args.gse]
    
    # Find files based on GSE naming conventions
    pattern = os.path.join(args.data_dir, f"{m['prefix']}_*.txt")
    acq_files = sorted(glob.glob(pattern))
    
    if not acq_files:
        print(f"❌ No files found for {args.gse} in {args.data_dir}")
        return

    print(f"🚀 DGX Spark: Starting {args.gse.upper()} CECL Pipeline...")
    all_outputs = []

    for acq_path in acq_files:
        # Determine performance file path
        if args.gse == 'fannie':
            perf_path = acq_path.replace("Acquisition", "Performance")
        else:
            perf_path = acq_path.replace(".txt", "_time.txt")
            
        if os.path.exists(perf_path):
            q_name = os.path.basename(acq_path)
            print(f"🔄 Processing {q_name}...")
            
            res_df, duration, mem = load_and_process(acq_path, perf_path, args.gse, args.cores)
            all_outputs.append(res_df)
            
            log_metrics(args.log, q_name, duration, mem)
            print(f"✅ Finished in {duration:.2f}s")

    # Final consolidate and save
    final_df = pd.concat(all_outputs)
    final_df.to_parquet(args.output, engine='pyarrow')
    print(f"🏁 Pipeline Complete. Total Rows: {len(final_df)} | Saved to: {args.output}")

if __name__ == "__main__":
    main()
