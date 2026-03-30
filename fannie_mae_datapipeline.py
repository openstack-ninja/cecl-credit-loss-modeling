##
# python run_cecl.py --data_dir ./fannie_data --gpu_batch --cores 64python run_cecl.py --data_dir ./fannie_data --gpu_batch --cores 64
# python run_cecl.py --data_dir ./test_samples --cores 8 --output test_results.parquet
# python run_cecl.py --help
# nohup python run_cecl.py --data_dir ./full_dataset --gpu_batch --cores 60 > cecl_log.txt &
##

import os
import glob
import argparse
import cudf
import pandas as pd
from joblib import Parallel, delayed

def get_args():
    parser = argparse.ArgumentParser(description="DGX Spark: Fannie Mae CECL Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Fannie Mae .txt files")
    parser.add_argument("--output", type=str, default="cecl_results.parquet", help="Output Parquet filename")
    parser.add_argument("--cores", type=int, default=64, help="Number of Grace CPU cores to use (default: 64)")
    parser.add_argument("--gpu_batch", action="store_true", help="Enable GPU-accelerated joins via cuDF")
    return parser.parse_args()

# Fannie Mae Column Mappings
ACQ_MAP = {0: 'loan_id', 2: 'seller', 4: 'orig_upb', 5: 'orig_term', 10: 'oltv'}
PERF_MAP = {0: 'loan_id', 1: 'period', 3: 'current_upb', 8: 'delq_status'}

def load_and_join(acq_path, perf_path, use_gpu=True):
    """Joins Acquisition and Performance files."""
    if use_gpu:
        # Use Blackwell GPU (cuDF)
        acq = cudf.read_csv(acq_path, sep='|', header=None, usecols=list(ACQ_MAP.keys()))
        perf = cudf.read_csv(perf_path, sep='|', header=None, usecols=list(PERF_MAP.keys()))
        acq.columns = [ACQ_MAP[c] for c in acq.columns]; perf.columns = [PERF_MAP[c] for c in perf.columns]
        return acq.merge(perf, on='loan_id').to_pandas()
    else:
        # Standard Pandas fallback
        acq = pd.read_csv(acq_path, sep='|', header=None, usecols=list(ACQ_MAP.keys()), names=list(ACQ_MAP.values()))
        perf = pd.read_csv(perf_path, sep='|', header=None, usecols=list(PERF_MAP.keys()), names=list(PERF_MAP.values()))
        return acq.merge(perf, on='loan_id')

def run_cecl_logic(df_chunk):
    """Placeholder for the wikihifi/cecl-credit-loss-modeling logic."""
    # This is where you would import and call the wikihifi functions
    df_chunk['expected_loss'] = df_chunk['current_upb'] * 0.015 # 1.5% reserve example
    return df_chunk

def main():
    args = get_args()
    
    acq_files = sorted(glob.glob(os.path.join(args.data_dir, "Acquisition_*.txt")))
    perf_files = sorted(glob.glob(os.path.join(args.data_dir, "Performance_*.txt")))

    if not acq_files:
        print(f"❌ Error: No Acquisition files found in {args.data_dir}")
        return

    print(f"🚀 Starting Pipeline on DGX Spark...")
    print(f"📍 Data: {args.data_dir} | 🧠 Cores: {args.cores} | ⚡ GPU: {args.gpu_batch}")

    final_results = []
    for a, p in zip(acq_files, perf_files):
        print(f"🔄 Processing {os.path.basename(a)}...")
        
        # 1. Join (Blackwell GPU)
        data_chunk = load_and_join(a, p, use_gpu=args.gpu_batch)
        
        # 2. Parallel CECL (Grace CPU Cores)
        shards = [data_chunk[i::args.cores] for i in range(args.cores)]
        processed = Parallel(n_jobs=args.cores)(delayed(run_cecl_logic)(s) for s in shards)
        
        final_results.append(pd.concat(processed))

    # 3. Save to Parquet (Optimized for DGX NVMe)
    pd.concat(final_results).to_parquet(args.output, engine='pyarrow')
    print(f"✅ Success! Results saved to {args.output}")

if __name__ == "__main__":
    main()
