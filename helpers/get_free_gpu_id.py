import os,sys
import subprocess
import pandas as pd

TH = 0.1

def fetch_id():
    gpu_device=-1
    try:
        fname = '/tmp/query.csv'
        cmd_list = f'nvidia-smi --format=csv --query-gpu=memory.total,memory.free,memory.used,pci.bus_id,index --filename={fname}'.split(' ')
        subprocess.check_output(cmd_list)
        if not os.path.exists(fname):
            raise ValueError("csv file not found")
        df = pd.read_csv(fname)
        df['gpu_mem_total']=df['memory.total [MiB]'].apply(lambda x: int(x.split(' ')[0]))
        df['gpu_mem_used']=df[' memory.used [MiB]'].apply(lambda x: int(x.split(' ')[1]))
        df['gpu_usage_prct']=df['gpu_mem_used']/df['gpu_mem_total']
        df['gpu_id']=df[' index']
        #print(df)
        df = df.sort_values('gpu_usage_prct')
        avail = df[df.gpu_usage_prct < TH].reset_index()
        if len(avail)>0:
            gpu_device = avail.loc[0,'gpu_id']
    except:
        pass

    return int(gpu_device)

def set_gpu():
    free_gpu=fetch_id()
    if free_gpu<0:
        print("No GPU devices with sufficient memory. Try again when its free")
        sys.exit(0)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(free_gpu)
    print(f'Running on GPU ID: {free_gpu}')

if __name__=='__main__':
    gpu_device = fetch_id()
    print(f'gpu_device {gpu_device}')