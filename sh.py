

import subprocess

def get_embedding_sh(net_type_i,data_type_i):
    script_path = "/biobert-pytorch/embedding/getbiovec.sh"  # 
    # Use the subprocess.run method to execute shell scripts, !!!!!
    # or use "bash" on terminal command line to run shell scripts !!!!
    result = subprocess.run(["bash", script_path, net_type_i, data_type_i], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)