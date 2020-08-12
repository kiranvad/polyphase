# trainer.py
from collections import Counter
import os
import sys
import time
import ray
import pdb

num_cpus = int(sys.argv[1])

# ip_head and redis_passwords are set by ray cluster shell scripts
print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0], redis_password=os.environ["redis_password"])

print("Nodes in the Ray cluster:")
print(ray.nodes())

@ray.remote
def f():
    time.sleep(1)
    return ray.services.get_node_ip_address()

for i in range(60):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(num_cpus)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)
    