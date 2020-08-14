# trainer.py
from collections import Counter
import os
import sys
import time
import ray
import pdb
import numpy as np

num_cpus = int(sys.argv[1])

# ip_head and redis_passwords are set by ray cluster shell scripts
print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0], redis_password=os.environ["redis_password"])

print("Nodes in the Ray cluster:")
print(ray.nodes())
a = np.zeros((10000, 2000)) 
a_id = ray.put(a)

@ray.remote
def f(a,i):
    print('Computing {} on {}'.format(i,ray.services.get_node_ip_address()), flush=True)
    time.sleep(np.random.uniform(0, 5))
    return time.time()
    

start_time = time.time()

remaining_result_ids = [f.remote(a_id,i) for i in range(num_cpus*2)]

# Get the results.
results = []
while len(remaining_result_ids) > 0:
    ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
    result_id = ready_result_ids[0]
    result = ray.get(result_id)
    results.append(result)
    print('Processing result which finished after {} seconds.'
          .format(result - start_time))    

end_time = time.time()
duration = end_time - start_time
print('Execution time was : {:.2f}'.format(duration))
    