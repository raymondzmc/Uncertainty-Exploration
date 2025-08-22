import datasets
from settings import settings
from huggingface_hub import login as hf_login


hf_login(token=settings.hf_access_token)
abstention_bench_data = datasets.load_dataset('facebook/AbstentionBench', trust_remote_code=True)

import pdb; pdb.set_trace()