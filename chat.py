# Command line tool to chat with the Llama2 model, using a GPU.

from typing import Optional
from llama import Llama
import fire
import torch
import os
import time


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
):
    if os.name == 'nt':
        # Use the gloo backend instead of nccl, as nccl was not supported by pytorch on windows
        torch.distributed.init_process_group('gloo')

        # If you have multiple GPUs, you can set a different device than 0
        # torch.cuda.device_count()
        torch.cuda.set_device('cuda:0')

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    while True:
        value = input("Enter prompt (or type exit): ")
        if value == 'exit':
            break
        dialogs = [[{"role": "user", "content": value}]]

        t = time.process_time()
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for result in results:
            print("Answer: ", result['generation']['content'])
            print("Elapsed time: ", time.process_time() - t)


if __name__ == "__main__":
    fire.Fire(main)
