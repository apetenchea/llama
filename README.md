# Running Llama 2 Locally

[Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) is a collection
of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7B to 70B parameters.  
At the time of this writing, the official way of getting the weights is through the
[Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

## Purpose of this fork

I wanted to set aside instructions for running Llama 2 locally, with minimal dependencies.
This is my how-to guide. The original README from facebookresearch is [here](https://github.com/facebookresearch/llama).

I have experimented with Llama 2 on two of my workstations:
- a System76 Gazelle laptop with an Intel i7-11800H CPU and an Nvidia RTX 3050 Ti GPU
- a Mifcom desktop with an AMD Ryzen 9 5950X 16-Core Processor CPU and a negligible GPU

## Create a separate environment

I use [conda](https://docs.conda.io/en/latest/) to manage my Python environments. Personally, I find a Miniconda
installation to be more than enough.

```shell
conda create -n llama
conda activate llama
```

## CUDA Toolkit

If you're going to use an Nvidia GPU to run your models, you need to install the appropriate driver.
Installation guides for both Windows and Linux are available at [nvidia.com](https://www.nvidia.com/download/index.aspx).
Strictly speaking, you don't need a GPU to run Llama, but it can speed up the inference considerably.  
Once you got it up and running, you can check your version with `nvidia-smi`:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 536.67                 Driver Version: 536.67       CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   55C    P8               6W /  60W |    792MiB /  4096MiB |      3%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```
Notice that my CUDA Version is 12.2. Yours might be different.  
The next step is to install the specific CUDA Toolkit version that matches your driver:
```shell
conda install cuda --channel nvidia/label/cuda-12.2.0
```

Go to the [PyTorch website](https://pytorch.org/get-started/locally/) and select your OS and CUDA version. In my case,
I went with the following:
[PyTorch Configuration](images/pytorch.png)

For making sure that everything is working, you can run the following:
```shell
nvcc --version
```
Which gave me this:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:42:34_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```
And then:
```shell
python -c "import torch; print(torch.__version__)"
```
Which gave me this:
```
2.1.0.dev20230805+cu121
```

## Pure Python

The most straight forward way to run Llama 2 is directly from Python.
This is also the slowest to run on layman hardware. First, make sure you have the necessary dependencies:
```shell
pip install -r requirements.txt
```

Before starting, move the desired Llama 2 model folder to the top-level directory of this repo. I picked
`llama-2-7b-chat`, and even for that one, my laptop took a while to load the model.

```shell
D:\llama>ls llama-2-7b-chat
checklist.chk  consolidated.00.pth  params.json
```

There are two important parameters of LLMs:
1. Temperature  
In short, the lower the temperature, the more deterministic the results in the sense that the highest probable
next token is always picked. Increasing temperature could lead to more randomness, which encourages more diverse
or creative outputs. You are essentially increasing the weights of the other possible tokens.
In terms of application, you might want to use a lower temperature value for tasks like fact-based QA to encourage more
factual and concise responses. For poem generation or other creative tasks, it might be beneficial to increase the
temperature value.
2. Top_p  
You can control how deterministic the model is at generating a response.
If you are looking for exact and factual answers keep this low. If you are looking for more diverse responses,
increase to a higher value. 

### GPU (Windows)

Start the chat script:
```
torchrun --nproc_per_node 1 chat.py --ckpt_dir llama-2-7b-chat --tokenizer_path tokenizer.model
```
The model loaded in 75.11 seconds.
```
Enter prompt (or type exit): Generate a chocolate chip cookies recipe
Answer:   Sure, here's a classic chocolate chip cookie recipe that yields delicious and chewy cookies:

Ingredients:

* 2 1/4 cups all-purpose flour
* 1 teaspoon baking soda
* 1 teaspoon salt
* 1 cup unsalted butter, at room temperature
* 3/4 cup white granulated sugar
* 3/4 cup brown sugar
* 2 large eggs
* 2 teaspoons vanilla extract
* 2 ounces (1/4 cup) semisweet chocolate chips

Instructions:

1. Preheat the oven to 375°F (190°C). Line a baking sheet with parchment paper.
2. In a medium bowl, whisk together the flour, baking soda, and salt.
3. In a large bowl, use an electric mixer to beat the butter and sugars until light and fluffy, about 2 minutes. Beat in the eggs one at a time, followed by the vanilla extract.
4. Gradually mix in the dry ingredients until just combined, being careful not to overmix.
5. Stir in the chocolate chips.
6. Drop rounded tablespoonfuls of the dough onto the prepared baking sheet, about 2 inches apart.
7. Bake for 10-12 minutes, or until the edges are lightly golden brown.
8. Remove the cookies from the oven and let them cool on the baking sheet for 5 minutes, then transfer them to a wire rack to cool completely.

Enjoy your delicious homemade chocolate chip cookies!
```

This took approximately 10 minutes (621 seconds). On my laptop, that would be quite a slow chat experience.

## llama.cpp

llama.cpp is a C++ project that allows you to run Llama 2 (and other models) using quantization.
The quantization I'm going to use in my examples is to int8.

**What is quantization?**
Quoting from [hugginface](https://huggingface.co/docs/optimum/concept_guides/quantization):

"Quantization is a technique to reduce the computational and memory costs of running inference by representing
the weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual
32-bit floating point (float32). 
Reducing the number of bits means the resulting model requires less memory storage, consumes less energy (in theory),
and operations like matrix multiplication can be performed much faster with integer arithmetic.
It also allows to run models on embedded devices, which sometimes only support integer data types."

In my own words, say that the model weights will go from 32-bit floating point numbers to 8-bit integers.
This is a lossy compression, but it can speed up the inference. How can 32-bit floats be represented as 8-bit integers,
while not rendering the model useless?
If you've ever worked with numerical features, you have probably encountered the concept of scaling. For example,
if you have a feature that ranges from 0 to 100, you can scale it to range from 0 to 1. Simply put, the same concept can be
applied to the weights and activations of the neural network. Check out how the
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) works if
you've never used it before, it's quite simple. This, combined with clipping, can be used to map the 32-bit floats to
8-bit integers.

There are other ways to quantize a model, but I'm not going to go into that here. You don't really need to understand
the details of quantization to use llama.cpp.

### CPU (Linux)

I first set up [llama.cpp](https://github.com/ggerganov/llama.cpp):
```shell
git clone git@github.com:ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
pip install -r requirements.txt
```

If you have multiple C++ compilers, or cmake can't find the right one, you can specify it like this (I'm using clang++14):
```shell
```shell
cmake -DCMAKE_CXX_COMPILER=clang++-14 ..
````

Then copy the llama 2 model weights into the `models` directory inside the llama.cpp repo. Run the following commands
from the top-level directory of llama.cpp and llama:
```shell
cp -r llama/llama-2-7b-chat llama.cpp/models/7B
cp -r llama/tokenizer_checklist.chk llama.cpp/models/
cp -r llama/tokenizer.model llama.cpp/models/
```

Convert the 7B model to ggml FP16 format:
```shell
python convert.py models/7B/
```

Quantize the model to 8 bits:
```shell
./build/bin/quantize ./models/7B/ggml-model-f32.bin ./models/7B/ggml-model-q8_0.bin q8_0
```

Run a sample inference to check that everything is working:
```shell
./build/bin/main -m ./models/7B/ggml-model-q8_0.bin -n 128
```

Try it out in chat mode:
```shell
./build/bin/main -m ./models/7B/ggml-model-q8_0.bin -n -1 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

I gave it the same prompt as before, but this time our whole interaction took just a little longer than 1 minute (72 seconds).
```shell
User:Generate a chocolate chip cookies recipe
Bob: Of course! Here is a simple recipe for chocolate chip cookies that yields delicious results:
Ingredients:
* 2 1/4 cups all-purpose flour
* 1 teaspoon baking soda
* 1 teaspoon salt
* 1 cup unsalted butter, softened
* 3/4 cup white granulated sugar
* 1 cup brown sugar
* 2 large eggs
* 2 teaspoons vanilla extract
* 2 ounces semisweet chocolate chips

Instructions:
1. Preheat your oven to 375 degrees Fahrenheit (190 degrees Celsius). Line a baking sheet with parchment paper.
2. In a medium-sized bowl, whisk together the flour, baking soda, and salt. Set aside.
3. In a large bowl, use an electric mixer to cream together the butter and sugars until light and fluffy.
4. Beat in the eggs and vanilla extract until well combined.
5. Gradually mix in the flour mixture until a dough forms.
6. Stir in the chocolate chips.
7. Drop rounded spoonfuls of the dough onto the prepared baking sheet, leaving about 2 inches of space between each cookie.
8. Bake for 10-12 minutes, or until the edges are lightly golden brown.
9. Remove the cookies from the oven and let them cool on the baking sheet for 5 minutes before transferring them to a wire rack to cool completely.

I hope you enjoy these chocolate chip cookies! Let me know if you have any questions or need any further assistance.
User:

llama_print_timings:        load time =   379.63 ms
llama_print_timings:      sample time =    53.33 ms /   400 runs   (    0.13 ms per token,  7500.33 tokens per second)
llama_print_timings: prompt eval time =  1912.72 ms /   110 tokens (   17.39 ms per token,    57.51 tokens per second)
llama_print_timings:        eval time = 66839.05 ms /   399 runs   (  167.52 ms per token,     5.97 tokens per second)
llama_print_timings:       total time = 72030.84 ms
```

The default configuration uses 16 threads. My computer has 32 cores, so I'm using half of them. You can change this by
tweaking the `-t` parameter. See `./build/bin/main -h` for more details.
[htop](images/htop.png)

Hence, the LLM is now usable on my computer. If the 8-bit model is too slow for you, try the 4-bit quantization.  
If you're on Windows, you can get llama.cpp binaries from their [releases page](https://github.com/ggerganov/llama.cpp/releases).

## Further experimentaion

See [llama-recipes](https://github.com/facebookresearch/llama-recipes).

## References

1. [promptingguide.ai](https://www.promptingguide.ai/)
2. [barrelsofdata.com](https://barrelsofdata.com/llama-ai-language-model-on-laptop)