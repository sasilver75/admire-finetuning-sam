# Model Descriptions

TODO: ADD LANGUAGE SUPPORT INFORMATION FOR EACH MODEL!

- `openbmb/MiniCPM-V-2_6`
    - [HuffingFace](https://huggingface.co/openbmb/MiniCPM-V-2_6)
    - The latest and most capable model in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B with a total of 8B parameters. Seems to surpass widely-used proprietary models like GPT-4o mini, GPT-4V, Claude 3.5 Sonnet for single-image understanding... and can also perform conversation/reasoning over multiple images. 
- `Qwen/Qwen2-VL-7B-Instruct`
    - [HuggingFace](https://huggingface.co/Qwen/
    Qwen2-VL-7B-Instruct)
    - Unlike earlier models, Qwen2-VL can handle arbitrary image resolutions, for a more human-like visual processing experience.
    - Built on DFN ViT image encoder, a CLIP model. 
    - Can handle multiple images in the input prompt.
    - There's also a 2B version of this model on huggingface at [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- `allenai/Molmo-7B-0-0924`
    - [HuggingFace](https://huggingface.co/allenai/Molmo-7B-O-0924)
    - Uses OpenAI CLIP as the vision backbone
    - Uses OLMo-7B as the language model
    - "This checkpoint is a preview of the Molmo release. All artifacts used in creating Molmo (PixMo dataset, training code, evaluations, intermediate checkpoints) will be made available at a later date, furthering our commitment to open-source AI development and reproducibility."
    - Can handle multiple image inputs
- `allenai/Molmo-7B-D-0924`
    - [HuggingFace](https://huggingface.co/allenai/Molmo-7B-D-0924)
    - Uses OpenAI CLIP as the vision backbone
    - Uses Qwen2-7B as the langauge model
    - Slightly better than Molmo-7B-O-0924 on an average of 11 academic benchmarks (Indeed, there's not a single of these 11 benchmarks for which the OLMo-based model performed better).
    - "This checkpoint is a preview of the Molmo release. All artifacts used in creating Molmo (PixMo dataset, training code, evaluations, intermediate checkpoints) will be made available at a later date, furthering our commitment to open-source AI development and reproducibility."
    - Can handle multipleimage inputs
- `vikhyatk/moondream1`
    - [HuggingFace](https://huggingface.co/vikhyatk/moondream1)
    - 1.6B parameter model built by @vikhyatk using SigLIP, Phi-1.5 and the LLaVa training dataset.
        - HF says 1.86B though?
    - Seems to perform similarly to some LLaVA models, despite being very small?
- `vikhyatk/moondream2`
    - [HuggingFace](https://huggingface.co/vikhyatk/moondream2)
    - Assumedly uses the same arch as Moondream1? Seems to be the same size too, maybe just a different checkpoint? Improved the VQAv2 score from 74.7 to 80.3 with latest release (2024-08-26).
- `vikhyatk/moondream-next`
    - [HuggingFace](https://huggingface.co/vikhyatk/moondream-next)
    - Submitted an access request to see it.
- `liuhaotian/llava-v1.5-7b`
    - [HuggingFace](https://huggingface.co/liuhaotian/llava-v1.5-7b)
    - I'm unsure if this 1.5 is the original LLaVA, it seems like it is. This is from the author of the paper [here](https://arxiv.org/abs/2304.08485).
        - It seems like it's not, see [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) 
    - Uses CLIP and Vicuna, and the then finetunes "e2e" on generated instructional vision-language data.
- `google/paligemma-3b-pt-224`
    - [HuggingFace](https://huggingface.co/google/paligemma-3b-pt-224)
- NVLM? Too big?
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
    - [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- `smolVLM`
    - [Instruct HF](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
    - [Base HF](https://huggingface.co/HuggingFaceTB/SmolVLM-Base)
    - Blogpost: https://huggingface.co/blog/smolvlm


SMOL Vision Language Models and their min GPU RAM requirements:
- SmolVLM - 5.02GB
- Qwen2-VL 2B - 13.70GB
- InternVL2 2B - 10.52GB
- PaliGemma 3B 448px - 6.72GB
- moondream2 - 3.87GB
- MiniCPM-V-2 - 7.88GB
- MM1.5 1B - NaN

Model	MMMU (val)	MathVista (testmini)	MMStar (val)	DocVQA (test)	TextVQA (val)	Min GPU RAM required (GB)
SmolVLM	38.8	44.6	42.1	81.6	72.7	5.02
Qwen2-VL 2B	41.1	47.8	47.5	90.1	79.7	13.70
InternVL2 2B	34.3	46.3	49.8	86.9	73.4	10.52
PaliGemma 3B 448px	34.9	28.7	48.3	32.2	56.0	6.72
moondream2	32.4	24.3	40.3	70.5	65.2	3.87
MiniCPM-V-2	38.2	39.8	39.1	71.9	74.1	7.88
MM1.5 1B	35.8	37.2	0.0	81.0	72.5	NaN