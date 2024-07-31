import argparse
import json
from PIL import Image
from typing import List
from io import BytesIO
import base64

import requests

from llava.conversation import conv_templates

def read_image(image_path: str):

    # 使用Image.open()方法读取图片
    with Image.open(image_path) as img:
        # 在这里可以对img进行各种操作，如显示、转换、保存等
        print("Image size:", img.size)
        print("Image mode:", img.mode)

        # 显示图片
        return img

def resize_image(image: Image):

    max_hw, min_hw = max(image.size), min(image.size)
    aspect_ratio = max_hw / min_hw
    max_len, min_len = 800, 400
    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = image.size
    if longest_edge != max(image.size):
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))

    return image


def get_images(image_path_list: List[str], return_pil=False):

    all_images = [read_image(image_path) for image_path in image_path_list]

    images = []

    for image in all_images:

        if return_pil:
            images.append(image)
        else:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_b64_str)

    return images


def main():

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], args.message)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "LLaVA Client"}
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "stop": conv.sep2,
    }

    pload['images'] = get_images(['images/chest_x_ray_coronal.png'])

    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)

    print(prompt, end="")
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split("[/INST]")[-1]
            print(output, end="\r")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--message", type=str, default=
        "Tell me a story with more than 1000 words.")
    args = parser.parse_args()

    main()