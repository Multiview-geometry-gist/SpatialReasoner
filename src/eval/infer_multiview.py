"""Multi-view inference for MVGen-Multi model evaluation.

This script evaluates MVGen-Multi model with two-view input (original + rotated),
matching the training format.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python src/eval/infer_multiview.py \
        --model_path /data/SpatialReasoner/checkpoints/Qwen2.5-VL-7B-SFT-MVGen-Multi \
        --data_path /data/SpatialReasoner/results/multiview_eval_600/sampled_data.tsv \
        --view_mapping /data/SpatialReasoner/results/multiview_eval_600/generated_views/view_mapping.json \
        --original_images /data/SpatialReasoner/results/multiview_eval_600/original_images \
        --output_path /home/ubuntu/SpatialReasoner/results/mvgen_multi_eval/multiview_inference.xlsx \
        --gpu_ids 0,1,2,3 \
        --view_angle 10.0
"""

import os
import json
import base64
import argparse
import string
from io import BytesIO

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.multiprocessing import Process, set_start_method, Manager

# System prompt matching training
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def get_args():
    parser = argparse.ArgumentParser(description="Multi-view inference for MVGen-Multi model")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str, help="Path to sampled_data.tsv")
    parser.add_argument("--view_mapping", required=True, type=str, help="Path to view_mapping.json")
    parser.add_argument("--original_images", required=True, type=str, help="Path to original images directory")
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--gpu_ids", default="0,1,2,3", type=str)
    parser.add_argument("--batch_size", default=1, type=int)  # Multi-image needs smaller batch
    parser.add_argument("--view_angle", default=10.0, type=float, help="Rotation angle for secondary view")
    return parser.parse_args()


def load_image_from_base64(image_string):
    """Decode base64 image string to PIL Image."""
    image_data = base64.b64decode(image_string)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image


def load_image_from_path(path):
    """Load PIL Image from file path."""
    return Image.open(path).convert('RGB')


def init_model(model_path, gpu_id):
    """Initialize model on specific GPU."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=f"cuda:{gpu_id}",
    )
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
    return model, processor


def format_multiview_conversation(question, options, primary_angle=0.0, secondary_angle=10.0):
    """Format conversation with two images matching training format.

    IMPORTANT: This must match the 'sequential' format used in sft_multiview.py training:
        View 1 (rotation: {primary_angle} degrees):
        View 2 (rotation: {secondary_angle} degrees):

        Question: ...
    """
    options_text = "\n".join([f"{opt}. {options[opt]}" for opt in ['A', 'B', 'C', 'D'] if opt in options])
    question_text = f"Question: {question}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."

    # Multi-view format: MUST match training's 'sequential' format exactly
    # Training format (from sft_multiview.py ConversationFormatter.format_multi_view):
    #   "View 1 (rotation: {primary_angle:.0f} degrees):\n"
    #   "View 2 (rotation: {secondary_angle:.0f} degrees):\n\n"
    #   "{question_text}"
    prompt_text = (
        f"View 1 (rotation: {primary_angle:.0f} degrees):\n"
        f"View 2 (rotation: {secondary_angle:.0f} degrees):\n\n"
        f"{question_text}"
    )

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Primary view (original)
                {"type": "image"},  # Secondary view (rotated)
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    return conversation


def infer_batch(batch_data, model, processor, device_id):
    """Run inference on a batch of multi-view samples."""
    results = []

    for item in batch_data:
        conversation = item['conversation']
        primary_image = item['primary_image']
        secondary_image = item['secondary_image']

        # Apply chat template
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Process with two images
        inputs = processor(
            text=[text],
            images=[primary_image, secondary_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{device_id}")

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                top_p=0.001,
                top_k=1,
                temperature=0.01,
                use_cache=True,
            )

        # Decode
        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        results.append(output_text)

    return results


def infer_on_gpu(model_path, device_id, chunk_data, view_mapping, original_images_dir, view_angle, results_dict):
    """Run inference on a single GPU."""
    model, processor = init_model(model_path, device_id)

    responses = []
    for _, row in tqdm(chunk_data.iterrows(), total=len(chunk_data),
                       desc=f"GPU {device_id}", position=device_id, leave=False):
        try:
            # Get image hash
            image_hash = row.get('image_hash', '')

            # Load original image
            if image_hash and os.path.exists(os.path.join(original_images_dir, f"{image_hash}.jpg")):
                primary_image = load_image_from_path(os.path.join(original_images_dir, f"{image_hash}.jpg"))
            elif 'image' in row and pd.notna(row['image']):
                primary_image = load_image_from_base64(row['image'])
            else:
                responses.append("ERROR: Could not load primary image")
                continue

            # Load secondary (rotated) view
            angle_key = str(view_angle)
            if image_hash in view_mapping and angle_key in view_mapping[image_hash]:
                secondary_path = view_mapping[image_hash][angle_key]
                if os.path.exists(secondary_path):
                    secondary_image = load_image_from_path(secondary_path)
                else:
                    # Fallback: use primary image
                    secondary_image = primary_image
            else:
                # Fallback: use primary image
                secondary_image = primary_image

            # Format conversation
            options = {opt: row[opt] for opt in ['A', 'B', 'C', 'D'] if opt in row and pd.notna(row[opt])}
            conversation = format_multiview_conversation(
                row['question'], options,
                primary_angle=0.0, secondary_angle=view_angle
            )

            # Inference
            batch_item = {
                'conversation': conversation,
                'primary_image': primary_image,
                'secondary_image': secondary_image,
            }
            result = infer_batch([batch_item], model, processor, device_id)
            responses.append(result[0])

        except Exception as e:
            responses.append(f"ERROR: {str(e)}")

    results_dict[device_id] = responses


def main():
    args = get_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path, sep='\t')
    print(f"Total samples: {len(df)}")

    # Load view mapping
    print(f"Loading view mapping from {args.view_mapping}")
    with open(args.view_mapping, 'r') as f:
        view_mapping = json.load(f)
    print(f"View mapping entries: {len(view_mapping)}")

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using GPUs: {gpu_ids}")

    # Split data across GPUs
    set_start_method("spawn", force=True)
    manager = Manager()
    results_dict = manager.dict()

    chunk_size = len(df) // num_gpus
    processes = []

    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(df)
        chunk = df.iloc[start_idx:end_idx].copy()

        p = Process(
            target=infer_on_gpu,
            args=(args.model_path, gpu_id, chunk, view_mapping,
                  args.original_images, args.view_angle, results_dict)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    # Collect results
    all_responses = []
    for i, gpu_id in enumerate(gpu_ids):
        all_responses.extend(results_dict[gpu_id])

    # Save results
    df['prediction'] = all_responses
    df.to_excel(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

    # Quick accuracy check
    import re
    def extract_answer(pred):
        if pd.isna(pred) or 'ERROR' in str(pred):
            return None
        pred = str(pred)
        match = re.search(r'<answer>\s*([A-D])', pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r'answer\s+is\s+([A-D])', pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    df['pred_answer'] = df['prediction'].apply(extract_answer)
    df['correct'] = df['pred_answer'] == df['answer']
    accuracy = df['correct'].mean() * 100
    print(f"\nOverall accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
