import os

from jinja2.ext import debug

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import shutil
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import gradio as gr
import cv2
import random
import shutil

logged_in_users = set()

device = torch.device("cuda")

# use bfloat16 for the entire notebook
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = None
inference_state = None
video_segments = {}

temp_dir = "/bask/projects/j/jiaoj-3d-vision/360XProject/segment-anything-2/temp"

shutil.rmtree(temp_dir, ignore_errors=True)
os.makedirs(temp_dir, exist_ok=True)

current_temp_dir = None

input_root_dir = '/bask/projects/j/jiaoj-3d-vision/360XProject/segment-anything-2-input'
output_root_dir = "/bask/projects/j/jiaoj-3d-vision/360XProject/segment-anything-2-output"

working_dir = '/bask/projects/j/jiaoj-3d-vision/360XProject/segment-anything-2-input/working'


def login(request: gr.Request):
    global logged_in_users
    username = request.session_hash
    if logged_in_users:
        gr.Info(f"{logged_in_users} is using the tool, please wait for a while", visible=True)
        gr.Error(f"{logged_in_users} is using the tool, please wait for a while", visible=True)
        return gr.update(visible=False), []

    logged_in_users.add(username)
    gr.Info("Login successfully")

    global predictor, inference_state, video_segments
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    predictor.cuda()
    inference_state = None
    video_segments = {}

    return gr.update(visible=True), list_all_files()


def logout(request: gr.Request):
    global logged_in_users
    logged_in_users.remove(request.session_hash)
    return gr.Info("Logout successfully")


def force_offload():
    global logged_in_users
    logged_in_users = set()
    return gr.Info("Force offload successfully")


def show_mask_pil(mask, obj_id=None, random_color=False):
    print('show_mask_pil')
    # 如果使用随机颜色
    if random_color:
        color = np.concatenate([np.random.random(3) * 255, np.array([153])], axis=0).astype(np.uint8)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6]) * 255
        color = color.astype(np.uint8)

    # 获取蒙版的高度和宽度
    h, w = mask.shape[-2:]

    # 创建颜色蒙版图像
    color_image = np.zeros((h, w, 4), dtype=np.uint8)
    color_image[..., :3] = color[:3]  # 设置颜色
    color_image[..., 3] = (mask * color[3]).astype(np.uint8)  # 设置透明度

    # 转换为PIL图像
    pil_mask_image = Image.fromarray(color_image, mode='RGBA')

    return pil_mask_image


def set_start_end_min(total_frames):
    print('set_start_end_min')
    return gr.update(visible=True, maximum=total_frames // 25 // 60), gr.update(visible=True,
                                                                                maximum=total_frames // 25 // 60)


def set_start_end_sec(total_frames, minuntes):
    print('set_start_end_sec')
    max_sec = min(59, total_frames // 25 - minuntes * 60)
    return gr.update(visible=True, maximum=max_sec)


def set_first_frame(start_min, start_sec):
    print('set_first_frame')
    first_frame = start_min * 25 + start_sec
    return gr.update()


def set_clip_time(start_min, start_sec, end_min, end_sec):
    print('set_clip_time')
    if not (isinstance(start_min, int) and isinstance(start_sec, int) and isinstance(end_min, int) and isinstance(
            end_sec, int)):
        gr.Error("Please select the start and end time")
    start_frame = (start_min * 60 + start_sec) * 25
    end_frame = (end_min * 60 + end_sec) * 25

    if start_frame >= end_frame:
        gr.Error("Start time should be less than end time")


def create_box(mask):
    print('create_box')
    # 提取非零元素的坐标
    y, x = np.where(mask[0])

    # 计算矩形框的坐标
    x0, y0 = x.min(), y.min()
    x1, y1 = x.max(), y.max()

    return [x0, y0, x1, y1]


def show_box_pil(box, image, color='green', width=2):
    print('show_box_pil')
    # 创建一个可绘制的Draw对象
    draw = ImageDraw.Draw(image)

    # 提取坐标
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]

    # 绘制矩形框
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    return image


def scan_all_images(video_dir):
    print('scan_all_images')
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def choose_video_source(current_video_choice, video_dir):
    print('choose_video_source')
    video_dir = os.path.join(input_root_dir, current_video_choice)

    frame_names = scan_all_images(video_dir)

    global current_temp_dir
    current_temp_dir = os.path.join(temp_dir, str(random.randint(0, 1000000)))
    os.makedirs(current_temp_dir, exist_ok=True)

    output_dir = os.path.join(output_root_dir, f'{os.path.basename(video_dir)}-on_process')

    start_time_min, end_time_min = set_start_end_min(len(frame_names))
    return (video_dir,
            frame_names,
            gr.update(value=len(frame_names)),
            start_time_min, end_time_min,
            gr.update(value=Image.open(os.path.join(video_dir, frame_names[0]))),
            output_dir)


def confirm_frames(current_video_choice, start_time_min, start_time_sec, end_time_min, end_time_sec):
    print('confirm_frames')
    video_dir = os.path.join(input_root_dir, current_video_choice)

    # frame_names = scan_all_images(video_dir)

    start_frame = int((start_time_min * 60 + start_time_sec) * 25)
    end_frame = int((end_time_min * 60 + end_time_sec) * 25)

    frame_names = [f'{start_frame + i:05d}.jpg' for i in range(end_frame - start_frame + 1)]
    global working_dir
    shutil.rmtree(working_dir, ignore_errors=True)
    os.makedirs(working_dir, exist_ok=True)
    for index in range(len(frame_names)):
        shutil.copy(os.path.join(video_dir, frame_names[index]), os.path.join(working_dir, frame_names[index]))

    global inference_state
    inference_state = predictor.init_state(video_path=working_dir)
    predictor.reset_state(inference_state)

    output_dir = os.path.join(output_root_dir, f'{os.path.basename(video_dir)}')

    os.makedirs(output_dir, exist_ok=True)
    return (video_dir,
            frame_names,
            gr.update(maximum=len(frame_names), visible=True),
            gr.update(value=Image.open(os.path.join(video_dir, frame_names[0]))),
            output_dir)


def change_frame(frame_index, video_dir, frame_names):
    print('change_frame')
    return gr.update(value=Image.open(os.path.join(video_dir, frame_names[frame_index - 1]))), [], []


def clear_all_points(video_dir, frame_index, frame_names):
    print('clear_all_points')
    return gr.update(value=Image.open(os.path.join(video_dir, frame_names[frame_index - 1]))), [], []


def click_image(image, evt: gr.SelectData, label_radio, prompt_dot, prompt_label):
    print('click_image')
    # 获取点击的x和y坐标
    x, y = int(evt.index[0]), int(evt.index[1])

    prompt_dot.append((x, y))
    prompt_label.append(1 if label_radio == "Positive" else 0)

    img_with_dot = image.copy()

    draw = ImageDraw.Draw(img_with_dot)

    for (x, y), label in zip(prompt_dot, prompt_label):
        color = (0, 255, 0) if label == 0 else (255, 0, 0)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=None)

    return img_with_dot, prompt_dot, prompt_label


def generate_mask(prompt_dot, prompt_label, frame_index, video_dir, frame_names):
    print('generate_mask')
    predictor.cuda()
    if not prompt_dot:
        gr.Error("Please click on the image to label the object")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_index - 1,
        obj_id=1,
        points=prompt_dot,
        labels=prompt_label,
    )

    image = Image.open(os.path.join(video_dir, frame_names[frame_index - 1]))
    mask = show_mask_pil((out_mask_logits[0] > 0.0).cpu().numpy(), obj_id=out_obj_ids[0])
    image.paste(mask, (0, 0), mask)

    try:
        box = create_box((out_mask_logits[0] > 0.0).cpu())

        image = show_box_pil(box, image)
    except:
        pass


    return image, gr.update(interactive=True)


def propagate_mask(video_dir, frame_names, output_index):
    print('propagate_mask')
    global video_segments
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    image = Image.open(os.path.join(video_dir, frame_names[output_index - 1]))
    for out_obj_id, out_mask in video_segments[output_index - 1].items():
        mask = show_mask_pil(out_mask, obj_id=out_obj_id)
        image.paste(mask, (0, 0), mask)

        try:
            box = create_box(torch.tensor(out_mask))

            image = show_box_pil(box, image)
        except:
            pass


    return image, gr.update(maximum=len(frame_names), visible=True, interactive=True)


def change_frame_for_output(output_index, video_dir, frame_names):
    print('change_frame_for_output')
    image = Image.open(os.path.join(video_dir, frame_names[output_index - 1]))
    for out_obj_id, out_mask in video_segments[output_index - 1].items():
        mask = show_mask_pil(out_mask, obj_id=out_obj_id)
        image.paste(mask, (0, 0), mask)

        try:
            box = create_box(torch.tensor(out_mask))

            image = show_box_pil(box, image)
        except:
            pass

    return image


def composite_video_fn(frame_names, video_dir):
    print('composite_video_fn')
    global current_temp_dir
    global video_segments

    current_temp_dir = os.path.join(temp_dir, str(random.randint(0, 1000000)))
    os.makedirs(current_temp_dir, exist_ok=True)
    for frame_idx, frame_name in enumerate(frame_names):
        image = Image.open(os.path.join(video_dir, frame_name))
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            mask = show_mask_pil(out_mask, obj_id=out_obj_id)
            image.paste(mask, (0, 0), mask)

            try:
                box = create_box(torch.tensor(out_mask))

                image = show_box_pil(box, image)
            except:
                pass

        image.save(os.path.join(current_temp_dir, frame_name))

    video_path = os.path.join(current_temp_dir, os.path.basename(video_dir) + ".mp4")
    os.system(f"ffmpeg -y -r 25 -i {current_temp_dir}/%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p {video_path}")

    return video_path


def save_masks(output_dir, postfix, start_time_min, start_time_sec, end_time_min, end_time_sec):
    print('save_masks')
    suboutput_dir = os.path.join(output_dir, f'{start_time_min}:{start_time_sec}-{end_time_min}:{end_time_sec}-{postfix}')
    os.makedirs(suboutput_dir, exist_ok=True)
    import pickle
    with open(os.path.join(suboutput_dir, "masks.pkl"), "wb") as f:
        pickle.dump(video_segments, f)

    gr.Info("Masks saved to " + os.path.join(suboutput_dir, "masks.pkl"))


def list_all_files():
    print('list_all_files')
    global video_choices
    video_choices = os.listdir(input_root_dir)

    saved_choice = os.listdir(output_root_dir)

    # video_choices = [choice for choice in video_choices if choice not in saved_choice]

    video_choices = [choice for choice in video_choices if os.path.isdir(os.path.join(input_root_dir, choice))]

    video_choices.sort()

    return gr.update(choices=video_choices)


start_instructions = """
<h1>Internal Labeling Tool for action labelling by Segment Anything 2</h1>
<p>Version 05/09/2024</p>
<p>Instructions:</p>
<ol>
    <li>Choose the video source</li>
    <li>Click on the image to label the object</li>
    <li>Click on "Generate mask" to generate the mask</li>
    <li>Click on "Propagate mask" to propagate the mask</li>
    <li>(Optional) Click on "Composite video" to composite the video (Just for preview)</li>
    <li>Click on "Save masks" to save the masks</li>
</ol>
<p>Notes:</p>
<p> If using all frames, the memory usage will be more than 100GB. So all clip should be less than 1500 frames maximum.</p>
"""
with gr.Blocks() as demo:
    list_all_files()

    instruction = gr.HTML(start_instructions)

    force_offload_button = gr.Button("Force offload", visible=True)

    prompt_dot = gr.Json(value=[], visible=False)
    prompt_label = gr.Json(value=[], visible=False)

    video_dir = gr.Text(value='',visible=False, label="Video directory")
    frame_names = gr.Json(value=[], visible=False)

    output_dir = gr.Text(visible=False, label="Output directory")

    logged = gr.Json(value=False, visible=False)

    with gr.Row(visible=False) as row:
        with gr.Column():
            with gr.Row():
                video_choice = gr.Dropdown(
                    [],
                    label="Video source",
                )
                refresh_button = gr.Button("Refresh")

            with gr.Group():
                total_seconds = gr.Json(value=0, visible=False)
                with gr.Row():
                    start_time_min = gr.Number(0, 0, 0, step=1, minimum=0, interactive=True, label="Start time (min)")
                    start_time_sec = gr.Number(0, 0, 0, minimum=0, maximum=59.9, interactive=True,
                                               label="Start time (sec)")
                with gr.Row():
                    end_time_min = gr.Number(0, 0, 0, step=1, minimum=0, interactive=True, label="End time (min)")
                    end_time_sec = gr.Number(0, 0, 0, minimum=0, maximum=59.9, interactive=True,
                                             label="End time (sec)")

                set_clip_button = gr.Button("Set clip time", interactive=True, visible=True)
            frame_index = gr.Slider(1, 0, 1, step=1, interactive=True, visible=False, label="Frame index")
            with gr.Group():
                input_img = gr.Image(type="pil", interactive=False, label="Original Frame")
                with gr.Row():
                    label_radio = gr.Radio(["Positive", "Negative"], value="Positive", label="Label")
                    clear_button = gr.Button("Clear all points")

            generate_mask_button = gr.Button("Generate mask")
        with gr.Column():
            propagate_button = gr.Button("Propagate mask", interactive=False)
            output_index = gr.Slider(1, 2, 1, step=1, interactive=False, visible=False, label="Output index")
            output_img = gr.Image(interactive=False, label="Masked Frame")

            composite_video = gr.Button("Composite video")

            final_video = gr.Video(label="Masked video", interactive=False)

            postfix = gr.Textbox(value="", label="Postfix", visible=True)

            save_button = gr.Button("Save masks")

            demo.load(login, None, [row, video_choice])
            demo.unload(logout)

    with gr.Row():
        gr.FileExplorer(glob=f"**/*",root_dir=output_root_dir, label="Output directory", visible=True, interactive=False)

    force_offload_button.click(force_offload)

    refresh_button.click(list_all_files, outputs=[video_choice])
    # video_choice.change(choose_video_source, inputs=[video_choice, video_dir],
    #                     outputs=[video_dir, frame_names, total_seconds, start_time_min, end_time_min, input_img,
    #                              output_dir])


    set_clip_button.click(confirm_frames, inputs=[video_choice, start_time_min, start_time_sec, end_time_min, end_time_sec], outputs=[video_dir, frame_names, frame_index,input_img,  output_dir])
    frame_index.change(change_frame, inputs=[frame_index, video_dir, frame_names],
                       outputs=[input_img, prompt_dot, prompt_label])
    clear_button.click(clear_all_points, inputs=[video_dir, frame_index, frame_names],
                       outputs=[input_img, prompt_dot, prompt_label])
    input_img.select(fn=click_image, inputs=[input_img, label_radio, prompt_dot, prompt_label],
                     outputs=[input_img, prompt_dot, prompt_label])

    generate_mask_button.click(generate_mask, inputs=[prompt_dot, prompt_label, frame_index, video_dir, frame_names],
                               outputs=[input_img, propagate_button])

    propagate_button.click(propagate_mask, inputs=[video_dir, frame_names, output_index],
                           outputs=[output_img, output_index])

    output_index.change(change_frame_for_output, inputs=[output_index, video_dir, frame_names], outputs=[output_img])

    composite_video.click(composite_video_fn, inputs=[frame_names, video_dir], outputs=[final_video])

    save_button.click(save_masks, inputs=[output_dir, postfix, start_time_min, start_time_sec, end_time_min, end_time_sec], outputs=[instruction])

demo.launch(server_name="0.0.0.0", server_port=7861, show_error=True, debug=True, share=True
            # auth=[("chenyuan", "chenyuan"), ("yuqi", "yuqi"), ("qiming", "qiming"), ('chenhao', 'chenhao'), ('xiaohan', 'xiaohan'), ('jianbo', 'jianbo')])
            )
