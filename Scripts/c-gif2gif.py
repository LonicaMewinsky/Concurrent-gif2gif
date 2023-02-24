import copy
import os
import modules.scripts as scripts
import modules.images
import gradio as gr
import numpy as np
import tempfile
import random
import cv2
from PIL import Image, ImageSequence, ImageOps
from modules.processing import Processed, process_images
from modules.shared import state, sd_upscalers

def stabilize_images(images):
    # Convert PIL images to numpy arrays
    images = [np.array(img) for img in images]

    # Initialize the output list
    stabilized_images = []

    # Load the first image
    prev_image = images[0]

    # Loop through the rest of the images
    for curr_image in images[1:]:
        # Convert the images to grayscale
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY)

        # Calculate the phase correlation between the previous and current images
        shape = curr_gray.shape
        h, w = shape[0], shape[1]
        fft_prev = np.fft.fft2(prev_gray)
        fft_curr = np.fft.fft2(curr_gray)
        cc = np.real(np.fft.ifft2((fft_prev * fft_curr.conj()) / np.abs(fft_prev * fft_curr.conj())))
        cc_max = np.unravel_index(np.argmax(cc), cc.shape)
        offset = np.array([cc_max[0] - h if cc_max[0] > h // 2 else cc_max[0],
                           cc_max[1] - w if cc_max[1] > w // 2 else cc_max[1]])
        
        # Apply the phase shift to the current image to generate the stabilized image
        M = np.float32([[1, 0, offset[1]], [0, 1, offset[0]]])
        curr_stabilized = cv2.warpAffine(curr_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Convert the stabilized image back to a PIL image and append it to the output list
        stabilized_images.append(Image.fromarray(cv2.cvtColor(curr_stabilized, cv2.COLOR_BGR2RGB)))

        # Set the current image as the previous image for the next iteration
        prev_image = curr_image

    # Return the list of stabilized PIL images
    return stabilized_images

#Rudimentary interpolation
def interp(gif, iframes, dur):
    try:
        working_images, resframes = [], []
        pilgif = Image.open(gif)
        for frame in ImageSequence.Iterator(pilgif):
            converted = frame.convert('RGBA')
            working_images.append(converted)
        resframes.append(working_images[0]) #Seed the first frame
        alphas = np.linspace(0, 1, iframes+2)[1:]
        for i in range(1, len(working_images), 1):
            for a in range(len(alphas)):
                intermediate_image = Image.blend(working_images[i-1],working_images[i],alphas[a])
                resframes.append(intermediate_image)
        resframes[0].save(gif,
            save_all = True, append_images = resframes[1:], loop = 0,
            optimize = False, duration = dur, format='GIF')
        return gif
    except:
        return False

def split_frames(lst, max_length):

    num_sublists = -(-len(lst) // max_length)  # ceil division
    sublists = [lst[i*max_length:(i+1)*max_length] for i in range(num_sublists)]
    last_sublist = sublists[-1] if len(sublists[-1]) == max_length else sublists.pop()
    last_sublist.extend([last_sublist[-1]] * (max_length - len(last_sublist)))
    sublists.append(last_sublist)
    return sublists

def MakeGrid(images, rows, cols):
    widths, heights = zip(*(i.size for i in images))

    grid_width = max(widths) * cols
    grid_height = max(heights) * rows
    cell_width = grid_width // cols
    cell_height = grid_height // rows


    final_image = Image.new('RGB', (grid_width, grid_height))

    x_offset = 0
    y_offset = 0
    for i in range(len(images)):
        final_image.paste(images[i], (x_offset, y_offset))
        x_offset += cell_width
        if x_offset == grid_width:
            x_offset = 0
            y_offset += cell_height

    # Save the final image
    return final_image

def BreakGrid(grid, rows, cols):
    width = grid.width // cols
    height = grid.height // rows
    outimages = []
    for row in range(rows):
            for col in range(cols):
                left = col * width
                top = row * height
                right = left + width
                bottom = top + height

                current_img = grid.crop((left, top, right, bottom))

                #current_img = current_img.resize([300, 300], Image.Resampling.LANCZOS)
                outimages.append(current_img)
    return outimages

def upscale(image, upscaler_name, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop):
    if upscale_mode == 1:
        upscale_by = max(upscale_to_width/image.width, upscale_to_height/image.height)
    
    upscaler = next(iter([x for x in sd_upscalers if x.name == upscaler_name]), None)
    assert upscaler or (upscaler_name is None), f'could not find upscaler named {upscaler_name}'

    image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)
    if upscale_mode == 1 and upscale_crop:
        cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
        cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
        image = cropped

    return image

class Script(scripts.Script):
    def __init__(self):
        self.gif_name = str()
        self.orig_fps = 0
        self.orig_duration = 0
        self.orig_total_seconds = 0
        self.orig_n_frames = 0
        self.orig_dimensions = (0,0)
        self.ready = False
        self.desired_fps = 0
        self.desired_interp = 0
        self.desired_duration = 0
        self.desired_total_seconds = 0
        self.desired_rows = 0
        self.desired_cols = 0
        self.readygrids = []
        self.slowmo = False
        self.gif2gifdir = tempfile.TemporaryDirectory()
        self.img2img_component = gr.Image()
        self.img2img_inpaint_component = gr.Image()
        self.img2img_width_component = gr.Slider()
        self.img2img_height_component = gr.Slider()
        self.upscale_args = {}
        return None

    def title(self):
        return "Concurrent gif2gif"
    def show(self, is_img2img):
        return True
    
    def ui(self, is_img2img):
        #Controls
        with gr.Box():    
            with gr.Column():
                upload_gif = gr.File(label="Upload GIF", file_types = ['.gif','.webp','.plc'], live=True, file_count = "single")
                with gr.Row():
                    with gr.Column():
                        grid_row_slider = gr.Slider(minimum = 4, maximum = 20, step=2.0, label = "Rows")
                        grid_col_slider = gr.Slider(minimum = 4, maximum = 20, step=2.0, label = "Columns")
                        gif_clear_frames = gr.Checkbox(value = True, label="Delete intermediate frames after GIF generation")
                        gif_common_seed = gr.Checkbox(value = True, label="For -1 seed, all frames in a GIF have common seed")
                    with gr.Column():
                        frames_per_sheet = gr.Textbox(value="0", interactive = False, label = "Frames per sheet")
                        number_of_sheets = gr.Textbox(value="0", interactive = False, label = "Number of sheets")
                with gr.Accordion("Upscaling", open = False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Tabs():
                                with gr.Tab("Scale by") as tab_scale_by:
                                    with gr.Box():   
                                        ups_scale_by = gr.Slider(1, 8, step = 0.1, value=2, interactive = True, label = "Factor")
                                with gr.Tab("Scale to") as tab_scale_to:
                                    with gr.Box():
                                        with gr.Column():
                                            ups_scale_to_w = gr.Slider(0, 8000, step = 8, value=512, interactive = True, label = "Target width")
                                            ups_scale_to_h = gr.Slider(0, 8000, step = 8, value=512, interactive = True, label = "Target height")
                                            ups_scale_to_crop = gr.Checkbox(value = False, label = "Crop to fit")
                        with gr.Column():
                            with gr.Box():
                                ups_upscaler_1 = gr.Dropdown(value = "None", interactive = True, choices = [x.name for x in sd_upscalers], label = "Upscaler 1")
                grid_gen_button = gr.Button(value = "Generate sheets")
                sheet_gallery = gr.Gallery(label = "Sheets for generation")                
                with gr.Accordion("Animation tweaks", open = False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Box():
                                fps_slider = gr.Slider(1, 50, step = 1, label = "Desired FPS")
                                interp_slider = gr.Slider(label = "Interpolation frames", value = 0)
                        with gr.Column():   
                            with gr.Row():
                                with gr.Box():
                                    with gr.Column():
                                        fps_actual = gr.Textbox(value="", interactive = False, label = "Actual FPS")
                                        seconds_actual = gr.Textbox(value="", interactive = False, label = "Actual total duration")
                                        frames_actual = gr.Textbox(value="", interactive = False, label = "Actual total frames")
                                with gr.Box():
                                    with gr.Column():
                                        fps_original = gr.Textbox(value="", interactive = False, label = "Original FPS")
                                        seconds_original = gr.Textbox(value="", interactive = False, label = "Original total duration")
                                        frames_original = gr.Textbox(value="", interactive = False, label = "Original total frames")
                    display_gif = gr.Image(inputs = upload_gif, visible = False, label = "Preview GIF", type= "filepath")

        #Control functions
        def processgif(gif):
            try:
                init_gif = Image.open(gif.name)
                self.gif_name = gif.name
                #Need to also put images in img2img/inpainting windows (ui will not run without)
                #Gradio painting tools act weird with smaller images.. resize to 480 if smaller
                self.orig_dimensions = init_gif.size
                self.orig_duration = init_gif.info["duration"]
                self.orig_n_frames = init_gif.n_frames
                self.orig_total_seconds = round((self.orig_duration * self.orig_n_frames)/1000, 2)
                self.orig_fps = round(1000 / int(init_gif.info["duration"]), 2)
                self.ready = True
                return gif.name, gr.Image.update(visible = True), self.orig_fps, self.orig_fps, (f"{self.orig_total_seconds} seconds"), self.orig_n_frames
            except:
                print(f"Failed to load {gif.name}. Not a valid animated GIF?")
                return None
        
        def cleargif(up_val):
            if (up_val == None):
                self.gif_name = None
                self.ready = False
                return gr.Image.update(visible = False)        
        
        def fpsupdate(fps, interp_frames):
            if (self.ready and fps and (interp_frames != None)):
                self.desired_fps = fps
                self.desired_interp = interp_frames
                total_n_frames = self.orig_n_frames + ((self.orig_n_frames -1) * self.desired_interp)
                calcdur = (1000 / fps) / (total_n_frames/self.orig_n_frames)
                if calcdur < 20:
                    calcdur = 20
                    self.slowmo = True
                self.desired_duration = calcdur
                self.desired_total_seconds = round((self.desired_duration * total_n_frames)/1000, 2)
                gifbuffer = (f"{self.gif2gifdir.name}/previewgif.gif")
                previewgif = Image.open(self.gif_name)
                previewgif.save(gifbuffer, format="GIF", save_all=True, duration=self.desired_duration, loop=0)
                if interp:
                    interp(gifbuffer, self.desired_interp, self.desired_duration)
                return gifbuffer, round(1000/self.desired_duration, 2), f"{self.desired_total_seconds} seconds", total_n_frames

        def gridgif(gif, rows, cols):
            pilframes = []
            grids = []
            self.desired_rows = rows
            self.desired_cols = cols
            framesper = cols * rows
            init_gif = Image.open(gif.name)
            #Break gif
            for frame in ImageSequence.Iterator(init_gif):
                interm = frame.convert("RGB")
                #Images need to remain a factor a 4!
                interm = interm.resize([(interm.width - (interm.width % 4)), (interm.height - (interm.height % 4))], Image.Resampling.LANCZOS)
                pilframes.append(interm)
            #Make chunks
            pilchunks = split_frames(pilframes, framesper)
            #Make grids from the chunks
            for chunk in pilchunks:
                grid = MakeGrid(chunk, rows, cols)
                #grid = grid.resize([(grid.width - (grid.width % cols)), (grid.height - (grid.height % rows))], Image.Resampling.LANCZOS)
                grids.append(grid)
            self.readygrids = grids
            #Update vanilla UI
            img_for_ui_path = (f"{self.gif2gifdir.name}/imgforui.gif")
            img_for_ui = grids[0]
            if img_for_ui.height < 480:
                img_for_ui = img_for_ui.resize((round(480*img_for_ui.width/img_for_ui.height), 480), Image.Resampling.LANCZOS)
            img_for_ui.save(img_for_ui_path)

            return grids, img_for_ui, img_for_ui, framesper, len(grids)

        #Control change events
        ups_scale_mode = gr.State(value = 0)
        fps_slider.change(fn=fpsupdate, inputs = [fps_slider, interp_slider], outputs = [display_gif, fps_actual, seconds_actual, frames_actual])
        interp_slider.change(fn=fpsupdate, inputs = [fps_slider, interp_slider], outputs = [display_gif, fps_actual, seconds_actual, frames_actual])
        upload_gif.upload(fn=processgif, inputs = upload_gif, outputs = [display_gif, display_gif, fps_slider, fps_original, seconds_original, frames_original])
        upload_gif.change(fn=cleargif, inputs = upload_gif, outputs = display_gif)
        grid_gen_button.click(fn=gridgif, inputs = [upload_gif, grid_row_slider, grid_col_slider], outputs = [sheet_gallery, self.img2img_component, self.img2img_inpaint_component, frames_per_sheet, number_of_sheets])
        tab_scale_by.select(fn=lambda: 0, inputs=[], outputs=[ups_scale_mode])
        tab_scale_to.select(fn=lambda: 1, inputs=[], outputs=[ups_scale_mode])

        return [gif_clear_frames, gif_common_seed, ups_scale_mode, ups_scale_by, ups_scale_to_w, ups_scale_to_h, ups_scale_to_crop, ups_upscaler_1]

    #Grab the img2img image components for update later
    #Maybe there's a better way to do this?
    def after_component(self, component, **kwargs):
        if component.elem_id == "img2img_image":
            self.img2img_component = component
            return self.img2img_component
        if component.elem_id == "img2maskimg":
            self.img2img_inpaint_component = component
            return self.img2img_inpaint_component
        if component.elem_id == "img2img_width":
            self.img2img_width_component = component
            return self.img2img_width_component
        if component.elem_id == "img2img_height":
            self.img2img_height_component = component
            return self.img2img_height_component
    
    #Main run
    def run(self, p, gif_clear_frames, gif_common_seed, ups_scale_mode, ups_scale_by, ups_scale_to_w, ups_scale_to_h, ups_scale_to_crop, ups_upscaler_1):

        return_images, all_prompts, infotexts, inter_images = [], [], [], []
        state.job_count = len(self.readygrids) * p.n_iter
        p.do_not_save_grid = True
        p.do_not_save_samples = gif_clear_frames
        gif_n_iter = p.n_iter
        p.n_iter = 1 #we'll be processing iters per-gif-set
        outpath = os.path.join(p.outpath_samples, "gif2gif")
        print(f"Will process {gif_n_iter * p.batch_size} GIF(s) with {state.job_count * p.batch_size} total sheet generations.")
        p.width = self.readygrids[0].width
        p.height = self.readygrids[0].height
        #Iterate batch count
        for x in range(gif_n_iter):
            if state.skipped: state.skipped = False
            if state.interrupted: break
            if(gif_common_seed and (p.seed == -1)):
                p.seed = random.randrange(100000000, 999999999)
            
            #Iterate grids
            for grid in self.readygrids:
                if state.skipped: state.skipped = False
                if state.interrupted: break
                state.job = f"{state.job_no + 1} out of {state.job_count}"
                copy_p = copy.copy(p)
                copy_p.init_images = [grid] * p.batch_size
                copy_p.control_net_input_image = grid.convert("RGB") #account for controlnet
                copy_p.control_net_lowvram = True
                copy_p.control_net_resize_mode = "Just Resize"
                if grid.height > grid.width:
                    copy_p.control_net_pres = grid.height
                else:
                    copy_p.control_net_pres = grid.width
                proc = process_images(copy_p) #process
                for pi in proc.images: #Just in case another extension spits out a non-image (like controlnet)
                    if type(pi) is Image.Image:
                        inter_images.append(pi)
                all_prompts += proc.all_prompts
                infotexts += proc.infotexts
            #Separate frames by batch size
            inter_batch = []
            for b in range(p.batch_size):
                if state.skipped: state.skipped = False
                if state.interrupted: break
                for bi in inter_images[(b)::p.batch_size]:
                    inter_batch.append(bi)
                #First make temporary file via save_images, then save actual gif over it..
                #Probably a better way to do this, but this easily maintains file name and .txt file logic
                gif_filename = (modules.images.save_image(self.readygrids[0], outpath, "gif2gif", extension = 'gif', info = infotexts[b])[0])
                print(f"gif2gif: Generating GIF to {gif_filename}..")
                grid_images = []
                for gridsheet in inter_batch:
                    gridsheet = upscale(gridsheet, ups_upscaler_1, ups_scale_mode, ups_scale_by, ups_scale_to_w, ups_scale_to_h, ups_scale_to_crop)
                    grid_images += BreakGrid(gridsheet, self.desired_rows, self.desired_cols) #break the grid

                #if gif_resize:
                #    for i in range(len(grid_images)):
                #        grid_images[i] = grid_images[i].resize(self.orig_dimensions)
                grid_images = grid_images[0:self.orig_n_frames]
                #grid_images = stabilize_images(grid_images)
                grid_images[0].save(gif_filename,
                    save_all = True, append_images = grid_images[1:], loop = 0,
                    optimize = False, duration = self.desired_duration)
                if(self.desired_interp > 0):
                    print(f"gif2gif: Interpolating {gif_filename}..")
                    interp(gif_filename, self.desired_interp, self.desired_duration)
                return_images.extend(grid_images)
                inter_batch = []
            inter_images = []
        return Processed(p, return_images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)