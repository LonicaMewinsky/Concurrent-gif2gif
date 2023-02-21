import copy
import os
import modules.scripts as scripts
import modules.images
import gradio as gr
import numpy as np
import tempfile
import random
from PIL import Image, ImageSequence
from modules.processing import Processed, process_images
from modules.shared import state

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
        self.slowmo = False
        self.gif2gifdir = tempfile.TemporaryDirectory()
        self.img2img_component = gr.Image()
        self.img2img_inpaint_component = gr.Image()
        self.img2img_width_component = gr.Slider()
        self.img2img_height_component = gr.Slider()
        return None

    def title(self):
        return "Concurrent gif2gif"
    def show(self, is_img2img):
        return is_img2img
    
    def ui(self, is_img2img):
        #Controls
        with gr.Box():    
            with gr.Column():
                upload_gif = gr.File(label="Upload GIF", file_types = ['.gif','.webp','.plc'], live=True, file_count = "single")
                display_gif = gr.Image(inputs = upload_gif, visible = False, label = "Preview GIF", type= "filepath").style(height=480)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            with gr.Box():
                                grid_x_slider = gr.Slider(1, 10, step = 1, interactive=True, label = "Grid rows")
                                grid_y_slider = gr.Slider(1, 10, step = 1, interactive=True, label = "Grid columns") 
                                gif_clear_frames = gr.Checkbox(value = True, label="Delete intermediate frames after GIF generation")
                                gif_common_seed = gr.Checkbox(value = True, label="For -1 seed, all frames in a GIF have common seed")
                        with gr.Column():
                            with gr.Box():
                                frames_per_sheet = gr.Textbox(value="", interactive = False, label = "Frames per sheet")
                                number_of_sheets = gr.Textbox(value="", interactive = False, label = "Number of sheets")
                grid_gen_button = gr.Button(value = "Generate sheets")
                sheet_gallery = gr.Gallery(label = "Sheets for generation")                
                with gr.Accordion("Animation tweaks", open = False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Box():
                                fps_slider = gr.Slider(1, 50, step = 1, label = "Desired FPS")
                                interp_slider = gr.Slider(label = "Interpolation frames", value = 0)
                                gif_resize = gr.Checkbox(value = True, label="Resize result back to original dimensions")
                                gif_clear_frames = gr.Checkbox(value = True, label="Delete intermediate frames after GIF generation")
                                gif_common_seed = gr.Checkbox(value = True, label="For -1 seed, all frames in a GIF have common seed")
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
        
        def gridgif(gif, cols, rows):
            pilframes = []
            grids = []
            framesper = cols * rows
            init_gif = Image.open(gif.name)
            #Break gif
            for frame in ImageSequence.Iterator(init_gif):
                interm = frame.convert("RGB")
                pilframes.append(interm)
            #Make chunks
            pilchunks = [pilframes[i:i+framesper] for i in range(0, len(pilframes), framesper)]
            #Make grids from the chunks
            for chunk in pilchunks:
                grids.append(MakeGrid(chunk, cols, rows))
            #Update vanilla UI
            img_for_ui_path = (f"{self.gif2gifdir.name}/imgforui.gif")
            img_for_ui = grids[0]
            if img_for_ui.height < 480:
                img_for_ui = img_for_ui.resize((round(480*img_for_ui.width/img_for_ui.height), 480), Image.Resampling.LANCZOS)
            img_for_ui.save(img_for_ui_path)

            return grids, img_for_ui, img_for_ui, framesper, len(grids)

        #Control change events
        fps_slider.change(fn=fpsupdate, inputs = [fps_slider, interp_slider], outputs = [display_gif, fps_actual, seconds_actual, frames_actual])
        interp_slider.change(fn=fpsupdate, inputs = [fps_slider, interp_slider], outputs = [display_gif, fps_actual, seconds_actual, frames_actual])
        upload_gif.upload(fn=processgif, inputs = upload_gif, outputs = [display_gif, display_gif, fps_slider, fps_original, seconds_original, frames_original])
        upload_gif.change(fn=cleargif, inputs = upload_gif, outputs = display_gif)
        grid_gen_button.click(fn=gridgif, inputs = [upload_gif, grid_x_slider, grid_y_slider], outputs = [sheet_gallery, self.img2img_component, self.img2img_inpaint_component, frames_per_sheet, number_of_sheets])

        return [gif_resize, gif_clear_frames, gif_common_seed]

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