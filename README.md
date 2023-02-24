# Concurrent-gif2gif
Experimental Automatic1111 Stable Diffusion WebUI extension, concurrent frame rendering. Plots an animation onto sheets of frames before generating a result. Intention is to create more consistency across rendered frames.

Very much a work-in-progress, but works.

**Instructions**
- Upload a GIF in script window
- Select number of rows and columns, being mindful of generation limits of your hardware
- img2img size sliders are overridden by this script; ignore them
- controlnet resolution, lowvram, and resize mode are overridden by this script; ignore them
- Push "generate sheets," which may take some time. Gallery should populate
- Generate as normal. Completed gif will be saved to ...img2img/gif2gif/

- 2/24/23: Restricted grid orientation; fixed controlnet issue
- 2/24/23: Removed size restrictions on sheets; no longer need to be square

![z7e1hnyqjzia1](https://user-images.githubusercontent.com/93007558/220376855-c586c6c0-8760-47b6-8c68-4c8f33509dc5.gif)
![tenor](https://user-images.githubusercontent.com/93007558/220376925-343f6b9f-81c6-440f-a42d-aeeb64adcb50.gif)
![image](https://user-images.githubusercontent.com/93007558/220376568-29705fcd-ded2-4139-a165-8a461b044190.png)
![image](https://user-images.githubusercontent.com/93007558/220376618-d60cea82-faae-428d-a91d-4fcd0a2175ad.png)
