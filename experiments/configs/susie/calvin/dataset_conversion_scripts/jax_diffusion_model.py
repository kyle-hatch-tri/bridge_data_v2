from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
# from susie_functions.model import create_sample_fn
# from susie_functions.jax_utils import initialize_compilation_cache
import numpy as np
from PIL import Image
import os

import time

class DiffusionModel:
    def __init__(self, num_denoising_steps=200, num_samples=1):
        initialize_compilation_cache()

        self.num_samples = num_samples

        self.sample_fn = create_sample_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            # num_timesteps=200,
            num_timesteps=num_denoising_steps,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
            num_samples=self.num_samples,
        )


        # self.sample_fn = 
        

    def generate(self, language_command : str, image_obs : np.ndarray):
        # Resize image to 256x256
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)


        t0 = time.time()
        sample = self.sample_fn(image_obs, language_command, prompt_w=7.5, context_w=1.5)

        t1 = time.time()
        print(f"\t>>t1 - t0: {t1 - t0:.3f}")
        # return np.array(Image.fromarray(sample).resize((200, 200))).astype(np.uint8)

        return np.array([np.array(Image.fromarray(s).resize((200, 200))).astype(np.uint8) for s in sample])
    
    
    

if __name__ == "__main__":
    model = DiffusionModel()