python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir output
git config --global user.email rishabh.srivastava@phot.ai
git config --global user.name rishabh
mkdir models
cd models &&\
 wget https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning/resolve/main/RealVisXL_V4.0_Lightning.safetensors && \
 wget https://huggingface.co/licyk/sd-vae/resolve/main/sdxl_1.0/sdxl_fp16_fix_vae.safetensors 
cd ..
mkdir repositories
cd repositories && git clone https://github.com/siliconflow/onediff.git \
    && git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd ..
cd repositories/onediff && python3 -m pip install -e .
cd ../..