Steps:
- ./setup.sh
- source venv/bin/activate
- cp models/RealVisXL_V4.0_Lightning.safetensors repositories/stable-diffusion-webui/models/Stable-diffusion/
- cd repositories/stable-diffusion-webui/
- bash webui.sh --xformers --reinstall-xformers --api
