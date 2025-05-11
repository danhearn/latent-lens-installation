# Latent Lens 

Script that takes a 3 channel 512x512 webcam input and reduces it to 16 latent dimensions. The values are streamed via OSC to 2 local network ports. 

`network: 127.0.0.0` (localhost)
`touchDesigner port: 9999`
`puredata/MAX portL 9998`

## Setup 

First cd into the project and create a new python virtual environment: 

````
cd ../../latent-lens-installation
python -m venv venv
source venv/bin/active
pip intall uv
pip install -r requirements.txt     

````

## Running 

With the venv activated, run `main.py` 

`python main.py`

