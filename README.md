# Latent Lens 

Script that takes a 3 channel 512x512 webcam input and reduces it to 16 latent dimensions. The values are streamed via OSC to 2 local network ports for sonification and visualisation. 

`network: 127.0.0.0` (localhost)
`touchDesigner port: 9999`
`puredata/MAX port: 9998`

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

## Max MSP 

- First you need to install `nn_tilde` from <a href="https://github.com/acids-ircam/nn_tilde">here<a/>.

- Then download this folder with these required files from <a href="https://drive.google.com/drive/folders/11v8k33vdOAJranC4ZX6IosW-x4-pnK_m?usp=sharing">here</a>

- Then open Max, in the top toolbar navigate to `options>file preferences`. Add a file path to the above folder. You will need to restart Max for the new packages to update.

- Finally you can open `latent_lens.maxpat` from this repository. 

