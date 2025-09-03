Run the notebook inside Docker (recommended, Pyfhel requires Linux):

1. Build the image from the repository root:

   docker build -t he-fl -f Dockerfile .

2. Start a container and run a Jupyter server (bind port 8888):

   docker run --rm -it -p 8888:8888 -v "$(pwd)":/workspace he-fl bash -c "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"

3. Open the notebook at the printed token URL and run `Encrypted FL Main-Rel.ipynb`.

Notes:
- Building Pyfhel inside the container may take several minutes.
- If you prefer not to use Docker, use a Linux VM with Python 3.8 and install packages from `requirements.txt`.
