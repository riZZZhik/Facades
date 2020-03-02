# to build use: docker image build -t oneonwar:jptr Docker
# to run use: nvidia-docker run -it --rm -v ~/Facades/:/Facades/ -p 4958:4958 oneonwar:jptr

# Use base Python image
FROM python:3.7

# Maintainer contact
LABEL maintain="t.me/riZZZhik"

# Add out work directory to Docker # TODO: Fucking why it is not working?
#COPY requirements.txt /tmp
#WORKDIR /tmp

# Install python packages
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyterlab

# Run jupyter lab
WORKDIR /
CMD jupyter lab --ip 0.0.0.0 --port 4958 --no-browser --allow-root