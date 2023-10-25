FROM python:3.10-slim-bullseye 
# ref: https://hub.docker.com/_/python
COPY requirements.txt /bot/

WORKDIR /bot
# below might not be required if switched to headless install for opencv
RUN apt-get update && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "src/bot.py"]

# Dockerfile ref: https://www.pythondiscord.com/pages/guides/python-guides/docker-hosting-guide/

# BUILD
# cd /path/to/project
# sudo docker build -t ubot .

# RUN
# sudo docker run -d --name ultrabot ubot:latest

# VIEW LOGS
# sudo docker logs -f ultrabot

# VIEW CONTAINER FILES (not recommended)
# sudo docker exec -it ultrabot bash