FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

<<<<<<< HEAD
RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
RUN mkdir output

COPY ["hmwk04_score_docker.py", "./"]

ENTRYPOINT ["python3", "hmwk04_score_docker.py"]
=======
WORKDIR /app
COPY [ "model2.bin", "model.bin" ]
    
>>>>>>> b0d76f3f2d20456a29b4ad9637f648a50f2ceba8
