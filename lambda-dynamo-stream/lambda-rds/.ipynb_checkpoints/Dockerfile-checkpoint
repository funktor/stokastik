FROM public.ecr.aws/lambda/python:3.8
RUN pip3 install psycopg2-binary
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY app.py rds_config.py ./
CMD ["app.handler"]