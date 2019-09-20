FROM python:3.6
WORKDIR /app
COPY . .
ENV BASE_PATH=/app
RUN pip3 install -r ./requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
EXPOSE 5000
CMD ["python3", "app.py"]