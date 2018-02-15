FROM tensorflow/tensorflow
RUN mkdir /app
COPY * /app/
RUN apt-get update && apt-get install wget git -y && git clone http://github.com/russell91/tensorbox
RUN cd tensorbox && ./download_data.sh --travis_tiny_data
RUN cd utils && make && cd ..
CMD [ "python", "/app/sleep.py" ]

