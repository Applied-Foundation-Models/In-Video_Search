FROM ubuntu:latest
LABEL authors="baatout"

ENTRYPOINT ["top", "-b"]