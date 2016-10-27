FROM continuumio/miniconda3

MAINTAINER IM version: 0.1

RUN echo "deb http://http.debian.net/debian jessie-backports main" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
	g++ \
	openjdk-8-jre-headless \
	libpng-dev \
    libfreetype6-dev

COPY ./scripts/bayespy /scripts/bayespy

WORKDIR /scripts/bayespy
RUN pip install -e .




	