# pull official base image
FROM python:3.8-buster

# set work directory
WORKDIR /app

# environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG False
ENV SECRET_KEY 0

RUN pip install --upgrade pip

# install dependencies
COPY ./requirements.txt .
RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install -r requirements.txt
# copy project
COPY . .

RUN python manage.py collectstatic --noinput

# run gunicorn
CMD gunicorn deep_app.wsgi:application --bind 0.0.0.0:$PORT