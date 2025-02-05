FROM python:3.10-alpine

WORKDIR /code

RUN apk add --no-cache \
    make \
    curl \
    gcc \
    musl-dev \
    python3-dev

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY ./pyproject.toml .
COPY ./uv.lock .
RUN uv pip install --system .

ARG PORT

COPY . /code/
EXPOSE $PORT

CMD ["make", "dev"]
