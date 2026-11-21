import os

from waitress import serve

from .web import create_app


def main():
    server = create_app()
    serve(server, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
          threads=5, ident="ReelTone", channel_timeout=45, max_request_body_size=68 * 1024 * 1024)


if __name__ == "__main__":
    main()
