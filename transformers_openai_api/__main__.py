import argparse
import os
from .app import make_transformers_openai_api
from .serve import run_server

def main():
    parser = argparse.ArgumentParser(
        description="An OpenAI API compatible server for locally running transformers models"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.getcwd(), "config.json"),
        help="Path to config.json (default: ./config.json)",
    )
    args = parser.parse_args()

    app = make_transformers_openai_api(args.config)
    run_server(app)

if __name__ == "__main__":
    main()