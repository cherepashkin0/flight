import pandas as pd
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument("--config", type=str, default="config_dry_run.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Configuration loaded from {args.config}")
    print(config)

if __name__ == "__main__":
    main()
