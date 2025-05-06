import argparse
from mi_agent import orchestrator

def main():
    p = argparse.ArgumentParser(
        prog="mi_agent",
        description="Run the MI-Agent pipeline end-to-end"
    )
    p.add_argument(
        "-p", "--problem-file",
        help="Path to a .txt file containing the problem statement"
    )
    p.add_argument(
        "-o", "--output-dir",
        help="Where to save generated code, images, PDF (defaults to data/)",
    )
    p.add_argument(
        "-m", "--model",
        help="OpenAI model name to use (defaults to gpt-4.1-mini)",
    )
    args = p.parse_args()

    orchestrator.run(problem_txt_path=args.problem_file, output_dir=args.output_dir, model_name=args.model)

if __name__ == "__main__":
    main()
