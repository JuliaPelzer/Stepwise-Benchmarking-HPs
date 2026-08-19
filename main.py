import argparse
from pathlib import Path

from code.train import train, prep_for_competition
from code.eval import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=['1', '2', '3', 'Dummy'], default="Dummy")
    parser.add_argument("--mode", choices=["train", "eval", "comp"], default="train")
    args = parser.parse_args()

    try:
        args.step = int(args.step)
    except ValueError: pass

    print(f"Running step {args.step} in {args.mode} mode")
    # check for data, model-implementation, and in case of evaluation also for model weights
    if args.step == 1:
        assert Path("data/step1").exists(), "Data for step 1 does not exist, please download it first"
        try:
            from code.model import Step1
        except ImportError:
            print("Model file for step 1 does not exist or does not contain a class called Step1, please provide a valid model file and class")
            return
        if args.mode == "eval":
            assert Path("results/step1_model.pt").exists(), "Model weights file results/step1_model.pt does not exist, please provide a valid model file and weights for evaluation"
        
    elif args.step == 2:
        assert Path("data/step2").exists(), "Data for step 2 does not exist, please download it first"
        try:
            from code.model import Step2
        except ImportError:
            print("Model file for step 2 does not exist or does not contain a class called Step2, please provide a valid model file and class")
            return
        if args.mode == "eval":
            assert Path("results/step2_model.pt").exists(), "Model weights file results/step2_model.pt does not exist, please provide a valid model file and weights for evaluation"
    elif args.step == 3:
        assert Path("data/step3").exists(), "Data for step 3 does not exist, please download it first"
        try:
            from code.model import Step3
        except ImportError:
            print("Model file for step 3 does not exist or does not contain a class called Step3, please provide a valid model file and class")
            return
        if args.mode == "eval":
            assert Path("results/step3_model.pt").exists(), "Model weights file results/step3_model.pt does not exist, please provide a valid model file and weights for evaluation"

    Path("results").mkdir(exist_ok=True)

    # train and/or evaluate model
    if args.mode == "train":
        train(args.step)

    if args.mode in ["train", "eval"]:
        evaluate(args.step)

    elif args.mode == "comp":
        prep_for_competition(args.step)

if __name__ == "__main__":
    main()