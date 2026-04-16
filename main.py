import argparse
import yaml
from pathlib import Path

from code.train import train
from code.eval import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=[1, 2, 3, "Dummy"], default="Dummy")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timedependent", type=bool, default=False, help="Whether to use the time-dependent dataset (True) or the last-timestep dataset (False)")
    args = parser.parse_args()

    # check for data, model-implementation, and in case of evaluation also for model weights
    if args.step == 1:
        assert Path("data/step1").exists(), "Data for step 1 does not exist, please download it first"
        try:
            from lorentz.code.model import Step1
        except ImportError:
            print("Model file for step 1 does not exist or does not contain a class called Step1, please provide a valid model file and class")
            return
        if args.mode == "eval":
            assert Path("step1.pt").exists(), "Model weights file step1.pt does not exist, please provide a valid model file and weights for evaluation"
        
    elif args.step == 2:
        assert Path("data/step2").exists(), "Data for step 2 does not exist, please download it first"
        try:
            from lorentz.code.model import Step2
        except ImportError:
            print("Model file for step 2 does not exist or does not contain a class called Step2, please provide a valid model file and class")
            return
        if args.mode == "eval":
            assert Path("step2.pt").exists(), "Model weights file step2.pt does not exist, please provide a valid model file and weights for evaluation"
    elif args.step == 3:
        assert Path("data/step3").exists(), "Data for step 3 does not exist, please download it first"
        try:
            from lorentz.code.model import Step3
        except ImportError:
            print("Model file for step 3 does not exist or does not contain a class called Step3, please provide a valid model file and class")
            return
        if args.mode == "eval":
            assert Path("step3.pt").exists(), "Model weights file step3.pt does not exist, please provide a valid model file and weights for evaluation"

    # train and/or evaluate model
    if args.mode == "train":
        train(args.step, args.timedependent)

    evaluate(args.step, args.timedependent)

if __name__ == "__main__":
    main()