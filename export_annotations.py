"""Export existing inference points as a portable Labelme/X-AnyLabeling package."""
import argparse
import json
from pathlib import Path
from inference.annotations import export_annotations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='Original images used for inference')
    p.add_argument('--predictions',required=True,type=Path,help='Run root containing meta/run.json and points/')
    p.add_argument('--output',required=True,type=Path,help='NEW directory for original image copies + editable JSON')
    p.add_argument('--label',default='leaf_tip')
    p.add_argument('--min-score',type=float,help='Optional extra filter; default keeps all saved predictions')
    a=p.parse_args()
    print(json.dumps(export_annotations(a.input,a.predictions,a.output,a.label,a.min_score),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
