"""Convert human-reviewed Labelme/X-AnyLabeling point JSON back to xy TXT."""
import argparse
import json
from pathlib import Path
from inference.annotations import annotations_to_points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='Reviewed package with JSON and original images')
    p.add_argument('--output',required=True,type=Path,help='NEW PointAnnotations directory')
    p.add_argument('--label',default='leaf_tip')
    a=p.parse_args()
    print(json.dumps(annotations_to_points(a.input,a.output,a.label),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
