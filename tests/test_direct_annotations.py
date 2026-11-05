import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from PIL import Image
from inference.annotations import annotation_payload, save_annotation_pair, corrected_points
from inference.images import draw_points


class DirectAnnotationTests(unittest.TestCase):
    def test_portable_pair(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            source=root/'source.png'
            Image.new('RGB',(64,64)).save(source)
            target=root/'annotations/sub/source.png'
            payload=annotation_payload(target.name,64,64,[dict(x=12.25,y=25.5,score=.9)])
            save_annotation_pair(source,target,payload)
            self.assertEqual(source.read_bytes(),target.read_bytes())
            data=json.loads(target.with_suffix('.json').read_text())
            self.assertIsNone(data['imageData'])
            self.assertEqual(data['imagePath'],'source.png')
            self.assertEqual(corrected_points(data),[(12.25,25.5)])

    def test_large_point_and_white_outline(self):
        original=Image.new('RGB',(3072,3072),(10,20,30))
        result=draw_points(original,[(1500,1500)])
        self.assertEqual(result.size,original.size)
        self.assertEqual(original.getpixel((1500,1500)),(10,20,30))
        self.assertEqual(result.getpixel((1500,1500)),(255,0,0))
        self.assertEqual(result.getpixel((1515,1500)),(255,255,255))
        self.assertEqual(result.getpixel((1516,1500)),(10,20,30))
        small=draw_points(original,[(1500,1500)],4)
        self.assertEqual(small.getpixel((1505,1500)),(10,20,30))

    def test_zero_predictions(self):
        self.assertEqual(annotation_payload('a.jpg',30,30,[])['shapes'],[])


if __name__=='__main__':
    unittest.main()
