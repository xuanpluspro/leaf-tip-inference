import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from inference.annotations import annotation_payload, corrected_points, export_annotations, annotations_to_points, relative_path
from inference.records import atomic_json


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        self.images=self.root/'原图';self.images.mkdir()
        self.image=self.images/'子目录/叶 尖.jpg';self.image.parent.mkdir();Image.new('RGB',(100,80),(20,100,30)).save(self.image)
        self.key='子目录/叶 尖.jpg';self.run=self.root/'prediction';self.review=self.root/'review'
        self.rows=[dict(x=12.25,y=34.5,score=.8),dict(x=60.,y=50.,score=.95)]
        self.record=dict(image=self.key,count=2,width=100,height=80,coordinate_order='xy',coordinate_unit='original-image pixels',points=self.rows)
        atomic_json(self.run/'points/子目录/叶 尖.json',self.record)
        atomic_json(self.run/'meta/run.json',dict(completed={self.key:dict(points='points/子目录/叶 尖.json',count=2,width=100,height=80)}))
    def tearDown(self):self.tmp.cleanup()
    def test_schema_xy_single_points(self):
        p=annotation_payload('a.jpg',100,80,self.rows)
        self.assertIsNone(p['imageData']);self.assertFalse(p['checked'])
        self.assertEqual(p['shapes'][0]['points'],[[12.25,34.5]])
        self.assertEqual(p['shapes'][0]['shape_type'],'point')
        self.assertEqual(p['shapes'][0]['label'],'leaf_tip')
    def test_export_portable_original_bytes_and_new_name(self):
        result=export_annotations(self.images,self.run,self.review)
        self.assertEqual(result['points'],2)
        self.assertEqual((self.review/self.key).read_bytes(),self.image.read_bytes())
        p=json.loads((self.review/'子目录/叶 尖.json').read_text())
        self.assertEqual(p['imagePath'],'叶 尖.jpg')
        self.assertTrue((self.review/'子目录'/p['imagePath']).is_file())
    def test_manual_changes_cannot_be_overwritten(self):
        export_annotations(self.images,self.run,self.review)
        p=self.review/'子目录/叶 尖.json';p.write_text('manual changes')
        with self.assertRaises(FileExistsError):export_annotations(self.images,self.run,self.review)
        self.assertEqual(p.read_text(),'manual changes')
    def test_min_score_explicit(self):
        r=export_annotations(self.images,self.run,self.review,min_score=.9)
        self.assertEqual(r['points'],1)
    def test_roundtrip_human_delete_move_add(self):
        export_annotations(self.images,self.run,self.review)
        p=self.review/'子目录/叶 尖.json';payload=json.loads(p.read_text())
        payload['shapes'].pop();payload['shapes'][0]['points']=[[15.125,35.75]]
        payload['shapes'].append(dict(label='leaf_tip',shape_type='point',points=[[75,65]]))
        atomic_json(p,payload)
        r=annotations_to_points(self.review,self.root/'PointAnnotations')
        self.assertEqual(r['points'],2)
        self.assertEqual((self.root/'PointAnnotations/子目录/叶 尖.txt').read_text(),'15.125 35.75\n75.0 65.0\n')
    def test_empty_shapes_exports_empty_txt(self):
        self.record['points']=[];self.record['count']=0
        atomic_json(self.run/'points/子目录/叶 尖.json',self.record)
        atomic_json(self.run/'meta/run.json',dict(completed={self.key:dict(points='points/子目录/叶 尖.json',count=0,width=100,height=80)}))
        export_annotations(self.images,self.run,self.review);annotations_to_points(self.review,self.root/'txt')
        self.assertEqual((self.root/'txt/子目录/叶 尖.txt').read_text(),'')
    def test_wrong_dimensions_no_partial_export(self):
        Image.new('RGB',(101,80)).save(self.image)
        with self.assertRaises(ValueError):export_annotations(self.images,self.run,self.review)
        self.assertFalse(self.review.exists())
    def test_exif_orientation_rejected(self):
        exif=Image.Exif();exif[274]=6;Image.new('RGB',(100,80)).save(self.image,exif=exif)
        with self.assertRaisesRegex(ValueError,'EXIF'):export_annotations(self.images,self.run,self.review)
    def test_invalid_points_rejected(self):
        for x in (float('nan'),float('inf'),100,-1,True):
            with self.assertRaises(ValueError):annotation_payload('a.jpg',100,80,[dict(x=x,y=1,score=.9)])
    def test_wrong_shape_label_duplicates_rejected(self):
        p=annotation_payload('a.jpg',100,80,self.rows)
        p['shapes'].append(p['shapes'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):corrected_points(p)
        p=annotation_payload('a.jpg',100,80,self.rows);p['shapes'][0]['shape_type']='rectangle'
        with self.assertRaises(ValueError):corrected_points(p)
        p=annotation_payload('a.jpg',100,80,self.rows);p['shapes'][0]['label']='other'
        with self.assertRaises(ValueError):corrected_points(p)
    def test_unsafe_paths_rejected(self):
        for path in ('../a.jpg','/root/a.jpg','C:/a.jpg','..\\a.jpg'):
            with self.assertRaises(ValueError):relative_path(path)
    def test_export_inside_input_rejected(self):
        with self.assertRaises(ValueError):export_annotations(self.images,self.run,self.images/'review')
    def test_conversion_refuses_existing(self):
        export_annotations(self.images,self.run,self.review)
        out=self.root/'txt';out.mkdir();(out/'keep.txt').write_text('existing')
        with self.assertRaises(FileExistsError):annotations_to_points(self.review,out)
        self.assertEqual((out/'keep.txt').read_text(),'existing')
    def test_stale_record_count_rejected(self):
        self.record['count']=100;atomic_json(self.run/'points/子目录/叶 尖.json',self.record)
        with self.assertRaises(ValueError):export_annotations(self.images,self.run,self.review)

if __name__=='__main__':unittest.main()
