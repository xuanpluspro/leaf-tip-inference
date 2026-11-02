import argparse
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from inference.checkpoints import select_state,pet_config,compact_checkpoint
from inference.engine import predict_image
from inference.geometry import padded_size,pet_pixels,grid_pixels,valid_predictions,tile_axis,owned_predictions
from inference.images import plan_images,draw_points,save_jpeg,normalized_image
from inference.records import open_run,atomic_json,export_counts

ROOT=Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_padding(self):
        self.assertEqual(padded_size(1004,1024,256,256),(1024,1024))
        self.assertEqual(padded_size(513,511,32,32),(544,512))
        self.assertEqual(padded_size(3072,3072,16,32),(3072,3072))

    def test_pet_yx_normalized_to_xy_pixels(self):
        np.testing.assert_allclose(pet_pixels([[.25,.5]],1024,2048),[[1024,256]])

    def test_p2r_grid_centers(self):
        np.testing.assert_allclose(grid_pixels([0,2],[1,3],32,64,8,16),[[5.5,1.5],[13.5,9.5]])

    def test_filter_padding_threshold_and_nonfinite(self):
        p=np.array([[0,0],[10,1],[1,10],[2,2],[np.nan,1],[3,3]],dtype=np.float32)
        s=np.array([.9,.9,.9,.5,.9,np.inf])
        points,scores=valid_predictions(p,s,10,10,.5)
        np.testing.assert_array_equal(points,[[0,0]])

    def test_empty_points(self):
        p,s=valid_predictions([],[],32,32,.5)
        self.assertEqual(p.shape,(0,2))
        self.assertEqual(s.shape,(0,))

    def test_tile_owned_regions_cover_once(self):
        for length in [17,512,513,1004,3072]:
            coverage=np.zeros(length,dtype=int)
            for left,right,own0,own1 in tile_axis(length,512,128):
                coordinates=np.arange(length)
                coverage[(coordinates>=own0)&(coordinates<own1)]+=1
                self.assertTrue(left<=own0<=own1<=right)
            np.testing.assert_array_equal(coverage,np.ones(length))

    def test_tile_boundary_is_half_open(self):
        p,s=owned_predictions([[0,0],[10,0]],[.8,.9],20,30,(20,30,30,40))
        np.testing.assert_array_equal(p,[[20,30]])

    def test_full_mode_no_resize_one_call(self):
        class Fake:
            def predict(self,image,amp):
                self.size=image.size
                return np.array([[2.,3.]],dtype=np.float32),np.array([.9])
        adapter=Fake()
        p,s=predict_image(adapter,Image.new('RGB',(3072,3072)))
        self.assertEqual(adapter.size,(3072,3072))
        np.testing.assert_array_equal(p,[[2,3]])

    def test_tiled_duplicate_owned_once(self):
        # Simulate the same global target observed by every overlapping crop.
        image=Image.fromarray(np.tile(np.arange(700,dtype=np.uint16),(32,1)))
        class Fake:
            def predict(self,crop,amp):
                left=int(np.asarray(crop)[0,0])
                x=350-left
                if 0<=x<crop.width:
                    return np.array([[x,10.]],dtype=np.float32),np.array([.9])
                return np.empty((0,2),dtype=np.float32),np.empty(0)
        p,s=predict_image(Fake(),image,'tile',512,128)
        np.testing.assert_array_equal(p,[[350,10]])


class CheckpointTests(unittest.TestCase):
    def test_student_teacher_selection(self):
        state={'student':{'a':1},'teacher':{'a':2}}
        self.assertEqual(select_state(state,'p2rloss')['a'],1)
        self.assertEqual(select_state(state,'p2rloss','teacher')['a'],2)

    def test_compact_drops_training_state(self):
        state={'student':{'a':1},'teacher':{'a':2},'optimizer':{},'rng':{},'epoch':3}
        compact=compact_checkpoint(state,'p2rloss')
        self.assertNotIn('teacher',compact)
        self.assertNotIn('optimizer',compact)
        self.assertEqual(select_state(compact,'p2rloss'),{'a':1})

    def test_compact_wrong_model_or_teacher_rejected(self):
        compact=compact_checkpoint({'student':{},'teacher':{}},'p2rloss')
        with self.assertRaises(ValueError):
            select_state(compact,'pet')
        with self.assertRaises(ValueError):
            select_state(compact,'p2rloss','teacher')

    def test_pet_saved_args_override_defaults(self):
        config=pet_config({'args':argparse.Namespace(dec_layers=3,hidden_dim=128)})
        self.assertEqual(config['dec_layers'],3)
        self.assertEqual(config['hidden_dim'],128)
        self.assertEqual(config['backbone'],'vgg16_bn')

    def test_model_config_retained(self):
        for name in ('sah','apgcc'):
            compact=compact_checkpoint({'model':{},'model_config':{'width':64}},name)
            self.assertEqual(compact['model_config'],{'width':64})


class FileTests(unittest.TestCase):
    def test_clean_names_and_unicode(self):
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'输入'
            source.mkdir()
            (source/'叶片.JPG').touch()
            self.assertEqual(plan_images(source,Path(tmp)/'输出')[0][2].name,'叶片_overlay.jpg')

    def test_same_stem_rejected(self):
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'in'
            source.mkdir()
            (source/'a.jpg').touch()
            (source/'a.png').touch()
            with self.assertRaises(ValueError):
                plan_images(source,Path(tmp)/'out')

    def test_output_inside_input_rejected(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plan_images(tmp,Path(tmp)/'out')

    def test_jpeg_points_preserve_dimensions(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'中文/图_overlay.jpg'
            image=draw_points(Image.new('RGB',(100,80)),[[70,60]],2)
            self.assertEqual(image.getpixel((70,60)),(255,0,0))
            save_jpeg(image,p)
            with Image.open(p) as saved:
                self.assertEqual(saved.size,(100,80))
                self.assertEqual(saved.format,'JPEG')

    def test_normalization(self):
        x=normalized_image(Image.new('RGB',(3,2),(255,0,0)))
        self.assertEqual(x.shape,(3,2,3))
        self.assertAlmostEqual(float(x[0,0,0]),(1-.485)/.229,places=5)

    def test_resume_and_overwrite(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'out'
            state,path=open_run(output,{'model':'pet'})
            state['completed']['a']={}
            atomic_json(path,state)
            self.assertIn('a',open_run(output,{'model':'pet'},resume=True)[0]['completed'])
            with self.assertRaises(ValueError):
                open_run(output,{'model':'sah'},resume=True)
            (output/'keep.txt').write_text('retain')
            state,_=open_run(output,{'model':'sah'},overwrite=True)
            self.assertFalse(state['completed'])
            self.assertEqual((output/'keep.txt').read_text(),'retain')

    def test_cli_default_and_conflict(self):
        from predict import parser
        args=parser().parse_args(['--model','pet','--weights','x.pth','--input','images','--output','out'])
        self.assertEqual(args.mode,'full')
        self.assertFalse(args.amp)
        self.assertTrue(args.save_points)

    def test_vendor_syntax_and_isolated_imports(self):
        for path in (ROOT/'vendor').rglob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level==0:
                    self.assertNotEqual(node.module,'util.misc')
        source=(ROOT/'vendor/pet/models/backbones/backbone_vgg.py').read_text()
        self.assertIn('vgg16_bn(pretrained=False)',source)
        p2=(ROOT/'vendor/p2rloss/models/vgg16bn.py').read_text()
        self.assertNotIn('torch.load(',p2)


if __name__=='__main__':
    unittest.main()
