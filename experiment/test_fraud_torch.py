import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fraud_torch import *

class PipelineTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(step=[0,6,12,24]*10, type=TYPES[:4]*10, amount=[0,1,2,3]*10, nameDest=['M1','C1','C2','M2']*10, isFraud=[0,0,0,1]*10))

    def test_features_and_validation(self):
        f = self.frame()
        s = fit_scaler(f)
        x = build_features(f,s)
        self.assertEqual(list(x), FEATURES)
        self.assertAlmostEqual(x.log_amount_z.mean(), 0)
        self.assertAlmostEqual(x.hour_sin.iloc[1], 1)
        self.assertEqual(x.transfer_amount.iloc[0], 0)
        before = s.mean_.copy()
        build_features(f.assign(amount=999),s)
        np.testing.assert_array_equal(before,s.mean_)
        self.assertEqual(fit_scaler(f.assign(amount=1)).scale_[0],1)
        for col,val in [('amount',-1),('step',.5),('isFraud',2),('type','BAD')]:
            with self.assertRaises(ValueError): preprocess(f.assign(**{col:val}))

    def test_metrics_threshold(self):
        y = np.array([0,0,1,1])
        p = np.array([.1,.4,.35,.8])
        self.assertAlmostEqual(evaluate(y,p)['average_precision'],5/6)
        self.assertEqual(select_threshold(y,p),.35)
        self.assertEqual(select_threshold(y,np.full(4,.5)),.5)
        self.assertEqual(evaluate(y,np.full(4,.5))['average_precision'],.5)
        self.assertAlmostEqual(evaluate(y,np.full(4,.5),positive_weight=3)['weighted_bce'],2*np.log(2))
        self.assertIsNone(evaluate([0],[.2],stats=True)['probability_stats']['fraud'])
        with self.assertRaises(ValueError): select_threshold([0],[.2])
        rng = np.random.default_rng(2)
        for _ in range(20):
            scores = rng.random(4)
            t = select_threshold(y,scores)
            self.assertAlmostEqual(evaluate(y,scores,t)['f2'],max(evaluate(y,scores,v)['f2'] for v in scores))

    def test_sample_train_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.frame()
            path = Path(tmp)/'data.csv'
            f.to_csv(path,index=False)
            a,b = load_sample(path,20),load_sample(path,20)
            pd.testing.assert_frame_equal(a,b)
            self.assertEqual(a.isFraud.sum(),5)
            s = fit_scaler(a)
            x = build_features(a,s)
            c = Config(batch_size=7)
            model = build_model(c)
            opt = build_optimizer(model,c)
            loader = make_loader(x,a.isFraud,7)
            self.assertEqual([len(batch[0]) for batch in loader],[7,7,6])
            train_epoch(model,loader,opt,3)
            p = predict(model,x)
            save_checkpoint(Path(tmp)/'model.pt',model,opt,s,.3,c)
            restored,optimizer,scaler,_,_ = load_checkpoint(Path(tmp)/'model.pt')
            np.testing.assert_array_equal(p,predict(restored,build_features(a,scaler)))
            train_epoch(restored,loader,optimizer,3)
            self.assertFalse(np.array_equal(p,predict(restored,x)))

if __name__ == '__main__': unittest.main()
