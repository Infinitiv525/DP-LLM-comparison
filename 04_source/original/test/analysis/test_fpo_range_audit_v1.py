import io
import unittest
from unittest.mock import patch
import numpy as np
from audit_flowbench_fpo_headers_v1 import RangeFile,header


class RangeTests(unittest.TestCase):
    def test_unbounded_large_read_rejected(self):
        with self.assertRaises(ValueError):RangeFile('https://example.invalid/a',2**30).read()

    def test_budget_rejected_before_network(self):
        with self.assertRaises(ValueError):RangeFile('https://example.invalid/a',100,budget=2).read(3)

    def test_server200_body_not_read(self):
        class Response:
            status=200
            headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):raise AssertionError('Full response body must not be read')
        with patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaises(ValueError):RangeFile('https://example.invalid/a',100).read(3)

    def test_seek_end_and_small_tail_read(self):
        class Response:
            status=206
            headers={'Content-Range':'bytes 97-99/100'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return b'abc'
        r=RangeFile('https://example.invalid/a',100);r.seek(-3,2)
        with patch('urllib.request.urlopen',return_value=Response()):self.assertEqual(r.read(),b'abc')
        self.assertEqual(r.tell(),100);self.assertEqual(r.used,3)

    def test_numeric_header(self):
        b=io.BytesIO();np.save(b,np.zeros((2,3)));b.seek(0)
        self.assertEqual(header(b)[0],(2,3))

    def test_object_header_refused(self):
        b=io.BytesIO();np.save(b,np.array([None],dtype=object));b.seek(0)
        with self.assertRaises(ValueError):header(b)


if __name__=='__main__':unittest.main(verbosity=2)
