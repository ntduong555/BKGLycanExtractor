from __future__ import print_function

from pygly.MonoFactory import MonoFactory
from pygly.Glycan import Glycan

mf = MonoFactory()
m1 = mf.new("GlcNAc")
m2 = mf.new("GlcNAc")
m1.add_child(m2)

g = Glycan(m1)

