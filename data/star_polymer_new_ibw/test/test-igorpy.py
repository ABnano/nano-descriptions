# Copyright (C) 2012 W. Trevor King <wking@tremily.us>
#
# This file is part of %(project)s.
#
# %(project)s is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# %(project)s is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with %(project)s.  If not, see <http://www.gnu.org/licenses/>.

r"""Test the igor.igorpy compatibility layer by loading sample files.

>>> from pprint import pprint
>>> import igor.igorpy as igor
>>> igor.ENCODING = 'UTF-8'

Load a packed experiment:

>>> path = data_path('polar-graphs-demo.pxp')
>>> d = igor.load(path)
>>> print(d)
<igor.Folder root>
>>> dir(d)  # doctest: +ELLIPSIS
['Packages', 'W_plrX5', 'W_plrX6', ..., 'radiusData', 'radiusQ1']


Navigation:

>>> print(d.Packages)
<igor.Folder root/Packages>
>>> print(d[0])  # doctest: +ELLIPSIS
<igor.igorpy.Variables object at 0x...>


Variables:

>>> v = d[0]
>>> dir(v)  # doctest: +ELLIPSIS
['__class__', ..., 'depstr', 'depvar', 'format', 'sysvar', 'userstr', 'uservar']
>>> v.depstr
{}
>>> v.depvar
{}
>>> v.format()
'<Variables: system 21, user 0, dependent 0>'
>>> pprint(v.sysvar)  # doctest: +REPORT_UDIFF
{'K0': 0.0,
 'K1': 0.0,
 'K10': 0.0,
 'K11': 0.0,
 'K12': 0.0,
 'K13': 0.0,
 'K14': 0.0,
 'K15': 0.0,
 'K16': 0.0,
 'K17': 0.0,
 'K18': 0.0,
 'K19': 0.0,
 'K2': 0.0,
 'K20': 128.0,
 'K3': 0.0,
 'K4': 0.0,
 'K5': 0.0,
 'K6': 0.0,
 'K7': 0.0,
 'K8': 0.0,
 'K9': 0.0}
>>> v.userstr
{}
>>> v.uservar
{}


Waves:

>>> d.W_plrX5
<igor.Wave W_plrX5 data (128)>
>>> dir(d.W_plrX5)  # doctest: +ELLIPSIS
['__array__', ..., 'axis', 'axis_units', 'data', ..., 'name', 'notes']
>>> d.W_plrX5.axis  # doctest: +ELLIPSIS
[array([ 0.04908739,  0.04870087,  0.04831436,  0.04792784,  0.04754133,
        0.04715481,  0.0467683 ,  0.04638178,  0.04599527,  0.04560875,
        ...
        0.00077303,  0.00038651,  0.        ]), array([], dtype=float64), array([], dtype=float64), array([], dtype=float64)]
>>> d.W_plrX5.data_units
(u'', '', '', '')
>>> d.W_plrX5.axis_units
(u'', '', '', '')
>>> d.W_plrX5.data  # doctest: +ELLIPSIS
array([  1.83690956e-17,   2.69450769e-02,   7.65399113e-02,
         1.44305170e-01,   2.23293692e-01,   3.04783821e-01,
         ...
        -2.72719120e-03,   5.24539061e-08], dtype=float32)


Dump the whole thing:

>>> print(d.format())
root
  <Variables: system 21, user 0, dependent 0>
  <History>
  radiusData data (128)
  angleData data (128)
  W_plrX5 data (128)
  W_plrY5 data (128)
  angleQ1 data (64)
  radiusQ1 data (64)
  W_plrX6 data (64)
  W_plrY6 data (64)
  Packages
    WMDataBase
      <Variables: system 21, user 6, dependent 0>
    PolarGraphs
      <Variables: system 21, user 38, dependent 0>
  <Recreation>
  <GetHistory>
  <Procedure>


Load a packed experiment without ignoring unknown records:

>>> d = igor.load(path, ignore_unknown=False)
>>> print(d.format())
root
  <Unknown type 11>
  <Unknown type 12>
  <Unknown type 13>
  <Unknown type 13>
  <Unknown type 13>
  <Unknown type 13>
  <Unknown type 13>
  <Unknown type 13>
  <Unknown type 13>
  <Unknown type 14>
  <Unknown type 15>
  <Unknown type 16>
  <Unknown type 16>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 16>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 17>
  <Unknown type 18>
  <Unknown type 11>
  <Unknown type 26>
  <Unknown type 26>
  <Variables: system 21, user 0, dependent 0>
  <History>
  radiusData data (128)
  angleData data (128)
  W_plrX5 data (128)
  W_plrY5 data (128)
  angleQ1 data (64)
  radiusQ1 data (64)
  W_plrX6 data (64)
  W_plrY6 data (64)
  Packages
    WMDataBase
      <Variables: system 21, user 6, dependent 0>
    PolarGraphs
      <Variables: system 21, user 38, dependent 0>
  <Recreation>
  <GetHistory>
  <Procedure>


Try to load a binary wave:

>>> path = data_path('mac-double.ibw')
>>> d = igor.load(path)
Traceback (most recent call last):
   ...
IOError: final record too long; bad pxp file?
"""

import os.path

from igor import LOG


_this_dir = os.path.dirname(__file__)
_data_dir = os.path.join(_this_dir, 'data')

def data_path(filename):
    LOG.info('Testing igorpy compatibility {}\n'.format(filename))
    path = os.path.join(_data_dir, filename)
    return path
